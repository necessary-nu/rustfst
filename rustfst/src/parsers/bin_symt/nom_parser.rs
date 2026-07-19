use nom::combinator::verify;
use nom::multi::count;
use nom::IResult;
use std::hash::BuildHasher;

use crate::parsers::bin_fst::fst_header::OpenFstString;
use crate::parsers::nom_utils::NomCustomError;
use crate::parsers::{parse_bin_i32, parse_bin_i64};
use crate::parsers::{write_bin_i32, write_bin_i64};
use crate::{Label, SymbolTable};
use anyhow::Result;
use std::io::Write;

static SYMBOL_TABLE_MAGIC_NUMBER: i32 = 2_125_658_996;

fn parse_row_symt(i: &[u8]) -> IResult<&[u8], (i64, OpenFstString), NomCustomError<&[u8]>> {
    let (i, symbol) = OpenFstString::parse(i)?;
    let (i, key) = parse_bin_i64(i)?;
    Ok((i, (key, symbol)))
}

pub(crate) fn parse_symbol_table_bin(
    i: &[u8],
) -> IResult<&[u8], SymbolTable, NomCustomError<&[u8]>> {
    let (i, _magic_number) = verify(parse_bin_i32, |v| *v == SYMBOL_TABLE_MAGIC_NUMBER)(i)?;
    let (i, _name) = OpenFstString::parse(i)?;
    let (i, _available_key) = parse_bin_i64(i)?;
    let (i, num_symbols) = parse_bin_i64(i)?;
    let (i, pairs_idx_symbols) = count(parse_row_symt, num_symbols as usize)(i)?;

    // Preserve each symbol's explicit (possibly sparse) key rather than reassigning
    // sequential labels. OpenFST/HFST symbol tables may carry holes — e.g. when a
    // symbol is dropped from the alphabet, the remaining symbols keep their original
    // global numbers — and the FST transitions reference those exact labels, so the
    // table must round-trip them verbatim.
    let mut symt = SymbolTable::empty();
    for (key, symbol) in pairs_idx_symbols.into_iter() {
        symt.add_symbol_with_key(symbol, key as Label);
    }

    Ok((i, symt))
}

pub(crate) fn write_bin_symt<W: Write, H: BuildHasher>(
    file: &mut W,
    symt: &SymbolTable<H>,
) -> Result<()> {
    write_bin_i32(file, SYMBOL_TABLE_MAGIC_NUMBER)?;
    // `SymbolTable` carries no name, so any constant here is arbitrary — but
    // OpenFST serializes the table's own name and HFST constructs its tables
    // unnamed, so the empty string is what OpenFST-lineage tools emit and the
    // only choice that byte-matches their output. Readers (ours included)
    // parse and discard it.
    OpenFstString::new("").write(file)?;
    // First field is `available_key` (the next free label = max label + 1, which
    // `len()` reports). The second field MUST be the number of rows that follow,
    // i.e. the count of real symbols — which differs from `len()` for sparse tables
    // carrying empty placeholder slots (see `insert_at`). Writing `len()` here would
    // claim more rows than are emitted and desync every reader past this table.
    let num_symbols = symt.iter().count();
    write_bin_i64(file, symt.len() as i64)?;
    write_bin_i64(file, num_symbols as i64)?;
    for (label, symbol) in symt.iter() {
        OpenFstString::new(symbol).write(file)?;
        write_bin_i64(file, label as i64)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SymbolTable;

    // Regression: a symbol table with a hole (a dropped label) must round-trip
    // through write/parse without the row count desyncing the reader. The bug was
    // that `write_bin_symt` wrote `len()` (which counts the empty placeholder slot)
    // as the number of rows, while only the real rows were emitted — so the parser
    // over-read past the table into whatever followed it in the stream.
    #[test]
    fn sparse_symbol_table_round_trips_without_overread() {
        let mut symt = SymbolTable::empty();
        symt.add_symbol_with_key("<eps>", 0);
        symt.add_symbol_with_key("a", 1);
        symt.add_symbol_with_key("b", 2);
        symt.add_symbol_with_key("g", 4); // hole at label 3

        let mut buf = Vec::new();
        write_bin_symt(&mut buf, &symt).unwrap();
        // Bytes that follow the table in a real (multi-section) stream.
        let sentinel = [0xDE_u8, 0xAD, 0xBE, 0xEF];
        buf.extend_from_slice(&sentinel);

        let (rest, parsed) = parse_symbol_table_bin(&buf).unwrap();
        // The parser must stop exactly at the table boundary, leaving the trailing
        // bytes untouched — not consume into them.
        assert_eq!(rest, &sentinel);

        // The sparse label is preserved verbatim (transitions reference it).
        assert_eq!(parsed.get_label("g"), Some(4));
        assert_eq!(parsed.get_symbol(4), Some("g"));
        assert_eq!(parsed.get_label("a"), Some(1));
    }
}

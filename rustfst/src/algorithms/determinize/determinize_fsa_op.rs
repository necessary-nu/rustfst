use std::borrow::Borrow;
use std::collections::btree_map::Entry as EntryBTreeMap;
use std::collections::hash_map::Entry as EntryHashMap;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::Result;

use crate::algorithms::determinize::divisors::CommonDivisor;
use crate::algorithms::determinize::{
    DeterminizeElement, DeterminizeStateTable, DeterminizeStateTuple, DeterminizeTr, WeightedSubset,
};
use crate::algorithms::lazy::FstOp;
use crate::fst_properties::FstProperties;
use crate::fst_traits::Fst;
use crate::fx_hasher::FxBuildHasher;
use crate::semirings::{DivideType, WeaklyDivisibleSemiring, WeightQuantize};
use crate::{Label, Semiring, StateId, Tr, Trs, TrsVec};

#[derive(Debug)]
pub struct DeterminizeFsaOp<W, F, CD, B, BT>
where
    W: Semiring,
    F: Fst<W>,
    CD: CommonDivisor<W>,
    B: Borrow<F> + Debug,
    BT: Borrow<[W]> + Debug,
{
    fst: B,
    state_table: DeterminizeStateTable<W, BT>,
    delta: f32,
    ghost: PhantomData<(CD, F)>,
}

struct DeterminizeTrAccumulator<W: Semiring> {
    det_tr: DeterminizeTr<W>,
    dest_weights: HashMap<StateId, W, FxBuildHasher>,
    raw_pairs: Vec<DeterminizeElement<W>>,
}

impl<W, F, CD, B, BT> FstOp<W> for DeterminizeFsaOp<W, F, CD, B, BT>
where
    W: Semiring + WeaklyDivisibleSemiring + WeightQuantize,
    F: Fst<W>,
    CD: CommonDivisor<W>,
    B: Borrow<F> + Debug,
    BT: Borrow<[W]> + Debug + PartialEq,
{
    fn compute_start(&self) -> Result<Option<StateId>> {
        if let Some(start_state) = self.fst.borrow().start() {
            let elt = DeterminizeElement::new(start_state, W::one());
            let tuple = DeterminizeStateTuple {
                subset: WeightedSubset::from_vec(vec![elt]),
                filter_state: start_state,
            };
            return Ok(Some(self.find_state(&tuple)?));
        }
        Ok(None)
    }

    fn compute_trs(&self, state: StateId) -> Result<TrsVec<W>> {
        // GetLabelMap
        let mut label_map: BTreeMap<Label, DeterminizeTrAccumulator<W>> = BTreeMap::new();
        let mut expansion_elements = 0usize;
        let src_tuple = self.state_table.find_tuple(state);
        for src_elt in src_tuple.subset.iter() {
            for tr in self.fst.borrow().get_trs(src_elt.state)?.trs() {
                let r = src_elt.weight.times(&tr.weight)?;
                let accumulator = match label_map.entry(tr.ilabel) {
                    EntryBTreeMap::Occupied(entry) => entry.into_mut(),
                    EntryBTreeMap::Vacant(entry) => entry.insert(DeterminizeTrAccumulator {
                        det_tr: DeterminizeTr::from_tr(tr, 0),
                        dest_weights: HashMap::with_hasher(FxBuildHasher::default()),
                        raw_pairs: Vec::new(),
                    }),
                };
                if CD::MERGE_BEFORE_DIVISOR {
                    match accumulator.dest_weights.entry(tr.nextstate) {
                        EntryHashMap::Vacant(entry) => {
                            expansion_elements = expansion_elements.saturating_add(1);
                            self.state_table
                                .check_expansion_elements(expansion_elements)?;
                            entry.insert(r);
                        }
                        EntryHashMap::Occupied(mut entry) => {
                            entry.get_mut().plus_assign(&r)?;
                        }
                    }
                } else {
                    expansion_elements = expansion_elements.saturating_add(1);
                    self.state_table
                        .check_expansion_elements(expansion_elements)?;
                    accumulator
                        .raw_pairs
                        .push(DeterminizeElement::new(tr.nextstate, r));
                }
            }
        }

        let mut trs = vec![];
        for mut accumulator in label_map.into_values() {
            accumulator.det_tr.dest_tuple.subset.pairs = if CD::MERGE_BEFORE_DIVISOR {
                accumulator
                    .dest_weights
                    .into_iter()
                    .map(|(state, weight)| DeterminizeElement::new(state, weight))
                    .collect()
            } else {
                accumulator.raw_pairs
            };
            self.norm_tr(&mut accumulator.det_tr, CD::MERGE_BEFORE_DIVISOR)?;
            let det_tr = accumulator.det_tr;
            trs.push(Tr::new(
                det_tr.label,
                det_tr.label,
                det_tr.weight,
                self.find_state(&det_tr.dest_tuple)?,
            ));
        }

        Ok(TrsVec(Arc::new(trs)))
    }

    fn compute_final_weight(&self, state: StateId) -> Result<Option<W>> {
        let tuple = self.state_table.find_tuple(state);
        let mut final_weight = W::zero();
        for det_elt in tuple.subset.iter() {
            final_weight.plus_assign(
                det_elt.weight.times(
                    self.fst
                        .borrow()
                        .final_weight(det_elt.state)?
                        .unwrap_or_else(W::zero),
                )?,
            )?;
        }
        if final_weight.is_zero() {
            Ok(None)
        } else {
            Ok(Some(final_weight))
        }
    }

    fn properties(&self) -> FstProperties {
        // Properties are set for the DeterminizeFst object. DeterminizeFsa shouldn't be used directly
        FstProperties::empty()
    }
}

impl<W, F, CD, B, BT> DeterminizeFsaOp<W, F, CD, B, BT>
where
    W: Semiring + WeaklyDivisibleSemiring + WeightQuantize,
    F: Fst<W>,
    CD: CommonDivisor<W>,
    B: Borrow<F> + Debug,
    BT: Borrow<[W]> + Debug + PartialEq,
{
    pub fn new_with_subset_limit(
        fst: B,
        in_dist: Option<BT>,
        delta: f32,
        max_subset_elements: Option<usize>,
    ) -> Result<Self> {
        if !fst.borrow().properties().contains(FstProperties::ACCEPTOR) {
            bail!("DeterminizeFsaImpl : expected acceptor as argument");
        }
        Ok(Self {
            fst,
            state_table: DeterminizeStateTable::new(in_dist, max_subset_elements),
            delta,
            ghost: PhantomData,
        })
    }

    fn norm_tr(&self, det_tr: &mut DeterminizeTr<W>, already_merged: bool) -> Result<()> {
        det_tr
            .dest_tuple
            .subset
            .pairs
            .sort_by_key(|element| element.state);

        for dest_elt in det_tr.dest_tuple.subset.pairs.iter() {
            det_tr.weight = CD::common_divisor(&det_tr.weight, &dest_elt.weight)?;
        }

        if !already_merged {
            let mut new_pairs = HashMap::with_hasher(FxBuildHasher::default());
            for element in &mut det_tr.dest_tuple.subset.pairs {
                match new_pairs.entry(element.state) {
                    EntryHashMap::Vacant(entry) => {
                        entry.insert(element.clone());
                    }
                    EntryHashMap::Occupied(mut entry) => {
                        entry.get_mut().weight.plus_assign(&element.weight)?;
                    }
                }
            }
            det_tr.dest_tuple.subset.pairs = new_pairs.into_values().collect();
            det_tr
                .dest_tuple
                .subset
                .pairs
                .sort_by_key(|element| element.state);
        }

        // The default divisor can merge equal destinations during scanning,
        // avoiding the former raw Vec + stable sort + second HashMap. The final
        // sort keeps the state-table key canonical regardless of hash order.

        for dest_elt in det_tr.dest_tuple.subset.pairs.iter_mut() {
            dest_elt.weight = dest_elt
                .weight
                .divide(&det_tr.weight, DivideType::DivideLeft)?;
            dest_elt.weight.quantize_assign(self.delta)?;
        }

        Ok(())
    }

    fn find_state(&self, tuple: &DeterminizeStateTuple<W>) -> Result<StateId> {
        self.state_table.find_id_from_ref(tuple)
    }

    pub fn out_dist(self) -> Result<Vec<W>> {
        let out_dist = self.state_table.out_dist();
        out_dist
            .into_iter()
            .enumerate()
            .map(|(s, e)| {
                e.ok_or_else(|| format_err!("Outdist for state {} has not been computed", s))
            })
            .collect::<Result<Vec<_>>>()
    }
}

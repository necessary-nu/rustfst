use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::hash::{BuildHasher, Hash};
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, ensure, Context, Result};
use tempfile::{Builder as TempDirBuilder, TempDir};

use crate::algorithms::lazy::StateTable;
use crate::fx_hasher::FxBuildHasher;
use crate::parsers::SerializeBinary;
use crate::StateId;
use crate::NO_STATE_ID;

const INITIAL_SLOT_COUNT: u64 = 16;
const SLOT_BYTES: u64 = 16;
const MAX_LOAD_NUMERATOR: u64 = 7;
const MAX_LOAD_DENOMINATOR: u64 = 10;

/// Scratch and resident-memory policy for a compose state-pair interner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComposeStateStoreConfig {
    /// Accounting ceiling for the in-memory forward and reverse maps.
    ///
    /// Capacity accounting is conservative for compose's fixed-size state
    /// tuples, but does not include heap data owned by a generic tuple or the
    /// short-lived overlap while an allocation is migrated to scratch. Callers
    /// that need a process-wide allowance should retain safety headroom.
    pub memory_cap_bytes: u64,
    /// Parent beneath which an operation-owned scratch directory is created.
    pub scratch_dir: PathBuf,
}

impl ComposeStateStoreConfig {
    pub fn new(memory_cap_bytes: u64, scratch_dir: impl Into<PathBuf>) -> Self {
        Self {
            memory_cap_bytes,
            scratch_dir: scratch_dir.into(),
        }
    }
}

#[derive(Clone)]
pub(crate) enum ComposeStateStore<T: Hash + Eq + Clone> {
    Memory(StateTable<T>),
    Spillable(Arc<Mutex<SpillableStateStore<T>>>),
}

impl<T: Hash + Eq + Clone> Default for ComposeStateStore<T> {
    fn default() -> Self {
        Self::Memory(StateTable::new())
    }
}

impl<T: Hash + Eq + Clone + fmt::Debug> fmt::Debug for ComposeStateStore<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory(table) => formatter.debug_tuple("Memory").field(table).finish(),
            Self::Spillable(store) => formatter.debug_tuple("Spillable").field(store).finish(),
        }
    }
}

impl<T: Hash + Eq + Clone> ComposeStateStore<T> {
    pub(crate) fn memory() -> Self {
        Self::default()
    }

    pub(crate) fn spillable(config: ComposeStateStoreConfig, codec: StateCodec<T>) -> Self {
        Self::Spillable(Arc::new(Mutex::new(SpillableStateStore::new(
            config, codec,
        ))))
    }

    pub(crate) fn intern(&self, tuple: T) -> Result<StateId> {
        match self {
            Self::Memory(table) => Ok(table.find_id(tuple)),
            Self::Spillable(store) => store
                .lock()
                .map_err(|error| anyhow::anyhow!("compose state store lock poisoned: {error}"))?
                .intern(tuple),
        }
    }

    pub(crate) fn resolve(&self, state: StateId) -> Result<T> {
        match self {
            Self::Memory(table) => Ok(table.find_tuple(state)),
            Self::Spillable(store) => store
                .lock()
                .map_err(|error| anyhow::anyhow!("compose state store lock poisoned: {error}"))?
                .resolve(state),
        }
    }

    pub(crate) fn is_spilled(&self) -> Result<bool> {
        match self {
            Self::Memory(_) => Ok(false),
            Self::Spillable(store) => Ok(store
                .lock()
                .map_err(|error| anyhow::anyhow!("compose state store lock poisoned: {error}"))?
                .disk
                .is_some()),
        }
    }

    pub(crate) fn scratch_path(&self) -> Result<Option<PathBuf>> {
        match self {
            Self::Memory(_) => Ok(None),
            Self::Spillable(store) => Ok(store
                .lock()
                .map_err(|error| anyhow::anyhow!("compose state store lock poisoned: {error}"))?
                .disk
                .as_ref()
                .map(|disk| disk.scratch.path().to_path_buf())),
        }
    }

    pub(crate) fn write_binary<W: Write>(&self, writer: &mut W) -> Result<()>
    where
        T: SerializeBinary,
    {
        match self {
            Self::Memory(table) => table.write_binary(writer),
            Self::Spillable(_) => {
                bail!("serializing a spillable compose state store is not supported")
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StateCodec<T> {
    encode: fn(&T) -> Result<Vec<u8>>,
    decode: fn(&[u8]) -> Result<T>,
}

impl<T> StateCodec<T> {
    pub(crate) fn serializable() -> Self
    where
        T: SerializeBinary,
    {
        Self {
            encode: encode_serializable::<T>,
            decode: decode_serializable::<T>,
        }
    }
}

impl<T> fmt::Debug for StateCodec<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("StateCodec")
    }
}

fn encode_serializable<T: SerializeBinary>(tuple: &T) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    tuple.write_binary(&mut bytes)?;
    Ok(bytes)
}

fn decode_serializable<T: SerializeBinary>(bytes: &[u8]) -> Result<T> {
    let (remaining, tuple) = T::parse_binary(bytes)
        .map_err(|error| anyhow::anyhow!("decoding compose state tuple: {error:?}"))?;
    ensure!(
        remaining.is_empty(),
        "compose state tuple decoder left {} trailing bytes",
        remaining.len()
    );
    Ok(tuple)
}

#[derive(Debug)]
pub(crate) struct SpillableStateStore<T: Hash + Eq + Clone> {
    config: ComposeStateStoreConfig,
    codec: StateCodec<T>,
    memory: Option<MemoryStateStore<T>>,
    disk: Option<DiskStateStore<T>>,
    last_resolved: Option<(StateId, T)>,
    poisoned: Option<String>,
}

impl<T: Hash + Eq + Clone> SpillableStateStore<T> {
    fn new(config: ComposeStateStoreConfig, codec: StateCodec<T>) -> Self {
        Self {
            config,
            codec,
            memory: Some(MemoryStateStore::default()),
            disk: None,
            last_resolved: None,
            poisoned: None,
        }
    }

    fn intern(&mut self, tuple: T) -> Result<StateId> {
        self.ensure_healthy()?;
        if self.disk.is_some() {
            return self.intern_disk(tuple);
        }

        let memory = self
            .memory
            .as_mut()
            .context("compose state store has neither memory nor disk backing")?;
        if let Some(state) = memory.tuple_to_id.get(&tuple) {
            return Ok(*state);
        }

        if !memory.reserve_one_within(self.config.memory_cap_bytes)? {
            self.spill()?;
            return self.intern_disk(tuple);
        }

        let state = state_id_for_len(memory.id_to_tuple.len())?;
        memory.tuple_to_id.insert(tuple.clone(), state);
        memory.id_to_tuple.push(tuple);
        Ok(state)
    }

    fn intern_disk(&mut self, tuple: T) -> Result<StateId> {
        let result = self
            .disk
            .as_mut()
            .context("compose state store spill did not install disk backing")?
            .intern(tuple);
        if let Err(error) = &result {
            self.poisoned = Some(format!("{error:#}"));
        }
        result
    }

    fn resolve(&mut self, state: StateId) -> Result<T> {
        self.ensure_healthy()?;
        if let Some((cached_state, tuple)) = &self.last_resolved {
            if *cached_state == state {
                return Ok(tuple.clone());
            }
        }
        if let Some(disk) = &mut self.disk {
            let result = disk.resolve(state);
            match result {
                Ok(tuple) => {
                    self.last_resolved = Some((state, tuple.clone()));
                    return Ok(tuple);
                }
                Err(error) => {
                    self.poisoned = Some(format!("{error:#}"));
                    return Err(error);
                }
            }
        }
        self.memory
            .as_ref()
            .context("compose state store has neither memory nor disk backing")?
            .id_to_tuple
            .get(state as usize)
            .cloned()
            .with_context(|| format!("compose state id {state} is absent"))
    }

    fn ensure_healthy(&self) -> Result<()> {
        if let Some(error) = &self.poisoned {
            bail!("compose state scratch is unusable after an earlier I/O failure: {error}")
        }
        Ok(())
    }

    fn spill(&mut self) -> Result<()> {
        let memory = self
            .memory
            .as_ref()
            .context("compose state store cannot spill without memory backing")?;
        let disk =
            DiskStateStore::from_memory(&self.config.scratch_dir, memory, self.codec.clone())?;
        self.disk = Some(disk);
        self.memory = None;
        Ok(())
    }
}

#[derive(Debug)]
struct MemoryStateStore<T: Hash + Eq + Clone> {
    tuple_to_id: HashMap<T, StateId, FxBuildHasher>,
    id_to_tuple: Vec<T>,
}

impl<T: Hash + Eq + Clone> Default for MemoryStateStore<T> {
    fn default() -> Self {
        Self {
            tuple_to_id: HashMap::default(),
            id_to_tuple: Vec::new(),
        }
    }
}

impl<T: Hash + Eq + Clone> MemoryStateStore<T> {
    fn reserve_one_within(&mut self, cap: u64) -> Result<bool> {
        let prospective_map =
            prospective_hash_capacity(self.tuple_to_id.len(), self.tuple_to_id.capacity());
        let prospective_reverse =
            prospective_reverse_capacity(self.id_to_tuple.len(), self.id_to_tuple.capacity());
        if estimated_memory_bytes::<T>(prospective_map, prospective_reverse)? > cap {
            return Ok(false);
        }

        self.tuple_to_id
            .try_reserve(1)
            .context("reserving compose pair hash table")?;
        self.id_to_tuple
            .try_reserve_exact(1)
            .context("reserving compose pair reverse table")?;
        Ok(
            estimated_memory_bytes::<T>(self.tuple_to_id.capacity(), self.id_to_tuple.capacity())?
                <= cap,
        )
    }
}

fn prospective_hash_capacity(len: usize, current: usize) -> usize {
    if len < current {
        current
    } else if current == 0 {
        3
    } else {
        // HashMap's current implementation grows through 3, 7, 14, ...
        // advertised entries. The extra entry keeps this prediction
        // conservative at growth boundaries without charging a resize when
        // the existing allocation still has room.
        current.saturating_mul(2).saturating_add(1)
    }
}

fn prospective_reverse_capacity(len: usize, current: usize) -> usize {
    if len < current {
        current
    } else {
        // The reverse table uses try_reserve_exact, so request only the next
        // logical slot. Actual allocator over-allocation is checked after the
        // reservation and triggers migration before the tuple is inserted.
        len.saturating_add(1)
    }
}

fn estimated_memory_bytes<T>(map_capacity: usize, reverse_capacity: usize) -> Result<u64> {
    // HashMap's raw control bytes and bucket alignment are private. Charging
    // sixteen bytes beyond the key/value payload per advertised entry is a
    // conservative bound for the supported state tuple layouts.
    let bucket_bytes = size_of::<T>()
        .checked_add(size_of::<StateId>())
        .and_then(|bytes| bytes.checked_add(16))
        .context("compose pair bucket-size accounting overflow")?;
    bytes_for(map_capacity, bucket_bytes)?
        .checked_add(bytes_for(reverse_capacity, size_of::<T>())?)
        .context("compose pair memory accounting overflow")
}

fn bytes_for(capacity: usize, element_bytes: usize) -> Result<u64> {
    u64::try_from(capacity)
        .context("compose pair capacity does not fit u64")?
        .checked_mul(u64::try_from(element_bytes).context("element size does not fit u64")?)
        .context("compose pair capacity accounting overflow")
}

fn state_id_for_len(len: usize) -> Result<StateId> {
    let state = StateId::try_from(len).context("compose state count exceeds StateId capacity")?;
    ensure!(
        state != NO_STATE_ID,
        "compose state count reached NO_STATE_ID"
    );
    Ok(state)
}

fn initial_slot_count(len: usize) -> Result<u64> {
    let len = u64::try_from(len).context("compose state count does not fit u64")?;
    let required = u128::from(len)
        .checked_mul(u128::from(MAX_LOAD_DENOMINATOR))
        .and_then(|scaled| scaled.checked_add(u128::from(MAX_LOAD_NUMERATOR - 1)))
        .map(|scaled| scaled / u128::from(MAX_LOAD_NUMERATOR))
        .context("compose initial hash capacity accounting overflow")?;
    let required = u64::try_from(required).context("compose initial hash capacity exceeds u64")?;
    required
        .max(INITIAL_SLOT_COUNT)
        .checked_next_power_of_two()
        .context("compose initial hash slot count overflow")
}

#[derive(Debug)]
struct DiskStateStore<T: Hash + Eq + Clone> {
    tuples: File,
    tuples_path: PathBuf,
    offsets: File,
    offsets_path: PathBuf,
    slots: File,
    slots_path: PathBuf,
    slot_count: u64,
    slot_generation: u64,
    len: u64,
    codec: StateCodec<T>,
    marker: PhantomData<T>,
    // Declared last so open files are dropped before TempDir removes the tree.
    scratch: TempDir,
}

impl<T: Hash + Eq + Clone> DiskStateStore<T> {
    fn from_memory(
        parent: &Path,
        memory: &MemoryStateStore<T>,
        codec: StateCodec<T>,
    ) -> Result<Self> {
        let scratch = TempDirBuilder::new()
            .prefix(".rustfst-compose-state-table.")
            .suffix(".scratch")
            .tempdir_in(parent)
            .with_context(|| {
                format!(
                    "creating rustfst compose state scratch beneath {}",
                    parent.display()
                )
            })?;
        let tuples_path = scratch.path().join("tuples.bin");
        let offsets_path = scratch.path().join("offsets.bin");
        let slots_path = scratch.path().join("slots-0.bin");
        let tuples = create_file(&tuples_path)?;
        let offsets = create_file(&offsets_path)?;
        let slots = create_file(&slots_path)?;
        let slot_count = initial_slot_count(memory.id_to_tuple.len())?;
        slots
            .set_len(
                slot_count
                    .checked_mul(SLOT_BYTES)
                    .context("compose initial hash file size overflow")?,
            )
            .with_context(|| format!("sizing compose hash slots {}", slots_path.display()))?;

        let mut disk = Self {
            tuples,
            tuples_path,
            offsets,
            offsets_path,
            slots,
            slots_path,
            slot_count,
            slot_generation: 0,
            len: 0,
            codec,
            marker: PhantomData,
            scratch,
        };
        for (index, tuple) in memory.id_to_tuple.iter().enumerate() {
            let expected = state_id_for_len(index)?;
            let actual = disk.insert_new(tuple)?;
            ensure!(
                actual == expected,
                "compose state spill changed id {expected} to {actual}"
            );
        }
        disk.flush()?;
        Ok(disk)
    }

    fn intern(&mut self, tuple: T) -> Result<StateId> {
        let hash = hash_tuple(&tuple);
        let mut vacant = match self.lookup(&tuple, hash)? {
            Lookup::Found(state) => return Ok(state),
            Lookup::Vacant(slot) => slot,
        };
        if self.needs_growth()? {
            self.grow_slots()?;
            vacant = match self.lookup(&tuple, hash)? {
                Lookup::Found(state) => return Ok(state),
                Lookup::Vacant(slot) => slot,
            };
        }
        self.insert_at(tuple, hash, vacant)
    }

    fn insert_new(&mut self, tuple: &T) -> Result<StateId> {
        let hash = hash_tuple(tuple);
        let mut vacant = match self.lookup(tuple, hash)? {
            Lookup::Found(state) => return Ok(state),
            Lookup::Vacant(slot) => slot,
        };
        if self.needs_growth()? {
            self.grow_slots()?;
            vacant = match self.lookup(tuple, hash)? {
                Lookup::Found(state) => return Ok(state),
                Lookup::Vacant(slot) => slot,
            };
        }
        self.insert_at(tuple.clone(), hash, vacant)
    }

    fn needs_growth(&self) -> Result<bool> {
        let scaled_len = self
            .len
            .checked_add(1)
            .and_then(|len| len.checked_mul(MAX_LOAD_DENOMINATOR))
            .context("compose hash load accounting overflow")?;
        let scaled_slots = self
            .slot_count
            .checked_mul(MAX_LOAD_NUMERATOR)
            .context("compose hash capacity accounting overflow")?;
        Ok(scaled_len > scaled_slots)
    }

    fn insert_at(&mut self, tuple: T, hash: u64, vacant: u64) -> Result<StateId> {
        let state =
            StateId::try_from(self.len).context("compose state count exceeds StateId capacity")?;
        ensure!(
            state != NO_STATE_ID,
            "compose state count reached NO_STATE_ID"
        );
        self.append_reverse(&tuple)?;
        let id_plus_one = self
            .len
            .checked_add(1)
            .context("compose scratch state id overflow")?;
        self.write_slot(vacant, hash, id_plus_one)?;
        self.len = id_plus_one;
        Ok(state)
    }

    fn resolve(&mut self, state: StateId) -> Result<T> {
        let id = state as u64;
        ensure!(id < self.len, "compose state id {state} is absent");
        let offset_position = id
            .checked_mul(8)
            .context("compose reverse-offset position overflow")?;
        self.offsets
            .seek(SeekFrom::Start(offset_position))
            .with_context(|| format!("seeking compose offsets {}", self.offsets_path.display()))?;
        let tuple_offset = read_u64(&mut self.offsets, &self.offsets_path, "tuple offset")?;
        self.tuples
            .seek(SeekFrom::Start(tuple_offset))
            .with_context(|| format!("seeking compose tuples {}", self.tuples_path.display()))?;
        let byte_count = read_u64(&mut self.tuples, &self.tuples_path, "tuple length")?;
        let byte_count = usize::try_from(byte_count).context("compose tuple is too large")?;
        let mut bytes = vec![0; byte_count];
        self.tuples.read_exact(&mut bytes).with_context(|| {
            format!("reading compose tuple from {}", self.tuples_path.display())
        })?;
        (self.codec.decode)(&bytes)
    }

    fn lookup(&mut self, tuple: &T, hash: u64) -> Result<Lookup> {
        let mut slot = hash & (self.slot_count - 1);
        for _ in 0..self.slot_count {
            let (stored_hash, id_plus_one) = self.read_slot(slot)?;
            if id_plus_one == 0 {
                return Ok(Lookup::Vacant(slot));
            }
            if stored_hash == hash {
                let raw_id = id_plus_one - 1;
                let state = StateId::try_from(raw_id)
                    .context("compose scratch state id exceeds StateId capacity")?;
                if self.resolve(state)? == *tuple {
                    return Ok(Lookup::Found(state));
                }
            }
            slot = (slot + 1) & (self.slot_count - 1);
        }
        bail!("compose state hash table has no vacant slot")
    }

    fn append_reverse(&mut self, tuple: &T) -> Result<()> {
        let bytes = (self.codec.encode)(tuple)?;
        let byte_count = u64::try_from(bytes.len()).context("compose tuple is too large")?;
        let mut record = Vec::with_capacity(8 + bytes.len());
        record.extend_from_slice(&byte_count.to_le_bytes());
        record.extend_from_slice(&bytes);
        let tuple_offset = self
            .tuples
            .seek(SeekFrom::End(0))
            .with_context(|| format!("seeking compose tuples {}", self.tuples_path.display()))?;
        self.offsets
            .seek(SeekFrom::End(0))
            .with_context(|| format!("seeking compose offsets {}", self.offsets_path.display()))?;
        self.offsets
            .write_all(&tuple_offset.to_le_bytes())
            .with_context(|| format!("writing compose offsets {}", self.offsets_path.display()))?;
        self.tuples
            .write_all(&record)
            .with_context(|| format!("writing compose tuple {}", self.tuples_path.display()))?;
        Ok(())
    }

    fn read_slot(&mut self, slot: u64) -> Result<(u64, u64)> {
        read_slot_from(&mut self.slots, &self.slots_path, slot)
    }

    fn write_slot(&mut self, slot: u64, hash: u64, id_plus_one: u64) -> Result<()> {
        write_slot_to(&mut self.slots, &self.slots_path, slot, hash, id_plus_one)
    }

    fn grow_slots(&mut self) -> Result<()> {
        let new_count = self
            .slot_count
            .checked_mul(2)
            .context("compose hash slot count overflow")?;
        let new_generation = self
            .slot_generation
            .checked_add(1)
            .context("compose hash generation overflow")?;
        let new_path = self
            .scratch
            .path()
            .join(format!("slots-{new_generation}.bin"));
        let mut new_file = create_file(&new_path)?;
        new_file
            .set_len(
                new_count
                    .checked_mul(SLOT_BYTES)
                    .context("compose hash file size overflow")?,
            )
            .with_context(|| format!("sizing compose hash slots {}", new_path.display()))?;

        self.slots
            .seek(SeekFrom::Start(0))
            .with_context(|| format!("seeking compose hash slots {}", self.slots_path.display()))?;
        for _ in 0..self.slot_count {
            let (hash, id_plus_one) = read_slot_next(&mut self.slots, &self.slots_path)?;
            if id_plus_one == 0 {
                continue;
            }
            place_raw_slot(&mut new_file, &new_path, new_count, hash, id_plus_one)?;
        }
        new_file
            .sync_data()
            .with_context(|| format!("flushing compose hash slots {}", new_path.display()))?;

        let old_path = std::mem::replace(&mut self.slots_path, new_path);
        let old_file = std::mem::replace(&mut self.slots, new_file);
        self.slot_count = new_count;
        self.slot_generation = new_generation;
        drop(old_file);
        fs::remove_file(&old_path)
            .with_context(|| format!("removing old compose hash slots {}", old_path.display()))?;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.tuples
            .flush()
            .with_context(|| format!("flushing compose tuples {}", self.tuples_path.display()))?;
        self.offsets
            .flush()
            .with_context(|| format!("flushing compose offsets {}", self.offsets_path.display()))?;
        self.slots
            .flush()
            .with_context(|| format!("flushing compose slots {}", self.slots_path.display()))?;
        Ok(())
    }
}

#[derive(Debug)]
enum Lookup {
    Found(StateId),
    Vacant(u64),
}

fn create_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating compose state scratch file {}", path.display()))
}

fn hash_tuple<T: Hash>(tuple: &T) -> u64 {
    FxBuildHasher::default().hash_one(tuple)
}

fn read_slot_from(file: &mut File, path: &Path, slot: u64) -> Result<(u64, u64)> {
    let position = slot
        .checked_mul(SLOT_BYTES)
        .context("compose hash slot offset overflow")?;
    file.seek(SeekFrom::Start(position))
        .with_context(|| format!("seeking compose hash slots {}", path.display()))?;
    read_slot_next(file, path)
}

fn read_slot_next(file: &mut File, path: &Path) -> Result<(u64, u64)> {
    let mut record = [0; SLOT_BYTES as usize];
    file.read_exact(&mut record)
        .with_context(|| format!("reading compose hash slot from {}", path.display()))?;
    let mut hash = [0; 8];
    hash.copy_from_slice(&record[..8]);
    let mut id_plus_one = [0; 8];
    id_plus_one.copy_from_slice(&record[8..]);
    Ok((u64::from_le_bytes(hash), u64::from_le_bytes(id_plus_one)))
}

fn write_slot_to(
    file: &mut File,
    path: &Path,
    slot: u64,
    hash: u64,
    id_plus_one: u64,
) -> Result<()> {
    let position = slot
        .checked_mul(SLOT_BYTES)
        .context("compose hash slot offset overflow")?;
    file.seek(SeekFrom::Start(position))
        .with_context(|| format!("seeking compose hash slots {}", path.display()))?;
    let mut record = [0; SLOT_BYTES as usize];
    record[..8].copy_from_slice(&hash.to_le_bytes());
    record[8..].copy_from_slice(&id_plus_one.to_le_bytes());
    file.write_all(&record)
        .with_context(|| format!("writing compose hash slots {}", path.display()))?;
    Ok(())
}

fn place_raw_slot(
    file: &mut File,
    path: &Path,
    slot_count: u64,
    hash: u64,
    id_plus_one: u64,
) -> Result<()> {
    let mut slot = hash & (slot_count - 1);
    for _ in 0..slot_count {
        let (_, occupied) = read_slot_from(file, path, slot)?;
        if occupied == 0 {
            return write_slot_to(file, path, slot, hash, id_plus_one);
        }
        slot = (slot + 1) & (slot_count - 1);
    }
    bail!("resized compose state hash table has no vacant slot")
}

fn read_u64(file: &mut File, path: &Path, field: &str) -> Result<u64> {
    let mut bytes = [0; 8];
    file.read_exact(&mut bytes)
        .with_context(|| format!("reading compose {field} from {}", path.display()))?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::compose::compose_filters::SequenceComposeFilterBuilder;
    use crate::algorithms::compose::filter_states::{FilterState, IntegerFilterState};
    use crate::algorithms::compose::matchers::GenericMatcher;
    use crate::algorithms::compose::{
        ComposeFst, ComposeFstOpOptions, ComposeFstOpState, ComposeStateTuple,
    };
    use crate::algorithms::lazy::NullCache;
    use crate::fst_impls::VectorFst;
    use crate::fst_traits::MutableFst;
    use crate::semirings::{Semiring, TropicalWeight};
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;

    type Tuple = ComposeStateTuple<IntegerFilterState>;
    type TestFst = VectorFst<TropicalWeight>;
    type TestBorrow = Arc<TestFst>;
    type TestMatcher = GenericMatcher<TropicalWeight, TestFst, TestBorrow>;
    type TestFilterBuilder = SequenceComposeFilterBuilder<
        TropicalWeight,
        TestFst,
        TestFst,
        TestBorrow,
        TestBorrow,
        TestMatcher,
        TestMatcher,
    >;
    type TestCompose = ComposeFst<
        TropicalWeight,
        TestFst,
        TestFst,
        TestBorrow,
        TestBorrow,
        TestMatcher,
        TestMatcher,
        TestFilterBuilder,
        NullCache<TropicalWeight>,
    >;

    fn tuple(filter: StateId, left: StateId, right: StateId) -> Tuple {
        ComposeStateTuple {
            fs: IntegerFilterState::new(filter),
            s1: left,
            s2: right,
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct CollidingTuple(u64);

    impl Hash for CollidingTuple {
        fn hash<H: Hasher>(&self, state: &mut H) {
            0_u8.hash(state);
        }
    }

    fn encode_colliding(tuple: &CollidingTuple) -> Result<Vec<u8>> {
        Ok(tuple.0.to_le_bytes().to_vec())
    }

    fn decode_colliding(bytes: &[u8]) -> Result<CollidingTuple> {
        ensure!(bytes.len() == 8, "invalid colliding tuple length");
        let mut value = [0; 8];
        value.copy_from_slice(bytes);
        Ok(CollidingTuple(u64::from_le_bytes(value)))
    }

    fn fail_encode(_: &CollidingTuple) -> Result<Vec<u8>> {
        Err(std::io::Error::new(
            std::io::ErrorKind::WriteZero,
            "injected compose scratch write failure",
        )
        .into())
    }

    fn colliding_codec() -> StateCodec<CollidingTuple> {
        StateCodec {
            encode: encode_colliding,
            decode: decode_colliding,
        }
    }

    fn compose_operands() -> Result<(TestBorrow, TestBorrow)> {
        let mut left = TestFst::new();
        left.add_states(3);
        left.set_start(0)?;
        left.emplace_tr(0, 10, 1, TropicalWeight::one(), 1)?;
        left.emplace_tr(0, 11, 2, TropicalWeight::one(), 2)?;
        left.emplace_tr(1, 12, 3, TropicalWeight::one(), 2)?;
        left.set_final(2, TropicalWeight::one())?;

        let mut right = TestFst::new();
        right.add_states(3);
        right.set_start(0)?;
        right.emplace_tr(0, 1, 20, TropicalWeight::one(), 1)?;
        right.emplace_tr(0, 2, 21, TropicalWeight::one(), 2)?;
        right.emplace_tr(1, 3, 22, TropicalWeight::one(), 2)?;
        right.set_final(2, TropicalWeight::one())?;
        Ok((Arc::new(left), Arc::new(right)))
    }

    fn compose_with_state(
        left: TestBorrow,
        right: TestBorrow,
        op_state: ComposeFstOpState<Tuple>,
    ) -> Result<TestFst> {
        let options: ComposeFstOpOptions<
            TestMatcher,
            TestMatcher,
            TestFilterBuilder,
            ComposeFstOpState<Tuple>,
        > = ComposeFstOpOptions::new(None, None, None, Some(op_state));
        TestCompose::new_with_options(left, right, options)?.compute()
    }

    #[test]
    fn zero_cap_spills_both_directions_and_preserves_dense_ids() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let store = ComposeStateStore::spillable(
            ComposeStateStoreConfig::new(0, parent.path()),
            StateCodec::serializable(),
        );
        let first = tuple(0, 10, 20);
        let second = tuple(1, 11, 21);

        assert_eq!(store.intern(first.clone())?, 0);
        assert!(store.is_spilled()?);
        assert_eq!(store.intern(first.clone())?, 0);
        assert_eq!(store.intern(second.clone())?, 1);
        assert_eq!(store.resolve(0)?, first);
        assert_eq!(store.resolve(1)?, second);

        let scratch = store.scratch_path()?.context("missing scratch path")?;
        let clone = store.clone();
        drop(store);
        assert!(
            scratch.exists(),
            "a shared clone still owns the scratch tree"
        );
        drop(clone);
        assert!(!scratch.exists(), "the last owner must remove scratch");
        Ok(())
    }

    #[test]
    fn forced_spill_compose_preserves_state_arc_and_final_order() -> Result<()> {
        let (left, right) = compose_operands()?;
        let resident = compose_with_state(
            Arc::clone(&left),
            Arc::clone(&right),
            ComposeFstOpState::new(),
        )?;

        let parent = tempfile::tempdir()?;
        let spill_state =
            ComposeFstOpState::new_spillable(ComposeStateStoreConfig::new(0, parent.path()))?;
        let observer = spill_state.clone();
        let spilled = compose_with_state(left, right, spill_state)?;
        assert!(observer.is_spilled()?);
        assert_eq!(resident, spilled);

        let scratch = observer.scratch_path()?.context("missing scratch path")?;
        assert!(scratch.exists());
        drop(observer);
        assert!(!scratch.exists());
        Ok(())
    }

    #[test]
    fn spilled_hash_table_grows_without_changing_ids() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let store = ComposeStateStore::spillable(
            ComposeStateStoreConfig::new(0, parent.path()),
            StateCodec::serializable(),
        );
        for state in 0..200 {
            let value = tuple(state % 3, state, state + 1);
            assert_eq!(store.intern(value)?, state);
        }
        for state in 0..200 {
            assert_eq!(store.resolve(state)?, tuple(state % 3, state, state + 1));
        }
        Ok(())
    }

    #[test]
    fn spilled_hash_table_resolves_true_hash_collisions() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let store = ComposeStateStore::spillable(
            ComposeStateStoreConfig::new(0, parent.path()),
            colliding_codec(),
        );

        for state in 0..64_u64 {
            assert_eq!(store.intern(CollidingTuple(state))?, state as StateId);
        }
        for state in (0..64_u64).rev() {
            assert_eq!(store.intern(CollidingTuple(state))?, state as StateId);
            assert_eq!(store.resolve(state as StateId)?, CollidingTuple(state));
        }
        Ok(())
    }

    #[test]
    fn migration_presizes_slots_for_resident_population() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let store = ComposeStateStore::spillable(
            ComposeStateStoreConfig::new(u64::MAX, parent.path()),
            StateCodec::serializable(),
        );
        for state in 0..100 {
            assert_eq!(store.intern(tuple(0, state, state + 1))?, state);
        }

        let shared = match &store {
            ComposeStateStore::Spillable(shared) => shared,
            ComposeStateStore::Memory(_) => unreachable!("test constructed a spillable store"),
        };
        shared
            .lock()
            .map_err(|error| anyhow::anyhow!("test store lock poisoned: {error}"))?
            .config
            .memory_cap_bytes = 0;

        assert_eq!(store.intern(tuple(0, 100, 101))?, 100);
        let guard = shared
            .lock()
            .map_err(|error| anyhow::anyhow!("test store lock poisoned: {error}"))?;
        let disk = guard.disk.as_ref().context("store did not spill")?;
        assert_eq!(disk.slot_count, initial_slot_count(100)?);
        assert_eq!(disk.slot_generation, 0, "migration must not rehash");
        drop(guard);
        assert_eq!(store.resolve(0)?, tuple(0, 0, 1));
        assert_eq!(store.resolve(100)?, tuple(0, 100, 101));
        Ok(())
    }

    #[test]
    fn first_post_spill_error_poisons_store_and_cleans_scratch() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let store = ComposeStateStore::spillable(
            ComposeStateStoreConfig::new(0, parent.path()),
            StateCodec {
                encode: fail_encode,
                decode: decode_colliding,
            },
        );

        let first_error = store
            .intern(CollidingTuple(1))
            .expect_err("injected first disk insertion must fail");
        assert!(format!("{first_error:#}").contains("injected compose scratch write failure"));
        let scratch = store
            .scratch_path()?
            .context("spill did not create scratch")?;
        assert!(scratch.exists());

        let retry_error = store
            .intern(CollidingTuple(1))
            .expect_err("a failed disk mutation must poison the store");
        assert!(format!("{retry_error:#}").contains("unusable after an earlier I/O failure"));
        drop(store);
        assert!(
            !scratch.exists(),
            "poisoned scratch must be removed on drop"
        );
        Ok(())
    }

    #[test]
    fn scratch_read_error_poisoning_prevents_reuse() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let store = ComposeStateStore::spillable(
            ComposeStateStoreConfig::new(0, parent.path()),
            colliding_codec(),
        );
        assert_eq!(store.intern(CollidingTuple(1))?, 0);

        let shared = match &store {
            ComposeStateStore::Spillable(shared) => shared,
            ComposeStateStore::Memory(_) => unreachable!("test constructed a spillable store"),
        };
        shared
            .lock()
            .map_err(|error| anyhow::anyhow!("test store lock poisoned: {error}"))?
            .disk
            .as_mut()
            .context("store did not spill")?
            .slots
            .set_len(0)?;

        let read_error = store
            .intern(CollidingTuple(2))
            .expect_err("truncated slots must produce an I/O error");
        assert!(format!("{read_error:#}").contains("reading compose hash slot"));
        let poisoned = store
            .resolve(0)
            .expect_err("store must not be reused after scratch I/O failure");
        assert!(format!("{poisoned:#}").contains("unusable after an earlier I/O failure"));
        Ok(())
    }

    #[test]
    fn nonzero_cap_uses_existing_capacity_before_spilling() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let two_tuple_cap = estimated_memory_bytes::<Tuple>(3, 2)?;
        let store = ComposeStateStore::spillable(
            ComposeStateStoreConfig::new(two_tuple_cap, parent.path()),
            StateCodec::serializable(),
        );
        assert_eq!(store.intern(tuple(0, 0, 0))?, 0);
        assert_eq!(store.intern(tuple(0, 1, 1))?, 1);
        assert!(
            !store.is_spilled()?,
            "an insertion that fits existing hash capacity must stay resident"
        );
        assert_eq!(store.intern(tuple(0, 2, 2))?, 2);
        assert!(
            store.is_spilled()?,
            "the next real reverse-table growth must cross the exact cap"
        );

        let one_byte_short = ComposeStateStore::spillable(
            ComposeStateStoreConfig::new(two_tuple_cap - 1, parent.path()),
            StateCodec::serializable(),
        );
        assert_eq!(one_byte_short.intern(tuple(0, 0, 0))?, 0);
        assert_eq!(one_byte_short.intern(tuple(0, 1, 1))?, 1);
        assert!(one_byte_short.is_spilled()?);
        Ok(())
    }

    #[test]
    fn invalid_scratch_parent_is_a_fallible_insert_error() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let not_a_directory = parent.path().join("file");
        File::create(&not_a_directory)?;
        let store = ComposeStateStore::spillable(
            ComposeStateStoreConfig::new(0, &not_a_directory),
            StateCodec::serializable(),
        );

        let error = store
            .intern(tuple(0, 0, 0))
            .expect_err("a file cannot contain compose scratch");
        let message = format!("{error:#}");
        assert!(message.contains("creating rustfst compose state scratch"));
        assert!(message.contains("file"));
        assert_eq!(fs::read_dir(parent.path())?.count(), 1);
        Ok(())
    }

    #[test]
    fn spillable_store_rejects_op_state_serialization() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let store = ComposeStateStore::<Tuple>::spillable(
            ComposeStateStoreConfig::new(0, parent.path()),
            StateCodec::serializable(),
        );
        let mut output = Vec::new();
        let error = store
            .write_binary(&mut output)
            .expect_err("scratch-backed state cannot use the legacy snapshot format");
        assert!(error.to_string().contains("spillable compose state store"));
        Ok(())
    }
}

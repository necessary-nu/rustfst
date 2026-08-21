use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::fmt::Debug;
use std::iter::{repeat, Map, Repeat, Zip};
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use itertools::izip;

use crate::algorithms::lazy::cache::CacheStatus;
use crate::algorithms::lazy::fst_op::{AccessibleOpState, FstOp, SerializableOpState};
use crate::algorithms::lazy::{FstCache, SerializableCache};
use crate::fst_properties::FstProperties;
use crate::fst_traits::{
    AllocableFst, CoreFst, Fst, FstIterData, FstIterator, MutableFst, StateIterator,
};
use crate::semirings::{Semiring, SerializableSemiring};
use crate::{StateId, SymbolTable, Trs, TrsVec};

/// Lazy materialization stopped before writing more transitions than the caller
/// allowed.
///
/// Distinct from the state budget: one state of a materialized machine can carry
/// an unbounded number of transitions, so a state count is not a memory bound.
/// A caller that hits this limit learns that the operation's *output* is too
/// large, not that its search failed to converge — retrying with a strategy that
/// splits paths further (weight encoding, say) can only make it worse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComputeTrLimitExceeded {
    /// Maximum number of transitions the caller allowed.
    pub limit: usize,
    /// Number the materialization had already produced when it stopped.
    pub attempted: usize,
}

impl fmt::Display for ComputeTrLimitExceeded {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "compute_bounded: transition budget of {} exceeded (attempted {})",
            self.limit, self.attempted
        )
    }
}

impl Error for ComputeTrLimitExceeded {}

#[derive(Debug, Clone)]
pub struct LazyFst<W: Semiring, Op: FstOp<W>, Cache> {
    cache: Cache,
    pub(crate) op: Op,
    w: PhantomData<W>,
    isymt: Option<Arc<SymbolTable>>,
    osymt: Option<Arc<SymbolTable>>,
}

impl<W: Semiring, Op: FstOp<W>, Cache: FstCache<W>> CoreFst<W> for LazyFst<W, Op, Cache> {
    type TRS = TrsVec<W>;

    fn start(&self) -> Option<StateId> {
        match self.cache.get_start() {
            CacheStatus::Computed(start) => start,
            CacheStatus::NotComputed => {
                // TODO: Need to return a Result
                let start = self.op.compute_start().unwrap();
                self.cache.insert_start(start);
                start
            }
        }
    }

    fn final_weight(&self, state_id: StateId) -> Result<Option<W>> {
        match self.cache.get_final_weight(state_id) {
            CacheStatus::Computed(final_weight) => Ok(final_weight),
            CacheStatus::NotComputed => {
                let final_weight = self.op.compute_final_weight(state_id)?;
                self.cache
                    .insert_final_weight(state_id, final_weight.clone());
                Ok(final_weight)
            }
        }
    }

    unsafe fn final_weight_unchecked(&self, state_id: StateId) -> Option<W> {
        self.final_weight(state_id).unwrap_unchecked()
    }

    fn num_trs(&self, s: StateId) -> Result<usize> {
        self.cache
            .num_trs(s)
            .ok_or_else(|| format_err!("State {:?} doesn't exist", s))
    }

    unsafe fn num_trs_unchecked(&self, s: StateId) -> usize {
        self.cache.num_trs(s).unwrap_unchecked()
    }

    fn get_trs(&self, state_id: StateId) -> Result<Self::TRS> {
        match self.cache.get_trs(state_id) {
            CacheStatus::Computed(trs) => Ok(trs),
            CacheStatus::NotComputed => {
                let trs = self.op.compute_trs(state_id)?;
                self.cache.insert_trs(state_id, trs.shallow_clone());
                Ok(trs)
            }
        }
    }

    unsafe fn get_trs_unchecked(&self, state_id: StateId) -> Self::TRS {
        self.get_trs(state_id).unwrap_unchecked()
    }

    fn properties(&self) -> FstProperties {
        self.op.properties()
    }

    fn num_input_epsilons(&self, state: StateId) -> Result<usize> {
        self.cache
            .num_input_epsilons(state)
            .ok_or_else(|| format_err!("State {:?} doesn't exist", state))
    }

    fn num_output_epsilons(&self, state: StateId) -> Result<usize> {
        self.cache
            .num_output_epsilons(state)
            .ok_or_else(|| format_err!("State {:?} doesn't exist", state))
    }
}

impl<'a, W, Op, Cache> StateIterator<'a> for LazyFst<W, Op, Cache>
where
    W: Semiring,
    Op: FstOp<W> + 'a,
    Cache: FstCache<W> + 'a,
{
    type Iter = StatesIteratorLazyFst<'a, Self>;

    fn states_iter(&'a self) -> Self::Iter {
        self.start();
        StatesIteratorLazyFst { fst: self, s: 0 }
    }
}

#[derive(Clone)]
pub struct StatesIteratorLazyFst<'a, T> {
    pub(crate) fst: &'a T,
    pub(crate) s: StateId,
}

impl<W, Op, Cache> Iterator for StatesIteratorLazyFst<'_, LazyFst<W, Op, Cache>>
where
    W: Semiring,
    Op: FstOp<W>,
    Cache: FstCache<W>,
{
    type Item = StateId;

    fn next(&mut self) -> Option<Self::Item> {
        let num_known_states = self.fst.cache.num_known_states();
        if (self.s as usize) < num_known_states {
            let s_cur = self.s;
            // Force expansion of the state
            self.fst.get_trs(self.s).unwrap();
            self.s += 1;
            Some(s_cur)
        } else {
            None
        }
    }
}

type ZipIter<'a, W, Op, Cache, SELF> =
    Zip<<LazyFst<W, Op, Cache> as StateIterator<'a>>::Iter, Repeat<&'a SELF>>;
type MapFunction<'a, W, SELF, TRS> = Box<dyn FnMut((StateId, &'a SELF)) -> FstIterData<W, TRS>>;
type MapIter<'a, W, Op, Cache, SELF, TRS> =
    Map<ZipIter<'a, W, Op, Cache, SELF>, MapFunction<'a, W, SELF, TRS>>;

impl<'a, W, Op, Cache> FstIterator<'a, W> for LazyFst<W, Op, Cache>
where
    W: Semiring,
    Op: FstOp<W> + 'a,
    Cache: FstCache<W> + 'a,
{
    type FstIter = MapIter<'a, W, Op, Cache, Self, Self::TRS>;

    fn fst_iter(&'a self) -> Self::FstIter {
        let it = repeat(self);
        izip!(self.states_iter(), it).map(Box::new(|(state_id, p): (StateId, &'a Self)| {
            FstIterData {
                state_id,
                trs: unsafe { p.get_trs_unchecked(state_id) },
                final_weight: unsafe { p.final_weight_unchecked(state_id) },
                num_trs: unsafe { p.num_trs_unchecked(state_id) },
            }
        }))
    }
}

impl<W, Op, Cache> Fst<W> for LazyFst<W, Op, Cache>
where
    W: Semiring,
    Op: FstOp<W> + 'static,
    Cache: FstCache<W> + 'static,
{
    fn input_symbols(&self) -> Option<&Arc<SymbolTable>> {
        self.isymt.as_ref()
    }

    fn output_symbols(&self) -> Option<&Arc<SymbolTable>> {
        self.osymt.as_ref()
    }

    fn set_input_symbols(&mut self, symt: Arc<SymbolTable>) {
        self.isymt = Some(symt);
    }

    fn set_output_symbols(&mut self, symt: Arc<SymbolTable>) {
        self.osymt = Some(symt);
    }

    fn take_input_symbols(&mut self) -> Option<Arc<SymbolTable>> {
        self.isymt.take()
    }

    fn take_output_symbols(&mut self) -> Option<Arc<SymbolTable>> {
        self.osymt.take()
    }
}

impl<W, Op, Cache> LazyFst<W, Op, Cache>
where
    W: Semiring,
    Op: FstOp<W>,
    Cache: FstCache<W>,
{
    pub fn from_op_and_cache(
        op: Op,
        cache: Cache,
        isymt: Option<Arc<SymbolTable>>,
        osymt: Option<Arc<SymbolTable>>,
    ) -> Self {
        Self {
            op,
            cache,
            isymt,
            osymt,
            w: PhantomData,
        }
    }

    /// Turns the Lazy FST into a static one.
    pub fn compute<F2: MutableFst<W> + AllocableFst<W>>(&self) -> Result<F2> {
        self.compute_bounded(None, None)
    }

    /// Materialize the lazy FST into a static one, optionally aborting once more
    /// than `max_states` states or `max_trs` transitions have been produced.
    ///
    /// With both bounds `None` this is unbounded and behaves identically to
    /// [`Self::compute`]. When a bound is supplied and the on-demand expansion
    /// exceeds it, the computation returns an `Err` instead of running away.
    /// This is the escape hatch for lazy operations (e.g. weighted
    /// determinization of a non-twins cyclic FST) that may never terminate: the
    /// caller can catch the error and retry with a different strategy.
    ///
    /// The two bounds measure different things and neither implies the other. A
    /// state count bounds how many DFA states exist; a transition count bounds
    /// the machine that is actually written out. Determinizing a union whose
    /// operands differ sharply in density gives every surviving state the
    /// densest operand's out-degree, so a modest state count can carry orders of
    /// magnitude more transitions than the input had — memory follows the
    /// transitions, not the states.
    ///
    /// Both bounds count only what the BFS has already produced, so the same
    /// input that terminates unbounded produces byte-identical output as long as
    /// neither bound is tripped.
    pub fn compute_bounded<F2: MutableFst<W> + AllocableFst<W>>(
        &self,
        max_states: Option<usize>,
        max_trs: Option<usize>,
    ) -> Result<F2> {
        let start_state = self.start();
        let mut fst_out = F2::new();
        let start_state = match start_state {
            Some(s) => s,
            None => return Ok(fst_out),
        };
        fst_out.add_states(start_state as usize + 1);
        fst_out.set_start(start_state)?;
        let mut queue = VecDeque::new();
        let mut visited_states = vec![];
        visited_states.resize(start_state as usize + 1, false);
        visited_states[start_state as usize] = true;
        let mut num_discovered: usize = 1;
        let mut num_trs: usize = 0;
        queue.push_back(start_state);
        while let Some(s) = queue.pop_front() {
            let trs_owner = self.get_trs(s)?;
            if let Some(bound) = max_trs {
                num_trs = num_trs.saturating_add(trs_owner.trs().len());
                if num_trs > bound {
                    return Err(ComputeTrLimitExceeded {
                        limit: bound,
                        attempted: num_trs,
                    }
                    .into());
                }
            }
            for tr in trs_owner.trs() {
                if (tr.nextstate as usize) >= visited_states.len() {
                    visited_states.resize(tr.nextstate as usize + 1, false);
                }
                if !visited_states[tr.nextstate as usize] {
                    queue.push_back(tr.nextstate);
                    visited_states[tr.nextstate as usize] = true;
                    num_discovered += 1;
                    if let Some(bound) = max_states {
                        if num_discovered > bound {
                            bail!(
                                "compute_bounded: state budget of {} exceeded (lazy FST did not \
                                 converge)",
                                bound
                            );
                        }
                    }
                }
                let n = fst_out.num_states();
                if (tr.nextstate as usize) >= n {
                    fst_out.add_states(tr.nextstate as usize - n + 1)
                }
            }
            unsafe { fst_out.set_trs_unchecked(s, trs_owner.trs().to_vec()) };
            if let Some(f_w) = self.final_weight(s)? {
                fst_out.set_final(s, f_w)?;
            }
        }
        fst_out.set_properties(self.properties());

        if let Some(isymt) = &self.isymt {
            fst_out.set_input_symbols(Arc::clone(isymt));
        }
        if let Some(osymt) = &self.osymt {
            fst_out.set_output_symbols(Arc::clone(osymt));
        }
        Ok(fst_out)
    }
}

impl<W, Op, Cache> SerializableLazyFst for LazyFst<W, Op, Cache>
where
    W: SerializableSemiring,
    Op: FstOp<W> + AccessibleOpState,
    Op::FstOpState: SerializableOpState,
    Cache: FstCache<W> + SerializableCache,
{
    /// Writes LazyFst interal states to a directory of files in binary format.
    fn write<P: AsRef<Path>>(&self, cache_dir: P, op_state_dir: P) -> Result<()> {
        self.cache.write(cache_dir)?;
        self.op.get_op_state().write(op_state_dir)?;
        Ok(())
    }
}

pub trait SerializableLazyFst {
    /// Writes LazyFst interal states to a directory of files in binary format.
    fn write<P: AsRef<Path>>(&self, cache_dir: P, op_state_dir: P) -> Result<()>;
}

impl<C: SerializableLazyFst, CP: Deref<Target = C> + Debug> SerializableLazyFst for CP {
    fn write<P: AsRef<Path>>(&self, cache_dir: P, op_state_dir: P) -> Result<()> {
        self.deref().write(cache_dir, op_state_dir)
    }
}

use std::marker::PhantomData;

use crate::algorithms::lazy::{CacheStatus, FstCache};
use crate::semirings::Semiring;
use crate::{StateId, TrsVec};

/// A cache that stores nothing.
///
/// The one-shot static algorithms (`determinize`, `compose`) build a `LazyFst`,
/// materialize it exactly once on the calling thread via
/// [`compute_bounded`](crate::algorithms::lazy::LazyFst::compute_bounded), and
/// drop it. That materialization is a single BFS that visits every state once:
/// each state's trs and final weight are requested from the `LazyFst` exactly
/// once, then copied straight into the output FST. The op that backs these lazy
/// FSTs is self-contained — `DeterminizeFsaOp`/`ComposeFstOp` resolve every
/// `compute_trs`/`compute_final_weight`/`compute_start` through their own
/// state-table plus the *input* FST(s), never through the `LazyFst`'s cache —
/// so nothing ever reads a value back out of this cache. The store-and-clone
/// the shared caches perform on every insert (a `HashMap`/`Vec` write plus an
/// epsilon-count scan of every arc) is therefore pure overhead on these paths.
///
/// `NullCache` makes every `get_*` a miss (forcing the one recompute the BFS
/// already relies on) and every `insert_*` a no-op, dropping the store layer
/// entirely. It carries no interior mutability, so it is `Send + Sync` and
/// imposes no thread-safety change on the FSTs that use it.
///
/// This is **not** suitable for any lazy FST that a caller keeps and queries
/// repeatedly (every access would recompute from scratch, and the `num_*`
/// bookkeeping is inert). It is wired only into the one-shot static entry
/// points, whose single-touch materialization is what makes it correct.
#[derive(Debug)]
pub struct NullCache<W: Semiring> {
    w: PhantomData<W>,
}

// Hand-written so the bounds are `W: Semiring` only. The derived `Default`/
// `Clone` would spuriously require `W: Default`/`W: Clone` because of the
// `PhantomData<W>` field, even though a `NullCache` holds no `W`.
impl<W: Semiring> Default for NullCache<W> {
    fn default() -> Self {
        Self { w: PhantomData }
    }
}

impl<W: Semiring> Clone for NullCache<W> {
    fn clone(&self) -> Self {
        Self { w: PhantomData }
    }
}

impl<W: Semiring> FstCache<W> for NullCache<W> {
    fn get_start(&self) -> CacheStatus<Option<StateId>> {
        CacheStatus::NotComputed
    }

    fn insert_start(&self, _id: Option<StateId>) {}

    fn get_trs(&self, _id: StateId) -> CacheStatus<TrsVec<W>> {
        CacheStatus::NotComputed
    }

    fn insert_trs(&self, _id: StateId, _trs: TrsVec<W>) {}

    fn get_final_weight(&self, _id: StateId) -> CacheStatus<Option<W>> {
        CacheStatus::NotComputed
    }

    fn insert_final_weight(&self, _id: StateId, _weight: Option<W>) {}

    fn num_known_states(&self) -> usize {
        0
    }

    fn compute_num_known_trs(&self) -> usize {
        0
    }

    fn num_trs(&self, _id: StateId) -> Option<usize> {
        None
    }

    fn num_input_epsilons(&self, _id: StateId) -> Option<usize> {
        None
    }

    fn num_output_epsilons(&self, _id: StateId) -> Option<usize> {
        None
    }

    fn len_trs(&self) -> usize {
        0
    }

    fn len_final_weights(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::Tr;
    use crate::semirings::TropicalWeight;
    use anyhow::Result;

    // Every read is a miss and every write is dropped: the cache never surfaces
    // a value, which is exactly what the single-touch materialization relies on.
    #[test]
    fn null_cache_never_stores() -> Result<()> {
        let cache = NullCache::<TropicalWeight>::default();
        assert!(cache.get_start().is_not_computed());
        cache.insert_start(Some(1));
        assert!(cache.get_start().is_not_computed());

        let mut trs = TrsVec::<TropicalWeight>::default();
        trs.push(Tr::new(0, 0, TropicalWeight::one(), 3));
        cache.insert_trs(0, trs);
        assert!(cache.get_trs(0).is_not_computed());
        assert_eq!(cache.num_trs(0), None);
        assert_eq!(cache.num_input_epsilons(0), None);
        assert_eq!(cache.num_output_epsilons(0), None);

        cache.insert_final_weight(0, Some(TropicalWeight::one()));
        assert!(cache.get_final_weight(0).is_not_computed());

        assert_eq!(cache.num_known_states(), 0);
        assert_eq!(cache.compute_num_known_trs(), 0);
        assert_eq!(cache.len_trs(), 0);
        assert_eq!(cache.len_final_weights(), 0);
        Ok(())
    }
}

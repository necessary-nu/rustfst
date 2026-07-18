use std::cell::RefCell;

use crate::algorithms::lazy::cache::cache_internal_types::{
    CacheTrs, CachedData, FinalWeight, StartState,
};
use crate::algorithms::lazy::{CacheStatus, FstCache};
use crate::semirings::Semiring;
use crate::{StateId, Trs, TrsVec, EPS_LABEL};

/// Single-threaded twin of [`SimpleVecCache`](super::SimpleVecCache).
///
/// The static `compose` entry point builds a `ComposeFst`, materializes it once
/// on the calling thread via `compute`, and drops it. It never shares the FST
/// across threads, so the per-access `Mutex` that `SimpleVecCache` pays on every
/// `get_trs`/`insert_trs`/`final_weight` buys nothing. This variant swaps the
/// three `Mutex` for `RefCell`. Per-state lookups are `Vec`-indexed (no hashing),
/// so unlike the hash-map cache there is no SipHash cost to remove here — the win
/// is purely the elided lock.
///
/// This type is deliberately **not** `Sync` (a `RefCell` is `!Sync`). It is not
/// public API; only the static one-shot compose entry point instantiates it. The
/// thread-safe `SimpleVecCache` remains the default for `ComposeFst` and every
/// lazy FST a caller may legitimately share.
#[derive(Debug)]
pub struct UnsyncVecCache<W: Semiring> {
    start: RefCell<CachedData<CacheStatus<StartState>>>,
    trs: RefCell<CachedData<Vec<CacheStatus<CacheTrs<W>>>>>,
    final_weights: RefCell<CachedData<Vec<CacheStatus<FinalWeight<W>>>>>,
}

impl<W: Semiring> Default for UnsyncVecCache<W> {
    fn default() -> Self {
        Self {
            start: RefCell::new(CachedData::default()),
            trs: RefCell::new(CachedData::default()),
            final_weights: RefCell::new(CachedData::default()),
        }
    }
}

impl<W: Semiring> Clone for UnsyncVecCache<W> {
    fn clone(&self) -> Self {
        Self {
            start: RefCell::new(self.start.borrow().clone()),
            trs: RefCell::new(self.trs.borrow().clone()),
            final_weights: RefCell::new(self.final_weights.borrow().clone()),
        }
    }
}

impl<W: Semiring> FstCache<W> for UnsyncVecCache<W> {
    fn get_start(&self) -> CacheStatus<StartState> {
        self.start.borrow().data
    }

    fn insert_start(&self, id: StartState) {
        let mut cached_data = self.start.borrow_mut();
        if let Some(s) = id {
            cached_data.num_known_states =
                std::cmp::max(cached_data.num_known_states, s as usize + 1);
        }
        cached_data.data = CacheStatus::Computed(id);
    }

    fn get_trs(&self, id: StateId) -> CacheStatus<TrsVec<W>> {
        self.trs
            .borrow()
            .get(id)
            .map(|e| e.trs.shallow_clone())
    }

    fn insert_trs(&self, id: StateId, trs: TrsVec<W>) {
        let id = id as usize;
        let mut cached_data = self.trs.borrow_mut();
        let mut niepsilons = 0;
        let mut noepsilons = 0;
        for tr in trs.trs() {
            cached_data.num_known_states =
                std::cmp::max(cached_data.num_known_states, tr.nextstate as usize + 1);
            if tr.ilabel == EPS_LABEL {
                niepsilons += 1;
            }
            if tr.olabel == EPS_LABEL {
                noepsilons += 1;
            }
        }
        if id >= cached_data.data.len() {
            cached_data.data.resize(id + 1, CacheStatus::NotComputed);
        }
        cached_data.data[id] = CacheStatus::Computed(CacheTrs {
            trs,
            niepsilons,
            noepsilons,
        });
    }

    fn get_final_weight(&self, id: StateId) -> CacheStatus<FinalWeight<W>> {
        let id = id as usize;
        let cached_data = self.final_weights.borrow();
        match cached_data.data.get(id) {
            Some(e) => e.clone(),
            None => CacheStatus::NotComputed,
        }
    }

    fn insert_final_weight(&self, id: StateId, weight: FinalWeight<W>) {
        let id = id as usize;
        let mut cached_data = self.final_weights.borrow_mut();
        cached_data.num_known_states = std::cmp::max(cached_data.num_known_states, id + 1);
        if id >= cached_data.data.len() {
            cached_data.data.resize(id + 1, CacheStatus::NotComputed);
        }
        cached_data.data[id] = CacheStatus::Computed(weight);
    }

    fn num_known_states(&self) -> usize {
        let mut n = 0;
        n = std::cmp::max(n, self.start.borrow().num_known_states);
        n = std::cmp::max(n, self.trs.borrow().num_known_states);
        n = std::cmp::max(n, self.final_weights.borrow().num_known_states);
        n
    }

    fn compute_num_known_trs(&self) -> usize {
        self.trs
            .borrow()
            .data
            .iter()
            .flat_map(|it| it.to_option())
            .map(|it| it.trs.trs().len())
            .sum()
    }

    fn num_trs(&self, id: StateId) -> Option<usize> {
        self.trs.borrow().get(id).map(|e| e.trs.len()).into_option()
    }

    fn num_input_epsilons(&self, id: StateId) -> Option<usize> {
        self.trs.borrow().get(id).map(|e| e.niepsilons).into_option()
    }

    fn num_output_epsilons(&self, id: StateId) -> Option<usize> {
        self.trs.borrow().get(id).map(|e| e.noepsilons).into_option()
    }

    fn len_trs(&self) -> usize {
        self.trs.borrow().data.len()
    }

    fn len_final_weights(&self) -> usize {
        self.final_weights.borrow().data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::Tr;
    use crate::semirings::TropicalWeight;
    use anyhow::Result;

    #[test]
    fn unsync_vec_cache_roundtrips() -> Result<()> {
        let mut trs = TrsVec::<TropicalWeight>::default();
        trs.push(Tr::new(0, 1, TropicalWeight::one(), 2));
        trs.push(Tr::new(1, 0, TropicalWeight::one(), 0));
        trs.push(Tr::new(0, 0, TropicalWeight::one(), 10));

        let cache = UnsyncVecCache::default();
        assert!(cache.get_start().is_not_computed());
        cache.insert_start(Some(1));
        assert_eq!(cache.get_start(), CacheStatus::Computed(Some(1)));

        assert!(cache.get_trs(2).is_not_computed());
        cache.insert_trs(2, trs.clone());
        assert_eq!(cache.num_trs(2), Some(3));
        // input epsilons = trs with ilabel 0: tr0 and tr2 -> 2.
        // output epsilons = trs with olabel 0: tr1 and tr2 -> 2.
        assert_eq!(cache.num_input_epsilons(2), Some(2));
        assert_eq!(cache.num_output_epsilons(2), Some(2));

        cache.insert_final_weight(0, Some(TropicalWeight::one()));
        assert_eq!(
            cache.get_final_weight(0),
            CacheStatus::Computed(Some(TropicalWeight::one()))
        );
        assert!(cache.get_final_weight(5).is_not_computed());

        // sparse insert: state 2's trs left index 0/1 NotComputed
        assert!(cache.get_trs(0).is_not_computed());
        assert_eq!(cache.compute_num_known_trs(), 3);
        assert_eq!(cache.num_known_states(), 11);
        Ok(())
    }

    #[test]
    fn unsync_vec_cache_no_borrow_conflict() -> Result<()> {
        let cache = UnsyncVecCache::<TropicalWeight>::default();
        let mut trs = TrsVec::<TropicalWeight>::default();
        trs.push(Tr::new(1, 1, TropicalWeight::one(), 1));
        cache.insert_trs(0, trs);
        let _read = cache.get_trs(0);
        let _fw = cache.get_final_weight(0);
        cache.insert_final_weight(0, Some(TropicalWeight::one()));
        Ok(())
    }
}

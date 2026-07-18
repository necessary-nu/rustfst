use std::cell::RefCell;
use std::collections::HashMap;

use crate::algorithms::lazy::cache::cache_internal_types::{
    CacheTrs, CachedData, FinalWeight, StartState,
};
use crate::algorithms::lazy::{CacheStatus, FstCache};
use crate::fx_hasher::FxBuildHasher;
use crate::semirings::Semiring;
use crate::{StateId, Trs, TrsVec, EPS_LABEL};

/// Single-threaded twin of [`SimpleHashMapCache`](super::SimpleHashMapCache).
///
/// The one-shot static algorithms (e.g. `determinize`) build a `LazyFst`,
/// materialize it once on the calling thread via `compute_bounded`, and drop
/// it. They never share the FST across threads, so the per-access `Mutex` that
/// `SimpleHashMapCache` pays on every `get_trs`/`insert_trs`/`final_weight`
/// buys nothing. This variant swaps the three `Mutex` for `RefCell` and swaps
/// the maps' default `RandomState` (SipHash) for `FxBuildHasher`: the map keys
/// are `StateId`s never exposed through any output-affecting iteration, so
/// SipHash's DoS resistance is dead weight and FxHash is a strict win.
///
/// This type is deliberately **not** `Sync` (a `RefCell` is `!Sync`). It is not
/// public API; only the static one-shot entry points instantiate it. The
/// thread-safe `SimpleHashMapCache` remains the default for every lazy FST that
/// a caller may legitimately share.
#[derive(Debug)]
pub struct UnsyncHashMapCache<W: Semiring> {
    start: RefCell<CachedData<CacheStatus<StartState>>>,
    trs: RefCell<CachedData<HashMap<StateId, CacheTrs<W>, FxBuildHasher>>>,
    final_weights: RefCell<CachedData<HashMap<StateId, FinalWeight<W>, FxBuildHasher>>>,
}

impl<W: Semiring> Default for UnsyncHashMapCache<W> {
    fn default() -> Self {
        Self {
            start: RefCell::new(CachedData::default()),
            trs: RefCell::new(CachedData {
                data: HashMap::with_hasher(FxBuildHasher::default()),
                num_known_states: 0,
            }),
            final_weights: RefCell::new(CachedData {
                data: HashMap::with_hasher(FxBuildHasher::default()),
                num_known_states: 0,
            }),
        }
    }
}

impl<W: Semiring> Clone for UnsyncHashMapCache<W> {
    fn clone(&self) -> Self {
        Self {
            start: RefCell::new(self.start.borrow().clone()),
            trs: RefCell::new(self.trs.borrow().clone()),
            final_weights: RefCell::new(self.final_weights.borrow().clone()),
        }
    }
}

impl<W: Semiring> FstCache<W> for UnsyncHashMapCache<W> {
    fn get_start(&self) -> CacheStatus<StartState> {
        self.start.borrow().data
    }

    fn insert_start(&self, id: StartState) {
        let mut data = self.start.borrow_mut();
        if let Some(s) = id {
            data.num_known_states = std::cmp::max(data.num_known_states, s as usize + 1);
        }
        data.data = CacheStatus::Computed(id);
    }

    fn get_trs(&self, id: StateId) -> CacheStatus<TrsVec<W>> {
        match self.trs.borrow().data.get(&id) {
            Some(e) => CacheStatus::Computed(e.trs.shallow_clone()),
            None => CacheStatus::NotComputed,
        }
    }

    fn insert_trs(&self, id: StateId, trs: TrsVec<W>) {
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
        cached_data.data.insert(
            id,
            CacheTrs {
                trs,
                niepsilons,
                noepsilons,
            },
        );
    }

    fn compute_num_known_trs(&self) -> usize {
        self.trs
            .borrow()
            .data
            .values()
            .map(|it| it.trs.trs().len())
            .sum()
    }

    fn get_final_weight(&self, id: StateId) -> CacheStatus<FinalWeight<W>> {
        match self.final_weights.borrow().data.get(&id) {
            Some(e) => CacheStatus::Computed(e.clone()),
            None => CacheStatus::NotComputed,
        }
    }

    fn insert_final_weight(&self, id: StateId, weight: FinalWeight<W>) {
        let mut cached_data = self.final_weights.borrow_mut();
        cached_data.num_known_states = std::cmp::max(cached_data.num_known_states, id as usize + 1);
        cached_data.data.insert(id, weight);
    }

    fn num_known_states(&self) -> usize {
        let mut n = 0;
        n = std::cmp::max(n, self.start.borrow().num_known_states);
        n = std::cmp::max(n, self.trs.borrow().num_known_states);
        n = std::cmp::max(n, self.final_weights.borrow().num_known_states);
        n
    }

    fn num_trs(&self, id: StateId) -> Option<usize> {
        self.trs.borrow().data.get(&id).map(|v| v.trs.len())
    }

    fn num_input_epsilons(&self, id: StateId) -> Option<usize> {
        self.trs.borrow().data.get(&id).map(|v| v.niepsilons)
    }

    fn num_output_epsilons(&self, id: StateId) -> Option<usize> {
        self.trs.borrow().data.get(&id).map(|v| v.noepsilons)
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
    fn unsync_hash_map_cache_roundtrips() -> Result<()> {
        let mut trs = TrsVec::<TropicalWeight>::default();
        trs.push(Tr::new(0, 1, TropicalWeight::one(), 2));
        trs.push(Tr::new(1, 0, TropicalWeight::one(), 0));
        trs.push(Tr::new(0, 0, TropicalWeight::one(), 10));

        let cache = UnsyncHashMapCache::default();
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

        assert_eq!(cache.len_trs(), 1);
        assert_eq!(cache.len_final_weights(), 1);
        assert_eq!(cache.compute_num_known_trs(), 3);
        // num_known_states: max nextstate+1 seen in trs (10 -> 11) and start (1 -> 2).
        assert_eq!(cache.num_known_states(), 11);
        Ok(())
    }

    // The materialization loop reads a state's trs, then (separately) its final
    // weight; a RefCell-backed cache must not panic on that interleaving.
    #[test]
    fn unsync_hash_map_cache_no_borrow_conflict() -> Result<()> {
        let cache = UnsyncHashMapCache::<TropicalWeight>::default();
        let mut trs = TrsVec::<TropicalWeight>::default();
        trs.push(Tr::new(1, 1, TropicalWeight::one(), 1));
        cache.insert_trs(0, trs);
        let _read = cache.get_trs(0);
        // read fully returned (owned clone) before we touch the cache again
        let _fw = cache.get_final_weight(0);
        cache.insert_final_weight(0, Some(TropicalWeight::one()));
        Ok(())
    }
}

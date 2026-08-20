use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::algorithms::determinize::{
    DeterminizeStateTuple, DeterminizeSubsetLimitExceeded, WeightedSubset,
};
use crate::fx_hasher::FxBuildHasher;
use crate::{Semiring, StateId};
use anyhow::Result;

#[derive(Debug, PartialEq)]
struct InnerDeterminizeStateTable<W: Semiring, B: Borrow<[W]>> {
    // id -> tuple is a plain vector; tuple -> id shares the same allocation
    // through the Arc, so a tuple is stored (and cloned) exactly once. The
    // former bimap kept two full directional maps and its find_tuple cloned
    // the whole weighted subset per state expansion.
    id_to_tuple: Vec<Arc<DeterminizeStateTuple<W>>>,
    // State ids come from insertion order (the Vec), never from map iteration,
    // so the hasher cannot influence the output; SipHash was ~5% of the whole
    // determinize profile hashing large weighted subsets.
    tuple_to_id: HashMap<Arc<DeterminizeStateTuple<W>>, StateId, FxBuildHasher>,
    // Distance to final NFA states.
    in_dist: Option<B>,
    // Distance to final DFA states.
    out_dist: Vec<Option<W>>,
    // Logical elements retained by the unique weighted subsets above. This is
    // deliberately separate from the DFA-state count: a single state may own
    // millions of elements.
    stored_subset_elements: usize,
    max_subset_elements: Option<usize>,
}

impl<W: Semiring, B: Borrow<[W]> + PartialEq> InnerDeterminizeStateTable<W, B> {
    fn compute_distance(&self, subset: &WeightedSubset<W>) -> Result<W> {
        let mut outd = W::zero();
        let weight_zero = W::zero();
        for element in subset.iter() {
            let ind = self
                .in_dist
                .as_ref()
                .unwrap()
                .borrow()
                .get(element.state as usize)
                .unwrap_or(&weight_zero);
            outd.plus_assign(element.weight.times(ind)?)?;
        }
        Ok(outd)
    }
}

// The table lives inside a `DeterminizeFsaOp`, inside a `DeterminizeFsa`, which
// is a one-shot type materialized and dropped on a single thread — it is never
// shared across threads (its `LazyFst` cache is a `NullCache` and this table is
// its only other interior-mutable state). So the `Mutex` that used to guard the
// inner table bought nothing but a per-`find_id`/`find_tuple` lock; a `RefCell`
// gives the same interior mutability with no synchronization. This makes the
// table (and hence `DeterminizeFsaOp`/`DeterminizeFsa`) `!Sync`, which is the
// intended shape for a single-threaded one-shot value.
pub struct DeterminizeStateTable<W: Semiring, B: Borrow<[W]>>(
    RefCell<InnerDeterminizeStateTable<W, B>>,
);

impl<W: Semiring, B: Borrow<[W]> + fmt::Debug> fmt::Debug for DeterminizeStateTable<W, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0.borrow())
    }
}

impl<W: Semiring, B: Borrow<[W]> + PartialEq> PartialEq for DeterminizeStateTable<W, B> {
    fn eq(&self, other: &Self) -> bool {
        self.0.borrow().eq(&*other.0.borrow())
    }
}

impl<W: Semiring, B: Borrow<[W]>> DeterminizeStateTable<W, B> {
    pub fn new(in_dist: Option<B>, max_subset_elements: Option<usize>) -> Self {
        Self(RefCell::new(InnerDeterminizeStateTable {
            in_dist,
            out_dist: vec![],
            id_to_tuple: Vec::new(),
            tuple_to_id: HashMap::default(),
            stored_subset_elements: 0,
            max_subset_elements,
        }))
    }

    /// Checks the transient elements accumulated while expanding one existing
    /// subset. The persistent table and in-progress expansion coexist, so both
    /// count against one logical element budget.
    pub fn check_expansion_elements(&self, expansion_elements: usize) -> Result<()> {
        let inner = self.0.borrow();
        let attempted = inner
            .stored_subset_elements
            .saturating_add(expansion_elements);
        if let Some(limit) = inner.max_subset_elements {
            if attempted > limit {
                return Err(DeterminizeSubsetLimitExceeded { limit, attempted }.into());
            }
        }
        Ok(())
    }

    /// Looks up tuple from integer ID. O(1); shares the stored allocation.
    pub fn find_tuple(&self, tuple_id: StateId) -> Arc<DeterminizeStateTuple<W>> {
        let inner = self.0.borrow();
        Arc::clone(&inner.id_to_tuple[tuple_id as usize])
    }

    pub fn out_dist(self) -> Vec<Option<W>> {
        let inner = self.0.into_inner();
        inner.out_dist
    }
}

impl<W: Semiring, B: Borrow<[W]> + PartialEq> DeterminizeStateTable<W, B> {
    /// Looks up integer ID from entry. Inserts if absent: one hash on the hit
    /// path, one hash plus one tuple clone on the miss path.
    pub fn find_id_from_ref(&self, tuple: &DeterminizeStateTuple<W>) -> Result<StateId> {
        let mut inner = self.0.borrow_mut();
        if let Some(id) = inner.tuple_to_id.get(tuple) {
            return Ok(*id);
        }
        let attempted = inner
            .stored_subset_elements
            .saturating_add(tuple.subset.pairs.len());
        if let Some(limit) = inner.max_subset_elements {
            if attempted > limit {
                return Err(DeterminizeSubsetLimitExceeded { limit, attempted }.into());
            }
        }
        let n = inner.id_to_tuple.len();
        if inner.in_dist.is_some() {
            if n >= inner.out_dist.len() {
                inner.out_dist.resize(n + 1, None);
            }
            if inner.out_dist[n].is_none() {
                let d = inner.compute_distance(&tuple.subset)?;
                inner.out_dist[n] = Some(d);
            }
        }
        let tuple = Arc::new(tuple.clone());
        inner.id_to_tuple.push(Arc::clone(&tuple));
        inner.tuple_to_id.insert(tuple, n as StateId);
        inner.stored_subset_elements = attempted;
        Ok(n as StateId)
    }
}

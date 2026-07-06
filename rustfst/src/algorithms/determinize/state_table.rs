use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::algorithms::determinize::{DeterminizeStateTuple, WeightedSubset};
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

pub struct DeterminizeStateTable<W: Semiring, B: Borrow<[W]>>(
    Mutex<InnerDeterminizeStateTable<W, B>>,
);

impl<W: Semiring, B: Borrow<[W]> + fmt::Debug> fmt::Debug for DeterminizeStateTable<W, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0.lock().unwrap())
    }
}

impl<W: Semiring, B: Borrow<[W]> + PartialEq> PartialEq for DeterminizeStateTable<W, B> {
    fn eq(&self, other: &Self) -> bool {
        self.0.lock().unwrap().eq(&*other.0.lock().unwrap())
    }
}

impl<W: Semiring, B: Borrow<[W]>> DeterminizeStateTable<W, B> {
    pub fn new(in_dist: Option<B>) -> Self {
        Self(Mutex::new(InnerDeterminizeStateTable {
            in_dist,
            out_dist: vec![],
            id_to_tuple: Vec::new(),
            tuple_to_id: HashMap::default(),
        }))
    }

    /// Looks up tuple from integer ID. O(1); shares the stored allocation.
    pub fn find_tuple(&self, tuple_id: StateId) -> Arc<DeterminizeStateTuple<W>> {
        let inner = self.0.lock().unwrap();
        Arc::clone(&inner.id_to_tuple[tuple_id as usize])
    }

    pub fn out_dist(self) -> Vec<Option<W>> {
        let inner = self.0.into_inner().unwrap();
        inner.out_dist
    }
}

impl<W: Semiring, B: Borrow<[W]> + PartialEq> DeterminizeStateTable<W, B> {
    /// Looks up integer ID from entry. Inserts if absent: one hash on the hit
    /// path, one hash plus one tuple clone on the miss path.
    pub fn find_id_from_ref(&self, tuple: &DeterminizeStateTuple<W>) -> Result<StateId> {
        let mut inner = self.0.lock().unwrap();
        if let Some(id) = inner.tuple_to_id.get(tuple) {
            return Ok(*id);
        }
        let n = inner.id_to_tuple.len();
        if inner.in_dist.is_some() {
            if n >= inner.out_dist.len() {
                inner.out_dist.resize(n + 1, None);
            }
            if inner.out_dist[n].is_none() {
                inner.out_dist[n] = Some(inner.compute_distance(&tuple.subset)?);
            }
        }
        let tuple = Arc::new(tuple.clone());
        inner.id_to_tuple.push(Arc::clone(&tuple));
        inner.tuple_to_id.insert(tuple, n as StateId);
        Ok(n as StateId)
    }
}

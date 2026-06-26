use anyhow::Result;

use crate::algorithms::{connect, shortest_distance};
use crate::fst_traits::{ExpandedFst, MutableFst};
use crate::semirings::Semiring;
use crate::StateId;

/// Natural-order strict-less: `w1` is strictly better than `w2` in the natural
/// order of the semiring, i.e. `Plus(w1, w2) == w1` and `w1 != w2`. This mirrors
/// OpenFST's `NaturalLess`. It is only meaningful for semirings with the path
/// property (e.g. the tropical semiring), which is exactly where `prune` applies.
fn natural_less<W: Semiring>(w1: &W, w2: &W) -> bool {
    match w1.plus(w2) {
        Ok(sum) => &sum == w1 && w1 != w2,
        // Plus is total for the semirings used here; treat an error as "not less".
        Err(_) => false,
    }
}

/// Removes states and transitions in `fst` that do not belong to a path whose
/// weight is within `weight_threshold` (times the shortest path weight) of the
/// shortest path of the whole FST. Port of OpenFST `fst::Prune` with the default
/// options (no state threshold, `threshold_initial = false`).
///
/// `W` must have the path property; HFST only ever prunes tropical transducers.
///
/// OpenFST computes the shortest distance from each state to a final state
/// (`fdistance`) and, lazily during a best-first walk, the shortest distance from
/// the start state to each state (`idistance`). A transition `s -> d` with weight
/// `w` is kept iff `idistance[s] (x) w (x) fdistance[d]` is not strictly worse
/// than `limit = fdistance[start] (x) weight_threshold`. Because the kept
/// transitions are exactly those on within-threshold paths, the true
/// start-to-state shortest distances (computed eagerly here via
/// `shortest_distance`) agree with OpenFST's lazy values on every surviving
/// state, so this eager formulation yields the identical surviving machine; the
/// final `connect` removes the states left unreachable.
pub fn prune<W, F>(fst: &mut F, weight_threshold: W) -> Result<()>
where
    W: Semiring,
    F: ExpandedFst<W> + MutableFst<W>,
{
    let num_states = fst.num_states();
    if num_states == 0 {
        return Ok(());
    }
    let start = match fst.start() {
        Some(s) => s,
        // No start state: the language is empty, nothing survives.
        None => {
            fst.del_all_states();
            return Ok(());
        }
    };

    // fdistance[s] = shortest distance from s to a final state (reverse = true).
    let fdistance = shortest_distance(fst, true)?;
    // idistance[s] = shortest distance from the start state to s.
    let idistance = shortest_distance(fst, false)?;

    let fdist = |s: StateId| -> W { fdistance.get(s as usize).cloned().unwrap_or_else(W::zero) };
    let idist = |s: StateId| -> W { idistance.get(s as usize).cloned().unwrap_or_else(W::zero) };

    // If the start state cannot reach any final state the result is empty.
    if fdist(start).is_zero() {
        fst.del_all_states();
        return Ok(());
    }

    let limit = fdist(start).times(&weight_threshold)?;

    for s in 0..num_states {
        let state = s as StateId;

        // Prune the final weight if the best path through it exceeds the limit.
        if let Some(final_weight) = fst.final_weight(state)? {
            if natural_less(&limit, &idist(state).times(&final_weight)?) {
                fst.delete_final_weight(state)?;
            }
        }

        // Keep only the transitions whose best path through is within the limit.
        let trs = fst.pop_trs(state)?;
        for tr in trs {
            let through = idist(state)
                .times(&tr.weight)?
                .times(&fdist(tr.nextstate))?;
            if !natural_less(&limit, &through) {
                fst.add_tr(state, tr)?;
            }
        }
    }

    // Drop the states left unreachable from the start or unable to reach a final.
    connect(fst)?;
    Ok(())
}

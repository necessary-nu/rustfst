use std::borrow::Borrow;

use anyhow::Result;

use crate::algorithms::determinize::divisors::CommonDivisor;
use crate::algorithms::determinize::DeterminizeFsa;
use crate::algorithms::determinize::{DefaultCommonDivisor, DeterminizeType, GallicCommonDivisor};
use crate::algorithms::factor_weight::factor_iterators::{
    GallicFactor, GallicFactorMin, GallicFactorRestrict,
};
use crate::algorithms::factor_weight::{factor_weight, FactorWeightOptions, FactorWeightType};
use crate::algorithms::weight_convert;
use crate::algorithms::weight_converters::{FromGallicConverter, ToGallicConverter};
use crate::fst_impls::VectorFst;
use crate::fst_properties::mutable_properties::determinize_properties;
use crate::fst_properties::FstProperties;
use crate::fst_traits::{AllocableFst, ExpandedFst, Fst, MutableFst};
use crate::semirings::SemiringProperties;
use crate::semirings::{
    GallicWeight, GallicWeightMin, GallicWeightRestrict, WeaklyDivisibleSemiring, WeightQuantize,
};
use crate::{EPS_LABEL, KDELTA};

pub fn determinize_with_distance<W, F1, F2>(
    ifst: &F1,
    in_dist: &[W],
    delta: f32,
) -> Result<(F2, Vec<W>)>
where
    W: WeaklyDivisibleSemiring + WeightQuantize,
    F1: ExpandedFst<W>,
    F2: MutableFst<W> + AllocableFst<W>,
{
    if !W::properties().contains(SemiringProperties::LEFT_SEMIRING) {
        bail!("determinize_fsa : weight must be left distributive")
    }
    let fst = DeterminizeFsa::<_, F1, DefaultCommonDivisor, _, _>::new(ifst, Some(in_dist), delta)?;
    fst.compute_with_distance()
}

pub fn determinize_fsa<W, F1, F2, CD>(
    fst_in: &F1,
    delta: f32,
    max_states: Option<usize>,
) -> Result<F2>
where
    W: WeaklyDivisibleSemiring + WeightQuantize,
    F1: Fst<W>,
    F2: MutableFst<W> + AllocableFst<W>,
    CD: CommonDivisor<W>,
{
    if !W::properties().contains(SemiringProperties::LEFT_SEMIRING) {
        bail!("determinize_fsa : weight must be left distributive")
    }
    let det_fsa: DeterminizeFsa<W, F1, CD, _, Vec<W>> = DeterminizeFsa::new(fst_in, None, delta)?;
    det_fsa.compute_bounded(max_states)
}

pub fn determinize_fst<W, F1, F2>(
    fst_in: &F1,
    det_type: DeterminizeType,
    delta: f32,
    max_states: Option<usize>,
) -> Result<F2>
where
    W: WeaklyDivisibleSemiring + WeightQuantize + 'static,
    F1: ExpandedFst<W>,
    F2: MutableFst<W> + AllocableFst<W>,
{
    let mut to_gallic = ToGallicConverter {};
    let mut from_gallic = FromGallicConverter {
        superfinal_label: EPS_LABEL,
    };

    let factor_opts = FactorWeightOptions {
        delta: KDELTA,
        mode: FactorWeightType::FACTOR_FINAL_WEIGHTS,
        final_ilabel: EPS_LABEL,
        final_olabel: EPS_LABEL,
        increment_final_ilabel: false,
        increment_final_olabel: false,
    };

    match det_type {
        DeterminizeType::DeterminizeDisambiguate => {
            if !W::properties().contains(SemiringProperties::PATH) {
                bail!("determinize : weight needs to have the path property to disambiguate output")
            }
            let fsa: VectorFst<GallicWeightMin<W>> =
                weight_convert(fst_in.borrow(), &mut to_gallic)?;
            let determinized_fsa: VectorFst<GallicWeightMin<W>> =
                determinize_fsa::<_, VectorFst<_>, _, GallicCommonDivisor>(
                    &fsa, delta, max_states,
                )?;
            let factored_determinized_fsa: VectorFst<GallicWeightMin<W>> =
                factor_weight::<_, VectorFst<GallicWeightMin<W>>, _, _, GallicFactorMin<W>>(
                    &determinized_fsa,
                    factor_opts,
                )?;
            weight_convert(&factored_determinized_fsa, &mut from_gallic)
        }
        DeterminizeType::DeterminizeFunctional => {
            let fsa: VectorFst<GallicWeightRestrict<W>> =
                weight_convert(fst_in.borrow(), &mut to_gallic)?;
            let determinized_fsa: VectorFst<GallicWeightRestrict<W>> =
                determinize_fsa::<_, VectorFst<_>, _, GallicCommonDivisor>(
                    &fsa, delta, max_states,
                )?;
            let factored_determinized_fsa: VectorFst<GallicWeightRestrict<W>> =
                factor_weight::<
                    _,
                    VectorFst<GallicWeightRestrict<W>>,
                    _,
                    _,
                    GallicFactorRestrict<W>,
                >(&determinized_fsa, factor_opts)?;
            weight_convert(&factored_determinized_fsa, &mut from_gallic)
        }
        DeterminizeType::DeterminizeNonFunctional => {
            let fsa: VectorFst<GallicWeight<W>> = weight_convert(fst_in.borrow(), &mut to_gallic)?;
            let determinized_fsa: VectorFst<GallicWeight<W>> =
                determinize_fsa::<_, VectorFst<_>, _, GallicCommonDivisor>(
                    &fsa, delta, max_states,
                )?;
            let factored_determinized_fsa: VectorFst<GallicWeight<W>> =
                factor_weight::<_, VectorFst<GallicWeight<W>>, _, _, GallicFactor<W>>(
                    &determinized_fsa,
                    factor_opts,
                )?;
            weight_convert(&factored_determinized_fsa, &mut from_gallic)
        }
    }
}

#[derive(Clone, Debug, Copy, PartialOrd, PartialEq)]
pub struct DeterminizeConfig {
    pub delta: f32,
    pub det_type: DeterminizeType,
    /// Optional upper bound on the number of states produced by the
    /// determinization. `None` (the default) is unbounded — today's behavior.
    /// When `Some(n)`, if the on-demand expansion produces more than `n` states,
    /// determinization returns an `Err` instead of running away. This gives
    /// callers an escape hatch for inputs on which weighted determinization does
    /// not terminate (e.g. non-twins cyclic FSTs). Any input that converges
    /// within the bound produces byte-identical output to the unbounded run.
    pub max_states: Option<usize>,
}

impl DeterminizeConfig {
    pub fn new(delta: f32, det_type: DeterminizeType) -> Self {
        Self {
            delta,
            det_type,
            max_states: None,
        }
    }

    pub fn with_delta(self, delta: f32) -> Self {
        Self { delta, ..self }
    }

    pub fn with_det_type(self, det_type: DeterminizeType) -> Self {
        Self { det_type, ..self }
    }

    pub fn with_max_states(self, max_states: Option<usize>) -> Self {
        Self { max_states, ..self }
    }
}

impl Default for DeterminizeConfig {
    fn default() -> Self {
        Self {
            delta: KDELTA,
            det_type: DeterminizeType::DeterminizeFunctional,
            max_states: None,
        }
    }
}

pub fn determinize<W, F1, F2>(fst_in: &F1) -> Result<F2>
where
    W: WeaklyDivisibleSemiring + WeightQuantize,
    F1: ExpandedFst<W>,
    F2: MutableFst<W> + AllocableFst<W>,
{
    determinize_with_config(fst_in, DeterminizeConfig::default())
}

/// This operations creates an equivalent FST that has the property that no
/// state has two transitions with the same input label. For this algorithm,
/// epsilon transitions are treated as regular symbols.
///
/// # Example
///
/// ## Input
///
/// ![determinize_in](https://raw.githubusercontent.com/Garvys/rustfst-images-doc/master/images/determinize_in.svg?sanitize=true)
///
/// ## Determinize
///
/// ![determinize_out](https://raw.githubusercontent.com/Garvys/rustfst-images-doc/master/images/determinize_out.svg?sanitize=true)
///
pub fn determinize_with_config<W, F1, F2>(fst_in: &F1, config: DeterminizeConfig) -> Result<F2>
where
    W: WeaklyDivisibleSemiring + WeightQuantize,
    F1: ExpandedFst<W>,
    F2: MutableFst<W> + AllocableFst<W>,
{
    let delta = config.delta;
    let det_type = config.det_type;
    let max_states = config.max_states;
    let iprops = fst_in.borrow().properties();
    let mut fst_res: F2 = if iprops.contains(FstProperties::ACCEPTOR) {
        determinize_fsa::<_, F1, _, DefaultCommonDivisor>(fst_in, delta, max_states)?
    } else {
        determinize_fst(fst_in, det_type, delta, max_states)?
    };

    let distinct_psubsequential_labels = !(det_type == DeterminizeType::DeterminizeNonFunctional);
    let mut props = determinize_properties(iprops, false, distinct_psubsequential_labels);
    if iprops.contains(FstProperties::ACCEPTOR) {
        // The FSA determinization path emits each state's transitions in
        // input-label order; for an acceptor that is also output-label-sorted.
        props |= FstProperties::I_LABEL_SORTED | FstProperties::O_LABEL_SORTED;
    }
    fst_res.set_properties(props);
    fst_res.set_symts_from_fst(fst_in.borrow());
    Ok(fst_res)
}

#[cfg(test)]
mod tests {
    use crate::fst_impls::VectorFst;
    use crate::fst_traits::CoreFst;
    use crate::semirings::TropicalWeight;
    use crate::tr::Tr;
    use crate::Semiring;
    use crate::StateId;
    use crate::SymbolTable;
    use proptest::prelude::any;
    use proptest::proptest;
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_determinize() -> Result<()> {
        let mut input_fst = VectorFst::<TropicalWeight>::new();
        let s0 = input_fst.add_state();
        let s1 = input_fst.add_state();

        input_fst.set_start(s0)?;
        input_fst.set_final(s1, TropicalWeight::one())?;

        input_fst.add_tr(s0, Tr::new(1, 1, 2.0, s1))?;
        input_fst.add_tr(s0, Tr::new(1, 1, 2.0, s1))?;
        input_fst.add_tr(s0, Tr::new(1, 1, 2.0, s1))?;

        let mut ref_fst = VectorFst::new();
        let s0 = ref_fst.add_state();
        let s1 = ref_fst.add_state();

        ref_fst.set_start(s0)?;
        ref_fst.set_final(s1, TropicalWeight::one())?;

        ref_fst.add_tr(s0, Tr::new(1, 1, TropicalWeight::new(2.0), s1))?;

        let determinized_fst: VectorFst<TropicalWeight> = determinize(&input_fst)?;

        assert_eq!(determinized_fst, ref_fst);
        Ok(())
    }

    #[test]
    fn test_determinize_2() -> Result<()> {
        let mut input_fst = VectorFst::<TropicalWeight>::new();
        let s0 = input_fst.add_state();
        let s1 = input_fst.add_state();
        let s2 = input_fst.add_state();
        let s3 = input_fst.add_state();

        input_fst.set_start(s0)?;
        input_fst.set_final(s3, TropicalWeight::one())?;

        input_fst.add_tr(s0, Tr::new(1, 1, 2.0, s1))?;
        input_fst.add_tr(s0, Tr::new(1, 1, 3.0, s2))?;

        input_fst.add_tr(s1, Tr::new(2, 2, 4.0, s3))?;
        input_fst.add_tr(s2, Tr::new(2, 2, 3.0, s3))?;

        let mut ref_fst = VectorFst::new();
        let s0 = ref_fst.add_state();
        let s1 = ref_fst.add_state();
        let s2 = ref_fst.add_state();

        ref_fst.set_start(s0)?;
        ref_fst.set_final(s2, TropicalWeight::one())?;

        ref_fst.add_tr(s0, Tr::new(1, 1, TropicalWeight::new(2.0), s1))?;
        ref_fst.add_tr(s1, Tr::new(2, 2, TropicalWeight::new(4.0), s2))?;

        let determinized_fst: VectorFst<TropicalWeight> = determinize(&input_fst)?;

        assert_eq!(determinized_fst, ref_fst);
        Ok(())
    }

    // Regression test for upstream issue #288: `determinize` must not blow up the
    // state count relative to OpenFST. This is the reporter's exact input (a 4-state
    // nondeterministic acceptor; states 2 and 3 are final). OpenFST's `Determinize`
    // yields 6 states / 36 transitions. Before the subset-canonicalization fix the
    // unmerged subsets exploded to 14 states / 84 transitions.
    #[test]
    fn test_determinize_issue_288_no_blowup() -> Result<()> {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s: Vec<_> = (0..4).map(|_| fst.add_state()).collect();
        fst.set_start(s[0])?;
        fst.set_final(s[2], TropicalWeight::one())?;
        fst.set_final(s[3], TropicalWeight::one())?;

        // state 0: a label-5 transition to state 1 *and* a label-5 self-loop make
        // the acceptor nondeterministic; labels 1..6 self-loop on state 0.
        fst.add_tr(s[0], Tr::new(5, 5, 0.0, s[1]))?;
        for l in [4, 3, 2, 1, 6, 5] {
            fst.add_tr(s[0], Tr::new(l, l, 0.0, s[0]))?;
        }
        fst.add_tr(s[1], Tr::new(6, 6, 0.0, s[2]))?;
        for l in [4, 3, 2, 1, 6, 5] {
            fst.add_tr(s[2], Tr::new(l, l, 0.0, s[3]))?;
            fst.add_tr(s[3], Tr::new(l, l, 0.0, s[3]))?;
        }

        let det: VectorFst<TropicalWeight> = determinize(&fst)?;

        let num_trs: usize = (0..det.num_states())
            .map(|st| det.num_trs(st as StateId).unwrap())
            .sum();
        assert_eq!(
            det.num_states(),
            6,
            "issue #288: determinize blew up the state count (got {}, OpenFST gives 6)",
            det.num_states()
        );
        assert_eq!(num_trs, 36);
        Ok(())
    }

    // Byte-identity invariant for the `max_states` bound: on an input that
    // converges, a generous bound must produce EXACTLY the same FST as the
    // unbounded run (the bound only guards runaway expansions).
    #[test]
    fn test_determinize_max_states_byte_identical_when_within_budget() -> Result<()> {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s: Vec<_> = (0..4).map(|_| fst.add_state()).collect();
        fst.set_start(s[0])?;
        fst.set_final(s[3], TropicalWeight::one())?;
        fst.add_tr(s[0], Tr::new(1, 1, 2.0, s[1]))?;
        fst.add_tr(s[0], Tr::new(1, 1, 3.0, s[2]))?;
        fst.add_tr(s[1], Tr::new(2, 2, 4.0, s[3]))?;
        fst.add_tr(s[2], Tr::new(2, 2, 3.0, s[3]))?;

        let unbounded: VectorFst<TropicalWeight> = determinize(&fst)?;
        let bounded: VectorFst<TropicalWeight> = determinize_with_config(
            &fst,
            DeterminizeConfig::default().with_max_states(Some(1_000_000)),
        )?;
        assert_eq!(
            unbounded, bounded,
            "a generous state budget must not perturb a converging determinization"
        );
        Ok(())
    }

    // The bound actually trips: a tiny budget on a determinization that produces
    // more states than the budget returns an Err instead of the FST.
    #[test]
    fn test_determinize_max_states_trips() -> Result<()> {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s: Vec<_> = (0..4).map(|_| fst.add_state()).collect();
        fst.set_start(s[0])?;
        fst.set_final(s[3], TropicalWeight::one())?;
        fst.add_tr(s[0], Tr::new(1, 1, 2.0, s[1]))?;
        fst.add_tr(s[0], Tr::new(1, 1, 3.0, s[2]))?;
        fst.add_tr(s[1], Tr::new(2, 2, 4.0, s[3]))?;
        fst.add_tr(s[2], Tr::new(2, 2, 3.0, s[3]))?;

        let res: Result<VectorFst<TropicalWeight>> =
            determinize_with_config(&fst, DeterminizeConfig::default().with_max_states(Some(1)));
        assert!(
            res.is_err(),
            "a budget of 1 must abort this determinization"
        );
        Ok(())
    }

    proptest! {
        #[test]
        fn test_proptest_determinize_keeps_symts(mut fst in any::<VectorFst::<TropicalWeight>>()) {
            let symt = Arc::new(SymbolTable::new());
            fst.set_input_symbols(Arc::clone(&symt));
            fst.set_output_symbols(Arc::clone(&symt));

            let fst : VectorFst<_> = determinize_with_config(&fst, DeterminizeConfig::default().with_det_type(DeterminizeType::DeterminizeNonFunctional)).unwrap();

            assert!(fst.input_symbols().is_some());
            assert!(fst.output_symbols().is_some());
        }
    }
}

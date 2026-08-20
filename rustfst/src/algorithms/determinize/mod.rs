use determinize_fsa::DeterminizeFsa;
use determinize_fsa_op::DeterminizeFsaOp;
pub use determinize_static::{
    determinize, determinize_with_config, determinize_with_distance, DeterminizeConfig,
};
use divisors::{DefaultCommonDivisor, GallicCommonDivisor};
use element::{DeterminizeElement, DeterminizeStateTuple, DeterminizeTr, WeightedSubset};
use state_table::DeterminizeStateTable;

use std::error::Error;
use std::fmt;

mod determinize_fsa;
mod determinize_fsa_op;
mod determinize_static;
mod divisors;
mod element;
mod state_table;

/// Determinization stopped before retaining or constructing more weighted
/// subset elements than the caller allowed.
///
/// The state-count limit controls the number of DFA states. This separate
/// limit controls the data *inside* those states: one DFA state can contain a
/// very large weighted NFA subset, so a state count alone is not a memory
/// bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeterminizeSubsetLimitExceeded {
    /// Maximum number of logical subset elements allowed.
    pub limit: usize,
    /// Number the operation would have retained or constructed.
    pub attempted: usize,
}

impl fmt::Display for DeterminizeSubsetLimitExceeded {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "determinize: weighted-subset element budget of {} exceeded (attempted {})",
            self.limit, self.attempted
        )
    }
}

impl Error for DeterminizeSubsetLimitExceeded {}

/// Determinization type.
#[derive(Debug, Clone, PartialEq, PartialOrd, Copy)]
pub enum DeterminizeType {
    /// Input transducer is known to be functional (or error).
    DeterminizeFunctional,
    /// Input transducer is NOT known to be functional.
    DeterminizeNonFunctional,
    /// Input transducer is not known to be functional but only keep the min of
    /// of ambiguous outputs.
    DeterminizeDisambiguate,
}

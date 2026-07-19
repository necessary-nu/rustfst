use anyhow::{anyhow, Result};

use super::EnumConversionError;
use crate::fst::CFst;
use crate::{get, wrap, RUSTFST_FFI_RESULT};

use ffi_convert::*;
use rustfst::algorithms::determinize::{
    determinize, determinize_with_config, DeterminizeConfig, DeterminizeType,
};
use rustfst::fst_impls::VectorFst;
use rustfst::semirings::TropicalWeight;

#[derive(RawPointerConverter)]
pub struct CDeterminizeType(usize);

impl AsRust<DeterminizeType> for CDeterminizeType {
    fn as_rust(&self) -> Result<DeterminizeType, AsRustError> {
        match self.0 {
            0 => Ok(DeterminizeType::DeterminizeFunctional),
            1 => Ok(DeterminizeType::DeterminizeNonFunctional),
            2 => Ok(DeterminizeType::DeterminizeDisambiguate),
            _ => Err(AsRustError::Other(Box::new(EnumConversionError {}))),
        }
    }
}

impl CDrop for CDeterminizeType {
    fn do_drop(&mut self) -> Result<(), CDropError> {
        Ok(())
    }
}

impl CReprOf<DeterminizeType> for CDeterminizeType {
    fn c_repr_of(value: DeterminizeType) -> Result<CDeterminizeType, CReprOfError> {
        let variant = match value {
            DeterminizeType::DeterminizeFunctional => 0,
            DeterminizeType::DeterminizeNonFunctional => 1,
            DeterminizeType::DeterminizeDisambiguate => 2,
        };
        Ok(CDeterminizeType(variant))
    }
}

#[derive(RawPointerConverter)]
pub struct CDeterminizeConfig {
    delta: f32,
    det_type: CDeterminizeType,
    // `Option` has no direct C representation (see `CSigmaMatcherConfig` for
    // the same pattern), so the conversions are hand-written; the C
    // constructor encodes `None` as 0 since a zero-state bound is
    // meaningless.
    max_states: Option<usize>,
}

impl AsRust<DeterminizeConfig> for CDeterminizeConfig {
    fn as_rust(&self) -> Result<DeterminizeConfig, AsRustError> {
        Ok(DeterminizeConfig {
            delta: self.delta,
            det_type: self.det_type.as_rust()?,
            max_states: self.max_states,
        })
    }
}

impl CDrop for CDeterminizeConfig {
    fn do_drop(&mut self) -> Result<(), CDropError> {
        Ok(())
    }
}

impl CReprOf<DeterminizeConfig> for CDeterminizeConfig {
    fn c_repr_of(value: DeterminizeConfig) -> Result<CDeterminizeConfig, CReprOfError> {
        Ok(CDeterminizeConfig {
            delta: value.delta,
            det_type: CDeterminizeType::c_repr_of(value.det_type)?,
            max_states: value.max_states,
        })
    }
}

/// # Safety
///
/// The pointers should be valid.
///
/// `max_states` bounds the number of states the determinization may produce;
/// `0` means unbounded (the pre-budget behavior). A bounded run that exceeds
/// the limit makes `fst_determinize_with_config` return an error instead of
/// running away on inputs where weighted determinization does not terminate.
#[no_mangle]
pub unsafe extern "C" fn fst_determinize_config_new(
    delta: libc::c_float,
    det_type: libc::size_t,
    max_states: libc::size_t,
    config: *mut *const CDeterminizeConfig,
) -> RUSTFST_FFI_RESULT {
    wrap(|| {
        let determinize_config = CDeterminizeConfig {
            delta,
            det_type: CDeterminizeType(det_type),
            max_states: if max_states == 0 {
                None
            } else {
                Some(max_states)
            },
        };
        unsafe { *config = determinize_config.into_raw_pointer() };
        Ok(())
    })
}

/// # Safety
///
/// The pointers should be valid.
#[no_mangle]
pub unsafe extern "C" fn fst_determinize(
    ptr: *const CFst,
    det_fst: *mut *const CFst,
) -> RUSTFST_FFI_RESULT {
    wrap(|| {
        let fst = get!(CFst, ptr);
        let vec_fst: &VectorFst<TropicalWeight> = fst
            .downcast_ref()
            .ok_or_else(|| anyhow!("Could not downcast to vector FST"))?;
        let fst: VectorFst<TropicalWeight> = determinize(vec_fst)?;
        let fst_ptr = CFst(Box::new(fst)).into_raw_pointer();
        unsafe { *det_fst = fst_ptr };
        Ok(())
    })
}

/// # Safety
///
/// The pointers should be valid.
#[no_mangle]
pub unsafe extern "C" fn fst_determinize_with_config(
    ptr: *const CFst,
    config: *const CDeterminizeConfig,
    det_fst: *mut *const CFst,
) -> RUSTFST_FFI_RESULT {
    wrap(|| {
        let fst = get!(CFst, ptr);
        let vec_fst: &VectorFst<TropicalWeight> = fst
            .downcast_ref()
            .ok_or_else(|| anyhow!("Could not downcast to vector FST"))?;

        let det_config = unsafe {
            <CDeterminizeConfig as ffi_convert::RawBorrow<CDeterminizeConfig>>::raw_borrow(config)?
        };
        let fst: VectorFst<TropicalWeight> =
            determinize_with_config(vec_fst, det_config.as_rust()?)?;
        let fst_ptr = CFst(Box::new(fst)).into_raw_pointer();
        unsafe { *det_fst = fst_ptr };
        Ok(())
    })
}

//! Safe element-wise operation wrappers (Add, InplaceAdd, Mul).

use crate::error::{Result, check_aclnn};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::{AclDataType, AclOpExecutor, AclScalar};
use std::os::raw::c_void;

/// In-place addition: self += other * alpha.
///
/// # Arguments
/// - `stream`: execution stream
/// - `self_`: tensor to add to (modified in-place)
/// - `other`: tensor to add
/// - `alpha`: scale factor for other (typically 1.0)
pub fn inplace_add(
    stream: &Stream,
    self_: &AclTensor,
    other: &AclTensor,
    alpha: f32,
) -> Result<()> {
    // Create alpha scalar
    // Safety: `aclCreateScalar` takes a pointer to a valid f32 value and
    // copies the data internally; the reference lifetime is only this block.
    let alpha_scalar = unsafe {
        let val = alpha;
        aclnn_sys::common::aclCreateScalar(
            core::ptr::addr_of!(val).cast::<c_void>(),
            AclDataType::Float,
        )
    };
    if alpha_scalar.is_null() {
        return Err(crate::error::AscendError::InvalidArgument(
            "aclCreateScalar returned null".to_string(),
        ));
    }

    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = core::ptr::null_mut();

    // Stage 1: Get workspace size
    // Safety: `self_.raw()` and `other.raw()` are valid tensor handles;
    // `alpha_scalar` is a valid scalar handle; output pointers are mutable refs.
    let result = check_aclnn(unsafe {
        aclnn_sys::elementwise::aclnnInplaceAddGetWorkspaceSize(
            self_.raw(),
            other.raw(),
            alpha_scalar as *const AclScalar,
            &mut workspace_size,
            &mut executor,
        )
    });

    // Cleanup scalar on error
    if result.is_err() {
        // Safety: `alpha_scalar` was successfully created by `aclCreateScalar` above.
        unsafe {
            aclnn_sys::common::aclDestroyScalar(alpha_scalar as *const AclScalar);
        }
        return result;
    }

    // Allocate workspace
    let workspace = if workspace_size > 0 {
        Some(DeviceBuffer::alloc(workspace_size as usize)?)
    } else {
        None
    };

    let ws_ptr = workspace
        .as_ref()
        .map(|b| b.ptr())
        .unwrap_or(core::ptr::null_mut());

    // Stage 2: Execute
    // Safety: `executor` was initialized by GetWorkspaceSize; `ws_ptr` is valid
    // device memory (or null for zero-size); `stream.raw()` is a valid handle.
    let result = check_aclnn(unsafe {
        aclnn_sys::elementwise::aclnnInplaceAdd(ws_ptr, workspace_size, executor, stream.raw())
    });

    // Cleanup scalar
    // Safety: Same as above — `alpha_scalar` was created by `aclCreateScalar`.
    unsafe {
        aclnn_sys::common::aclDestroyScalar(alpha_scalar as *const AclScalar);
    }

    result
}

/// Element-wise multiplication: out = a * b.
///
/// # Arguments
/// - `stream`: execution stream
/// - `a`: first input tensor
/// - `b`: second input tensor
/// - `out`: output tensor (must be pre-allocated, same shape)
pub fn mul(stream: &Stream, a: &AclTensor, b: &AclTensor, out: &mut AclTensor) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = core::ptr::null_mut();

    // Safety: All tensor handles (`a.raw()`, `b.raw()`, `out.raw()`) are
    // non-null and valid. Output pointers are valid mutable references.
    check_aclnn(unsafe {
        aclnn_sys::elementwise::aclnnMulGetWorkspaceSize(
            a.raw(),
            b.raw(),
            out.raw(),
            &mut workspace_size,
            &mut executor,
        )
    })?;

    let workspace = if workspace_size > 0 {
        Some(DeviceBuffer::alloc(workspace_size as usize)?)
    } else {
        None
    };

    let ws_ptr = workspace
        .as_ref()
        .map(|b| b.ptr())
        .unwrap_or(core::ptr::null_mut());

    // Safety: `executor` was initialized by GetWorkspaceSize; `ws_ptr` points
    // to valid device memory (or null); `stream.raw()` is a valid stream handle.
    check_aclnn(unsafe {
        aclnn_sys::elementwise::aclnnMul(ws_ptr, workspace_size, executor, stream.raw())
    })
}

/// Type cast: out = cast(self, target_dtype).
///
/// # Arguments
/// - `stream`: execution stream
/// - `input`: source tensor
/// - `target_dtype`: desired output data type
/// - `out`: output tensor (must be pre-allocated with `target_dtype`)
pub fn cast(
    stream: &Stream,
    input: &AclTensor,
    target_dtype: AclDataType,
    out: &mut AclTensor,
) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = core::ptr::null_mut();

    // Safety: `input.raw()` and `out.raw()` are valid tensor handles;
    // `target_dtype` is a valid enum value; output pointers are mutable refs.
    check_aclnn(unsafe {
        aclnn_sys::elementwise::aclnnCastGetWorkspaceSize(
            input.raw(),
            target_dtype,
            out.raw(),
            &mut workspace_size,
            &mut executor,
        )
    })?;

    let workspace = if workspace_size > 0 {
        Some(DeviceBuffer::alloc(workspace_size as usize)?)
    } else {
        None
    };

    let ws_ptr = workspace
        .as_ref()
        .map(|b| b.ptr())
        .unwrap_or(core::ptr::null_mut());

    // Safety: `executor` was initialized by GetWorkspaceSize; `ws_ptr` points
    // to valid device memory (or null); `stream.raw()` is a valid stream handle.
    check_aclnn(unsafe {
        aclnn_sys::elementwise::aclnnCast(ws_ptr, workspace_size, executor, stream.raw())
    })
}

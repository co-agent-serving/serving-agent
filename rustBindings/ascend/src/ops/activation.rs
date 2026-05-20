//! Safe activation wrappers (SiLU, SwiGLU).

use crate::error::{Result, check_aclnn};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::AclOpExecutor;

/// SiLU activation: out = x * sigmoid(x).
///
/// # Arguments
/// - `stream`: execution stream
/// - `x`: input tensor [*, hidden_size]
/// - `out`: output tensor [*, hidden_size] (must be pre-allocated)
pub fn silu(stream: &Stream, x: &AclTensor, out: &mut AclTensor) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = core::ptr::null_mut();

    // Safety: All pointer arguments are valid: `x.raw()` and `out.raw()`
    // are non-null aclTensor handles; `workspace_size` and `executor` are
    // mutable references. The CANN API expects these exact types.
    check_aclnn(unsafe {
        aclnn_sys::activation::aclnnSiluGetWorkspaceSize(
            x.raw(),
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

    // Safety: `executor` was initialized by the GetWorkspaceSize call above;
    // `ws_ptr` points to valid device memory (or is null for zero-size);
    // `stream.raw()` is a valid stream handle.
    check_aclnn(unsafe {
        aclnn_sys::activation::aclnnSilu(ws_ptr, workspace_size, executor, stream.raw())
    })
}

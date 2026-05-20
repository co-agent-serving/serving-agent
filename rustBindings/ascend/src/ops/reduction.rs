//! Safe reduction wrappers (Softmax, ArgMax).

use crate::error::{Result, check_aclnn};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::AclOpExecutor;

/// Softmax: out = softmax(x, dim).
///
/// # Arguments
/// - `stream`: execution stream
/// - `x`: input tensor
/// - `dim`: dimension to apply softmax over
/// - `out`: output tensor (must be pre-allocated, same shape as x)
pub fn softmax(stream: &Stream, x: &AclTensor, dim: i64, out: &mut AclTensor) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = core::ptr::null_mut();

    // Safety: Tensor handles are valid; `dim` is a valid i64 argument.
    check_aclnn(unsafe {
        aclnn_sys::reduction::aclnnSoftmaxGetWorkspaceSize(
            x.raw(),
            dim,
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

    // Safety: `executor` was initialized by GetWorkspaceSize; `ws_ptr` is valid
    // device memory (or null); `stream.raw()` is a valid stream handle.
    check_aclnn(unsafe {
        aclnn_sys::reduction::aclnnSoftmax(ws_ptr, workspace_size, executor, stream.raw())
    })
}

/// ArgMax: find index of maximum value along a dimension.
///
/// # Arguments
/// - `stream`: execution stream
/// - `x`: input tensor
/// - `dim`: dimension to reduce
/// - `keepdim`: whether to keep the reduced dimension
/// - `out`: output tensor (Int64, reduced shape)
pub fn argmax(
    stream: &Stream,
    x: &AclTensor,
    dim: i64,
    keepdim: bool,
    out: &mut AclTensor,
) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = core::ptr::null_mut();

    // Safety: Tensor handles are valid; `dim` and `keepdim` are valid args.
    check_aclnn(unsafe {
        aclnn_sys::reduction::aclnnArgMaxGetWorkspaceSize(
            x.raw(),
            dim,
            keepdim,
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

    // Safety: Same pattern — `executor` from GetWorkspaceSize, valid pointers.
    check_aclnn(unsafe {
        aclnn_sys::reduction::aclnnArgMax(ws_ptr, workspace_size, executor, stream.raw())
    })
}

//! Safe wrappers for Paged Attention KV Cache operators.
//!
//! Provides the core decode-phase paged attention function:
//! `paged_attention_decode` — uses IncreFlashAttentionV4 with paged KV cache

use crate::error::{Result, check_aclnn};
use crate::memory::DeviceBuffer;
use crate::stream::Stream;
use crate::tensor::AclTensor;
use aclnn_sys::common::{AclFormat, AclIntArray, AclOpExecutor};

/// Perform decode-phase Paged Attention using IncreFlashAttentionV4.
///
/// KV cache is passed as raw device pointers + shape, NOT as AclTensor.
/// This allows us to create list-owned tensor descriptors that can be
/// safely destroyed by `aclDestroyTensorList` without double-free.
///
/// # Arguments
/// - `stream`: execution stream
/// - `query`: Q tensor [batch_size, 1, num_heads * head_dim] FP16
/// - `k_cache_ptr` / `v_cache_ptr`: raw device pointers to K/V cache memory
/// - `cache_shape`: BSH shape [1, total_slots, kv_hidden]
/// - `actual_seq_lengths`: raw pointer to int array of context lengths
/// - `num_heads`: number of Q heads
/// - `num_kv_heads`: number of KV heads (GQA)
/// - `scale`: 1/sqrt(head_dim)
/// - `block_size`: tokens per block (e.g., 16)
/// - `block_table`: [batch_size, max_blocks_per_seq] INT32
/// - `attention_out`: output tensor [batch_size, 1, num_heads * head_dim] FP16
pub fn paged_attention_decode(
    stream: &Stream,
    query: &AclTensor,
    k_cache_ptr: *mut std::os::raw::c_void,
    v_cache_ptr: *mut std::os::raw::c_void,
    cache_shape: &[i64],
    actual_seq_lengths: *const AclIntArray,
    num_heads: i64,
    num_kv_heads: i64,
    scale: f64,
    block_size: i64,
    block_table: &AclTensor,
    attention_out: &mut AclTensor,
) -> Result<()> {
    let mut workspace_size: u64 = 0;
    let mut executor: *mut AclOpExecutor = core::ptr::null_mut();

    let layout_cstr = std::ffi::CString::new("BSH").map_err(|_| {
        crate::error::AscendError::InvalidArgument("invalid layout string".to_string())
    })?;

    // Create dedicated raw aclTensor descriptors for the tensor list.
    // These are OWNED by the list and will be destroyed by aclDestroyTensorList.
    // They are NOT managed by Rust AclTensor wrappers, so no double-free.
    let ndim = cache_shape.len() as u64;
    // Safety: `aclCreateTensor` expects valid pointers to shape arrays and
    // a valid device pointer. `k_cache_ptr` is a valid device memory pointer
    // for the KV cache. The tensor is NOT managed by Rust — it will be added
    // to a list and destroyed by `aclDestroyTensorList`.
    let k_raw = unsafe {
        aclnn_sys::common::aclCreateTensor(
            cache_shape.as_ptr(),
            ndim,
            query.dtype(),
            core::ptr::null(),
            0,
            AclFormat::Nd,
            cache_shape.as_ptr(),
            ndim,
            k_cache_ptr,
        )
    };
    // Safety: Same as `k_raw` — `v_cache_ptr` is a valid device memory pointer
    // for the V cache. The tensor is list-owned.
    let v_raw = unsafe {
        aclnn_sys::common::aclCreateTensor(
            cache_shape.as_ptr(),
            ndim,
            query.dtype(),
            core::ptr::null(),
            0,
            AclFormat::Nd,
            cache_shape.as_ptr(),
            ndim,
            v_cache_ptr,
        )
    };

    // Safety: `k_raw` is a valid tensor handle; `aclCreateTensorList` copies
    // the pointer into the list. The list takes ownership of the tensor.
    let key_list = unsafe {
        aclnn_sys::paged_attention::aclCreateTensorList(
            core::ptr::addr_of!(k_raw) as *const *const aclnn_sys::common::AclTensor,
            1,
        )
    };
    // Safety: Same as `key_list` — `v_raw` is a valid tensor handle.
    let value_list = unsafe {
        aclnn_sys::paged_attention::aclCreateTensorList(
            core::ptr::addr_of!(v_raw) as *const *const aclnn_sys::common::AclTensor,
            1,
        )
    };

    // Stage 1: Get workspace size
    // Safety: All tensor/list handles are valid: `query.raw()`, `block_table.raw()`,
    // `key_list`, `value_list` were created above. Optional params are null.
    // `actual_seq_lengths` is a valid AclIntArray pointer. Output pointers are
    // valid mutable references.
    let status = unsafe {
        aclnn_sys::paged_attention::aclnnIncreFlashAttentionV4GetWorkspaceSize(
            query.raw(),
            key_list,
            value_list,
            core::ptr::null(), // pseShift
            core::ptr::null(), // attenMask
            actual_seq_lengths,
            core::ptr::null(), // dequantScale1
            core::ptr::null(), // quantScale1
            core::ptr::null(), // dequantScale2
            core::ptr::null(), // quantScale2
            core::ptr::null(), // quantOffset2
            core::ptr::null(), // antiquantScale
            core::ptr::null(), // antiquantOffset
            block_table.raw(),
            core::ptr::null(), // kvPaddingSize
            num_heads,
            scale,
            layout_cstr.as_ptr() as *const std::os::raw::c_char,
            num_kv_heads,
            block_size,
            0, // innerPrecise
            attention_out.raw(),
            &mut workspace_size,
            &mut executor,
        )
    };
    check_aclnn(status)?;

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
    // device memory (or null); `stream.raw()` is a valid stream handle.
    check_aclnn(unsafe {
        aclnn_sys::paged_attention::aclnnIncreFlashAttentionV4(
            ws_ptr,
            workspace_size,
            executor,
            stream.raw(),
        )
    })?;

    // Safely destroy tensor lists. aclDestroyTensorList destroys the container
    // AND the individual tensors (k_raw, v_raw). Since these raw tensors were
    // created specifically for the list (not shared with Rust AclTensor wrappers),
    // this is safe — no double-free.
    // Safety: `key_list` and `value_list` were created by `aclCreateTensorList`
    // above. `aclDestroyTensorList` destroys the container AND the individual
    // tensors (k_raw, v_raw). Since these raw tensors are not managed by Rust
    // AclTensor wrappers, there is no double-free.
    unsafe {
        aclnn_sys::paged_attention::aclDestroyTensorList(key_list);
        aclnn_sys::paged_attention::aclDestroyTensorList(value_list);
    }

    // Sync to ensure attention output is ready
    stream.synchronize()?;

    Ok(())
}

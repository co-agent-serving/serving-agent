//! Stream management with RAII.

use crate::error::{Result, check_acl};
use ascendcl_sys::AclrtStream;

/// RAII wrapper around `aclrtStream`.
///
/// Creates a stream on construction, destroys on drop.
#[derive(Debug)]
pub struct Stream {
    raw: AclrtStream,
}

impl Stream {
    /// Create a new stream on the current device.
    pub fn new() -> Result<Self> {
        let mut raw: AclrtStream = core::ptr::null_mut();
        // Safety: `raw` is a valid mutable reference; `aclrtCreateStream` is
        // thread-safe and will initialize `raw` with a new stream handle.
        check_acl(unsafe { ascendcl_sys::aclrtCreateStream(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Block the host until all tasks on this stream are complete.
    pub fn synchronize(&self) -> Result<()> {
        // Safety: `self.raw` is a valid stream handle created by
        // `aclrtCreateStream`. Synchronizing is safe on any valid stream.
        check_acl(unsafe { ascendcl_sys::aclrtSynchronizeStream(self.raw) })
    }

    /// Get the raw stream handle (for passing to aclnn operators).
    pub fn raw(&self) -> AclrtStream {
        self.raw
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Safety: `self.raw` is a valid stream handle from `aclrtCreateStream`;
            // null handles are filtered above. `aclrtDestroyStream` is the
            // canonical teardown call.
            unsafe {
                let _ = ascendcl_sys::aclrtDestroyStream(self.raw);
            }
        }
    }
}

// SAFETY: Streams can be sent between threads (actual synchronization
// is handled by the AscendCL runtime).
unsafe impl Send for Stream {}
// SAFETY: Same as `Send` — the raw handle is only used under the CANN
// runtime's own synchronization guarantees.
unsafe impl Sync for Stream {}

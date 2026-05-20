/// Build script for hccl-sys.
///
/// Links against libhccl.so and libhcomm.so from a CANN SDK installation.
/// Set ASCEND_HOME_PATH env var to the CANN SDK root (e.g. /usr/local/Ascend/ascend-toolkit/latest).
///
/// With the "stub" feature, linking is skipped — CANN is not required.
fn main() {
    let ascend_home_path = match std::env::var("ASCEND_HOME_PATH") {
        Ok(home) => home,
        Err(_) if cfg!(feature = "stub") => {
            // Stub mode: CANN not required, skip linking silently
            return;
        }
        Err(_) => {
            panic!(
                "CANN SDK not found (looking for libhcomm.so). Set ASCEND_HOME_PATH env var, \
                 or build with --features stub to skip linking (development without CANN)."
            );
        }
    };

    let lib_dir = format!("{ascend_home_path}/lib64");
    let lib_path = format!("{lib_dir}/libhcomm.so");

    if !std::path::Path::new(&lib_path).exists() {
        panic!(
            "libhcomm.so not found at {lib_path}. Is CANN SDK installed correctly?\n\
             ASCEND_HOME_PATH = {ascend_home_path}"
        );
    }

    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=dylib=hccl");
    println!("cargo:rustc-link-lib=dylib=hcomm");

    println!("cargo:rerun-if-env-changed=ASCEND_HOME_PATH");
    println!("cargo:rerun-if-changed=build.rs");
}

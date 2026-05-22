/// Build script for ascendcl-sys.
///
/// Links against libascendcl.so from a CANN SDK installation.
/// Set ASCEND_HOME_PATH env var to the CANN SDK root (e.g. /usr/local/Ascend/ascend-toolkit/latest).
///
/// With the "stub" feature, linking is skipped — CANN is not required.
fn main() {
    let ascend_home_path =
        std::env::var("ASCEND_HOME_PATH").expect("CANN SDK required: set ASCEND_HOME_PATH");

    let lib_dir = format!("{ascend_home_path}/lib64");
    let lib_path = format!("{lib_dir}/libascendcl.so");

    if !std::path::Path::new(&lib_path).exists() {
        panic!(
            "libascendcl.so not found at {lib_path}. Is CANN SDK installed correctly?\n\
             ASCEND_HOME_PATH = {ascend_home_path}"
        );
    }

    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=dylib=ascendcl");

    println!("cargo:rerun-if-env-changed=ASCEND_HOME_PATH");
    println!("cargo:rerun-if-changed=build.rs");
}

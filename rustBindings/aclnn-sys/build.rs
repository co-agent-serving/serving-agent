/// Build script for aclnn-sys.
///
/// Links against libopapi.so and libnnopbase.so from a CANN SDK installation.
/// Set ASCEND_HOME_PATH env var to the CANN SDK root (e.g. /usr/local/Ascend/ascend-toolkit/latest).
///
/// CANN shared libraries have many transitive dependencies (libruntime,
/// libge_runner, etc.) that are resolved at runtime via LD_LIBRARY_PATH.
fn main() {
    let ascend_home_path =
        std::env::var("ASCEND_HOME_PATH").expect("CANN SDK required: set ASCEND_HOME_PATH");

    let lib_dir = format!("{ascend_home_path}/lib64");

    // CANN's shared libs have many transitive deps (libruntime, libge_runner,
    // libplatform, etc.) that are resolved at runtime. Tell the linker to
    // not error on undefined symbols in shared libraries.
    println!("cargo:rustc-link-arg=-Wl,--allow-shlib-undefined");

    // Add RPATH so the runtime linker finds CANN libs
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");

    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=dylib=opapi");
    println!("cargo:rustc-link-lib=dylib=nnopbase");

    println!("cargo:rerun-if-env-changed=ASCEND_HOME_PATH");
    println!("cargo:rerun-if-changed=build.rs");
}

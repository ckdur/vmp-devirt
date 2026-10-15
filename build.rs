// Link system libatomic for unicorn-engine-sys static archive
// (__atomic_*_16 used by QEMU cpu helpers). Linux only: on macOS the
// atomics live in compiler-rt/libSystem and there is no libatomic.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=atomic");
    }
}

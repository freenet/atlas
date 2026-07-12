use std::{env, fs, path::Path};

/// Embed the committed contract WASM so the CLI's contract is byte-identical to
/// what gets deployed (a mismatch would compute a different contract key and
/// silently target a non-existent contract). The committed file is the single
/// source of truth; `contracts/index-contract/build-wasm.sh` reproducibly
/// rebuilds and refreshes it.
fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&out_dir).join("atlas_index_contract.wasm");
    let src = "../contracts/index-contract/atlas_index_contract.wasm";
    println!("cargo:rerun-if-changed={src}");
    if Path::new(src).exists() {
        fs::copy(src, &dest).expect("copy contract wasm");
    } else {
        fs::write(&dest, b"").expect("write placeholder wasm");
        println!(
            "cargo:warning=committed contract wasm missing; run contracts/index-contract/build-wasm.sh"
        );
    }
}

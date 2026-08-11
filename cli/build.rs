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

    // Lineage codegen: parse `legacy_contracts.toml`, decode + validate every
    // registered code hash AT BUILD TIME (a malformed hash is a build failure,
    // not a silently skipped probe), and emit `CONTRACT_LINEAGE` into
    // `$OUT_DIR/lineage.rs` for `src/migration.rs` to include. Also prints
    // `cargo:rerun-if-changed=legacy_contracts.toml`.
    freenet_migrate_build::codegen()
        .registry("legacy_contracts.toml")
        .emit()
        .expect("codegen legacy index-contract lineage");
}

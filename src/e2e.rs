//! End-to-end verification of the Rust inference path against the Python/ONNX
//! reference. Ignored by default; run explicitly with the model + onnxruntime
//! lib available:
//!
//! ```bash
//! ORT_DYLIB_PATH=vendored/onnxruntime/lib/libonnxruntime.so \
//!   cargo test --bin abfv --  --ignored e2e
//! ```
//!
//! It runs preprocess -> ort -> pdb for the example Fv, writes
//! `out/atom37_rust.f32` (same format as `workers/export_onnx.py` dumps) and
//! `out/complex_rust.pdb`. Numerical parity is then checked by
//! `workers/compare_atom37.py`.

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::inference::Predictor;
    use crate::pdb;
    use crate::preprocess::string_to_input;

    const HEAVY: &str = "EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPHYYGSSHWYFDVWGQGTLVTVSS";
    const LIGHT: &str = "DIQMTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK";

    #[test]
    #[ignore = "needs out/abb3.onnx + ORT_DYLIB_PATH"]
    fn predict_example_fv() {
        let model = Path::new("out/abb3.onnx");
        assert!(model.exists(), "run workers/export_onnx.py first to build {model:?}");

        let input = string_to_input(HEAVY, LIGHT);
        let mut predictor = Predictor::load(model).expect("load onnx");
        let out = predictor.predict(&input).expect("inference");

        assert_eq!(out.coords.shape(), &[input.n, 37, 3]);

        // Dump atom37 in the same format as the Python reference dump.
        let mut buf = format!("{}\n", input.n).into_bytes();
        for v in out.coords.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write("out/atom37_rust.f32", buf).expect("write dump");

        // Write the PDB via pdbtbx.
        let pdb_path = Path::new("out/complex_rust.pdb");
        pdb::write_complex(&out, &input, pdb_path).expect("write pdb");
        assert!(pdb_path.exists());

        eprintln!("[e2e] wrote out/atom37_rust.f32 and out/complex_rust.pdb (N={})", input.n);
    }
}

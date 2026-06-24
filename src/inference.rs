//! ONNX inference: sequence-derived tensors -> atom37 coordinates.
//!
//! Loads the exported ABodyBuilder3 graph (`workers/export_onnx.py` output) and
//! runs it via `ort`, returning the final-block atom37 coordinates and the
//! per-residue atom-presence mask. This replaces the Python worker subprocess
//! (`workers/predict.py`) entirely.

use std::path::Path;

use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;

use crate::preprocess::{ModelInput, PAIR_DIM, SINGLE_DIM};

/// Number of atom37 slots per residue.
pub const N_ATOM37: usize = 37;

#[derive(Debug, thiserror::Error)]
pub enum InferError {
    #[error("onnxruntime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("unexpected output shape for {name}: {shape:?}")]
    Shape { name: &'static str, shape: Vec<i64> },
}

pub struct Predictor {
    session: Session,
}

/// Model output for one Fv.
#[derive(Debug)]
pub struct Atom37 {
    /// `[N, 37, 3]` coordinates (masked positions are zero).
    pub coords: Array3<f32>,
    /// `[N, 37]` 1.0 where an atom exists for that residue, else 0.0.
    pub exists: Array2<f32>,
}

impl Predictor {
    pub fn load(model_path: &Path) -> Result<Self, InferError> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self { session })
    }

    /// Run inference for one preprocessed Fv.
    pub fn predict(&mut self, input: &ModelInput) -> Result<Atom37, InferError> {
        let n = input.n;

        // Build tensors via the core (shape, Vec) constructor rather than the
        // ndarray integration: ort pins its own ndarray version, and feeding it
        // our `ndarray` arrays trips a trait-mismatch across the two copies.
        // `.iter()` yields C-order, matching the row-major shapes below.
        let single = Tensor::from_array((
            [1usize, n, SINGLE_DIM],
            input.single.iter().copied().collect::<Vec<f32>>(),
        ))?;
        let pair = Tensor::from_array((
            [1usize, n, n, PAIR_DIM],
            input.pair.iter().copied().collect::<Vec<f32>>(),
        ))?;
        let aatype = Tensor::from_array((
            [1usize, n],
            input.aatype.iter().copied().collect::<Vec<i64>>(),
        ))?;

        let outputs = self.session.run(ort::inputs![
            "single" => single,
            "pair" => pair,
            "aatype" => aatype,
        ])?;

        let (cshape, cdata) = outputs["atom37"].try_extract_tensor::<f32>()?;
        let cdims: &[i64] = cshape;
        if cdims.len() != 3 || cdims[0] != n as i64 || cdims[1] != N_ATOM37 as i64 || cdims[2] != 3
        {
            return Err(InferError::Shape {
                name: "atom37",
                shape: cdims.to_vec(),
            });
        }
        let coords = Array3::from_shape_vec((n, N_ATOM37, 3), cdata.to_vec())
            .expect("atom37 shape checked above");

        let (mshape, mdata) = outputs["atom37_exists"].try_extract_tensor::<f32>()?;
        let mdims: &[i64] = mshape;
        if mdims.len() != 2 || mdims[0] != n as i64 || mdims[1] != N_ATOM37 as i64 {
            return Err(InferError::Shape {
                name: "atom37_exists",
                shape: mdims.to_vec(),
            });
        }
        let exists = Array2::from_shape_vec((n, N_ATOM37), mdata.to_vec())
            .expect("mask shape checked above");

        Ok(Atom37 { coords, exists })
    }
}

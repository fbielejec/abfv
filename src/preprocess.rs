//! Sequence -> model input tensors.
//!
//! Faithful Rust port of `abodybuilder3.utils.string_to_input` +
//! `ABDataset.single_and_double_from_datapoint` (rel_pos_dim=64,
//! edge_chain_feature=True, no PLM embeddings). Produces exactly the three
//! tensors the exported ONNX graph consumes:
//!
//! * `single` `[1, N, 23]`  = one_hot(aatype, 21) ++ one_hot(is_heavy, 2)
//! * `pair`   `[1, N, N, 132]` = one_hot(chain_code, 3) ++ one_hot(relpos, 129)
//! * `aatype` `[1, N]` (i64)
//!
//! where, for residues i (heavy: 0..H) and j,
//!   relpos    = clamp(idx[j] - idx[i], -64, 64) + 64        in 0..=128
//!   idx[k]    = k for heavy, 500 + (k - H) for light        (500-residue gap)
//!   chain_code = 2*h_i*h_j + (1-h_i)*(1-h_j)                in {0,1,2}

use ndarray::{Array2, Array3, Array4};

/// Amino-acid one-letter -> index, matching `restype_order_with_x`.
/// Order: ARNDCQEGHILKMFPSTWYV then X=20 (unknown).
const RESTYPES_X: &str = "ARNDCQEGHILKMFPSTWYV";

const REL_POS_DIM: i64 = 64;
const N_AA: usize = 21; // 20 + X
const N_CHAIN: usize = 2;
const N_RELPOS: usize = (2 * REL_POS_DIM + 1) as usize; // 129
const N_CHAIN_CODE: usize = 3;

pub const SINGLE_DIM: usize = N_AA + N_CHAIN; // 23
pub const PAIR_DIM: usize = N_CHAIN_CODE + N_RELPOS; // 132
/// Residue-index gap inserted between the heavy and light chains.
const CHAIN_GAP: i64 = 500;

#[derive(Debug)]
pub struct ModelInput {
    /// `[1, N, 23]`
    pub single: Array3<f32>,
    /// `[1, N, N, 132]`
    pub pair: Array4<f32>,
    /// `[1, N]`
    pub aatype: Array2<i64>,
    /// Number of heavy-chain residues (first `n_heavy` positions).
    pub n_heavy: usize,
    /// Total residues N.
    pub n: usize,
}

/// Map an uppercase amino-acid letter to its `restype_order_with_x` index.
/// Unknown / non-standard letters map to X (20), matching `restype_order_with_x`
/// which only contains the 20 standard residues plus X.
fn aa_index(c: char) -> i64 {
    match RESTYPES_X.find(c) {
        Some(i) => i as i64,
        None => 20, // X
    }
}

/// Build the model input from already-validated heavy/light sequences
/// (uppercase, standard residues; `X` permitted -> index 20).
pub fn string_to_input(heavy: &str, light: &str) -> ModelInput {
    let n_heavy = heavy.chars().count();
    let n_light = light.chars().count();
    let n = n_heavy + n_light;

    // Per-residue: amino-acid index, chain flag, gapped residue index.
    let mut aatype = vec![0i64; n];
    let mut is_heavy = vec![0u8; n];
    let mut res_idx = vec![0i64; n];

    for (i, c) in heavy.chars().enumerate() {
        aatype[i] = aa_index(c);
        is_heavy[i] = 1;
        res_idx[i] = i as i64;
    }
    for (k, c) in light.chars().enumerate() {
        let i = n_heavy + k;
        aatype[i] = aa_index(c);
        is_heavy[i] = 0;
        res_idx[i] = CHAIN_GAP + k as i64;
    }

    // single: one_hot(aatype,21) ++ one_hot(is_heavy,2)
    let mut single = Array3::<f32>::zeros((1, n, SINGLE_DIM));
    for i in 0..n {
        single[[0, i, aatype[i] as usize]] = 1.0;
        single[[0, i, N_AA + is_heavy[i] as usize]] = 1.0;
    }

    // pair: one_hot(chain_code,3) ++ one_hot(relpos,129)
    let mut pair = Array4::<f32>::zeros((1, n, n, PAIR_DIM));
    for i in 0..n {
        let hi = is_heavy[i] as i64;
        for j in 0..n {
            let hj = is_heavy[j] as i64;
            // chain_code in {0,1,2}: both-light=1, both-heavy=2, mixed=0
            let chain_code = (2 * hi * hj + (1 - hi) * (1 - hj)) as usize;
            pair[[0, i, j, chain_code]] = 1.0;

            // relpos = clamp(idx[j]-idx[i], -64, 64) + 64  -> 0..=128
            let rel = (res_idx[j] - res_idx[i]).clamp(-REL_POS_DIM, REL_POS_DIM) + REL_POS_DIM;
            pair[[0, i, j, N_CHAIN_CODE + rel as usize]] = 1.0;
        }
    }

    let aatype = Array2::from_shape_vec((1, n), aatype).expect("aatype shape");

    ModelInput {
        single,
        pair,
        aatype,
        n_heavy,
        n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_and_basic_onehots() {
        let inp = string_to_input("EVQ", "DIK");
        assert_eq!(inp.n, 6);
        assert_eq!(inp.n_heavy, 3);
        assert_eq!(inp.single.shape(), &[1, 6, 23]);
        assert_eq!(inp.pair.shape(), &[1, 6, 6, 132]);
        assert_eq!(inp.aatype.shape(), &[1, 6]);

        // E=6, V=19, Q=5 ; D=3, I=9, K=11  (restype_order_with_x)
        assert_eq!(
            inp.aatype.as_slice().unwrap(),
            &[6, 19, 5, 3, 9, 11]
        );

        // single is a sum of two one-hots -> exactly two ones per residue.
        for i in 0..inp.n {
            let row = inp.single.slice(ndarray::s![0, i, ..]);
            let ones: f32 = row.sum();
            assert_eq!(ones, 2.0, "residue {i} should have aa + chain one-hots");
            assert_eq!(row[inp.aatype[[0, i]] as usize], 1.0);
        }
        // chain flag: heavy -> idx 21, light -> idx 22
        assert_eq!(inp.single[[0, 0, 22]], 1.0); // heavy
        assert_eq!(inp.single[[0, 5, 21]], 1.0); // light
    }

    #[test]
    fn pair_features_are_two_onehots() {
        let inp = string_to_input("EVQ", "DIK");
        // each pair entry = chain one-hot (3) + relpos one-hot (129) = two ones
        for i in 0..inp.n {
            for j in 0..inp.n {
                let v = inp.pair.slice(ndarray::s![0, i, j, ..]);
                assert_eq!(v.sum(), 2.0, "pair[{i},{j}] should be two one-hots");
            }
        }
    }

    #[test]
    fn chain_code_values() {
        let inp = string_to_input("EVQ", "DIK"); // heavy 0..3, light 3..6
        // both heavy -> code 2
        assert_eq!(inp.pair[[0, 0, 1, 2]], 1.0);
        // both light -> code 1
        assert_eq!(inp.pair[[0, 4, 5, 1]], 1.0);
        // mixed -> code 0
        assert_eq!(inp.pair[[0, 0, 5, 0]], 1.0);
    }

    #[test]
    fn relpos_diagonal_and_gap() {
        let inp = string_to_input("EVQ", "DIK");
        // diagonal: rel = 0 -> bucket 64, stored at offset 3+64
        assert_eq!(inp.pair[[0, 2, 2, 3 + 64]], 1.0);
        // heavy res 0 vs light res 3: idx 0 vs 500 -> diff 500 clamps to +64
        assert_eq!(inp.pair[[0, 0, 3, 3 + 128]], 1.0);
        // light res 3 vs heavy res 0: diff -500 clamps to -64 -> bucket 0 (offset 3)
        assert_eq!(inp.pair[[0, 3, 0, 3]], 1.0);
    }
}

//! Write a complex PDB from atom37 coordinates.
//!
//! Replaces `abodybuilder3.utils.output_to_pdb` (minus the pdbfixer/OpenMM
//! `fix_pdb` cleanup, which only adds hydrogens/missing atoms that the
//! downstream FreeSASA heavy-atom pass ignores).
//!
//! The records are written byte-for-byte in the same column layout as openfold
//! `protein.to_pdb`, because the downstream is strict about columns:
//!   * FreeSASA's NACCESS `.rsa` writer needs a right-justified residue number
//!     (a left-justified one merges chain+resnum into e.g. `H0`, which
//!     `main.rs::parse_rsa` cannot split).
//!   * `split_chain` reads the chain id at byte 21 (column 22).
//!
//! NOTE: pdbtbx 0.12's PDB writer left-justifies serial/resSeq fields
//! (`save/pdb.rs` `get_line` uses `{trimmed:length$}`), producing
//! non-conforming records that FreeSASA mis-parses — hence the hand-rolled
//! formatter here rather than `pdbtbx::save_pdb`.
//!
//! Layout: heavy residues -> chain `H`, light -> chain `L`; residue numbers are
//! the 0-based global index (`arange(N)`); atom names/order follow
//! `residue_constants.atom_types`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::inference::Atom37;
use crate::preprocess::ModelInput;

/// atom37 slot -> atom name (`residue_constants.atom_types`).
const ATOM_TYPES: [&str; 37] = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2", "ND1",
    "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2",
    "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
];

/// aatype index (restype order ARNDCQEGHILKMFPSTWYV) -> 3-letter name; 20 = UNK.
const RESTYPE_3: [&str; 21] = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
];

#[derive(Debug, thiserror::Error)]
pub enum PdbError {
    #[error("failed to write PDB {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

/// Format one ATOM record, matching openfold `to_pdb` column-for-column.
fn atom_line(serial: usize, name: &str, res3: &str, chain: char, res_seq: usize, p: [f32; 3]) -> String {
    // PDB atom-name quirk: 4-char names start at col 13, shorter names get a
    // leading space (so 1-3 char names occupy cols 14-16).
    let name_field = if name.len() == 4 {
        name.to_string()
    } else {
        format!(" {name:<3}")
    };
    let element = &name[0..1]; // C, N, O, S
    let mut s = String::with_capacity(80);
    let _ = write!(
        s,
        "{:<6}{:>5} {:<4}{:>1}{:>3} {:>1}{:>4}{:>1}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}{:>2}",
        "ATOM", serial, name_field, "", res3, chain, res_seq, "", p[0], p[1], p[2], 1.0_f64, 0.0_f64, element, "",
    );
    s
}

/// Build a complex PDB (chains H + L) and write it to `path`.
pub fn write_complex(atom37: &Atom37, input: &ModelInput, path: &Path) -> Result<PathBuf, PdbError> {
    let mut out = String::new();
    let mut serial = 1usize;

    for (chain, range) in [('H', 0..input.n_heavy), ('L', input.n_heavy..input.n)] {
        let mut last_res3 = "UNK";
        let mut last_seq = 0usize;
        for i in range {
            let aa = input.aatype[[0, i]] as usize;
            let res3 = RESTYPE_3[aa.min(20)];
            for (a, name) in ATOM_TYPES.iter().enumerate() {
                if atom37.exists[[i, a]] < 0.5 {
                    continue;
                }
                let p = [
                    atom37.coords[[i, a, 0]],
                    atom37.coords[[i, a, 1]],
                    atom37.coords[[i, a, 2]],
                ];
                out.push_str(&atom_line(serial, name, res3, chain, i, p));
                out.push('\n');
                serial += 1;
            }
            last_res3 = res3;
            last_seq = i;
        }
        // TER closes the chain (matches openfold's TER record).
        let _ = writeln!(
            out,
            "{:<6}{:>5}      {:>3} {:>1}{:>4}",
            "TER", serial, last_res3, chain, last_seq
        );
        serial += 1;
    }
    out.push_str("END\n");

    std::fs::write(path, out).map_err(|source| PdbError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(path.to_path_buf())
}

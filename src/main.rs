use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, prelude::*};

use clap::{Parser, Subcommand};
use thiserror::Error;

const STANDARD_AA: &str = "ACDEFGHIKLMNPQRSTVWY";

const DEFAULT_PYTHON: &str = "/home/filip/CloudStation/Python/abodybuilder3/.venv/bin/python";
const DEFAULT_SCRIPT: &str = "workers/predict.py";
const DEFAULT_CHECKPOINT: &str =
    "/home/filip/CloudStation/Python/abodybuilder3/output/plddt-loss/best_second_stage.ckpt";
const DEFAULT_OUT_DIR: &str = "out";
const DEFAULT_OUT_FILE: &str = "complex.pdb";
const DEFAULT_FREESASA: &str = "/home/filip/CloudStation/Python/freesasa/src/freesasa";
const DEFAULT_FORMAT: &str = "rsa";

#[derive(Parser, Debug)]
#[command(name = "abfv")]
struct Args {
    /// Heavy-chain (VH) amino-acid sequence.
    #[arg(long, value_name = "SEQ", required_unless_present = "heavy_file")]
    heavy: Option<String>,

    /// File with the VH sequence (plain text or FASTA).
    #[arg(long, value_name = "PATH", conflicts_with = "heavy")]
    heavy_file: Option<PathBuf>,

    /// Light-chain (VL) amino-acid sequence.
    #[arg(long, value_name = "SEQ", required_unless_present = "light_file")]
    light: Option<String>,

    /// File with the VL sequence (plain text or FASTA).
    #[arg(long, value_name = "PATH", conflicts_with = "light")]
    light_file: Option<PathBuf>,

    /// Contact threshold (percent for `first`, percentage points for `second`).
    #[arg(long, default_value_t = 0.1)]
    threshold: f64,

    /// Allow `X` (unknown) residues. Off by default.
    #[arg(long)]
    allow_unknown: bool,

    /// Output directory for all tools (predict + freesasa).
    #[arg(long, value_name = "DIR", default_value = DEFAULT_OUT_DIR)]
    out_dir: PathBuf,

    /// `predict` step + its options (omit to use defaults). Chains into `freesasa`.
    #[command(subcommand)]
    predict: Option<PredictCmd>,

    // /// Skip the visualization step.
    // #[arg(long)]
    // no_viz: bool,
    /// Tracing filter that honors standard RUST_LOG, with sensible defaults
    #[arg(long, env = "RUST_LOG", default_value = "info,api=debug,common=debug")]
    log: String,
}

#[derive(Subcommand, Debug)]
enum PredictCmd {
    /// Fold the Fv with the ABodyBuilder3 worker script.
    Predict(PredictArgs),
}

// TODO : fix seed
#[derive(clap::Args, Debug)]
struct PredictArgs {
    /// Python interpreter (defaults to the ABodyBuilder3 venv).
    #[arg(long, value_name = "PATH", default_value = DEFAULT_PYTHON)]
    python: PathBuf,

    /// Worker script that wraps the predictor.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_SCRIPT)]
    script: PathBuf,

    /// ABodyBuilder3 checkpoint (.ckpt).
    #[arg(long, value_name = "PATH", default_value = DEFAULT_CHECKPOINT)]
    checkpoint: PathBuf,

    /// Output PDB file name (written under the top-level `--out-dir`).
    #[arg(long, value_name = "FILE", default_value = DEFAULT_OUT_FILE)]
    out_file: String,

    /// `freesasa` step + its options (omit to use defaults).
    #[command(subcommand)]
    freesasa: Option<FreesasaCmd>,
}

#[derive(Subcommand, Debug)]
enum FreesasaCmd {
    /// Run FreeSASA on the predicted PDBs.
    Freesasa(FreesasaArgs),
}

impl Default for PredictArgs {
    fn default() -> Self {
        Self {
            python: DEFAULT_PYTHON.into(),
            script: DEFAULT_SCRIPT.into(),
            checkpoint: DEFAULT_CHECKPOINT.into(),
            out_file: DEFAULT_OUT_FILE.into(),
            freesasa: None,
        }
    }
}

/// Options for the FreeSASA shell-out.
#[derive(clap::Args, Debug)]
struct FreesasaArgs {
    /// FreeSASA binary.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_FREESASA)]
    binary: PathBuf,

    /// Output format, passed as `--format=<FORMAT>`.
    #[arg(long, default_value = DEFAULT_FORMAT)]
    format: String,
}

impl Default for FreesasaArgs {
    fn default() -> Self {
        Self {
            binary: DEFAULT_FREESASA.into(),
            format: DEFAULT_FORMAT.into(),
        }
    }
}

#[derive(Debug, Error)]
enum AbfvError {
    #[error("invalid {chain} sequence: {reason}")]
    InvalidSequence { chain: &'static str, reason: String },

    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{tool} failed (exit {code}): {stderr}")]
    Subprocess {
        tool: String,
        code: i32,
        stderr: String,
    },

    #[error("failed to parse {what}: {why}")]
    Parse { what: String, why: String },

    #[error("failed to read: {why}")]
    Data { why: String },
}

// composite key - chain (H/L), residue_name (e.g., ASP, ILE), residue_number
pub type ResidueKey = (char, String, u32);

/// Solvent Accessible SUrface Area entry as parsed from freeSASA .rsa file
#[derive(Debug)]
pub struct ResidueSasa {
    /// ABS Absolute accessibility value
    pub side_chain_absolute: f64,
}

struct ContactRow {
    chain: char,
    residue_name: String,
    residue_number: u32,
    iso_side_chain_absolute: f64,
    complex_side_chain_absolute: f64,
    contact_metric: f64,
    is_contact: bool,
}

fn clean_and_validate(
    raw: &str,
    chain: &'static str,
    allow_unknown: bool,
) -> Result<String, AbfvError> {
    let mut seq: String = raw
        .lines()
        .filter(|l| !l.starts_with('>'))
        .flat_map(|l| l.chars())
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_uppercase();

    if seq.ends_with('*') {
        seq.pop();
    }

    if seq.is_empty() {
        return Err(AbfvError::InvalidSequence {
            chain,
            reason: "empty after normalization".into(),
        });
    }

    for (i, c) in seq.chars().enumerate() {
        let pos = i + 1;

        if c == 'X' {
            if allow_unknown {
                continue;
            }

            return Err(AbfvError::InvalidSequence {
                chain,
                reason: format!("'X' at position {pos} (pass --allow-unknown to permit)"),
            });
        }

        if !STANDARD_AA.contains(c) {
            return Err(AbfvError::InvalidSequence {
                chain,
                reason: format!("invalid residue '{c}' at position {pos}"),
            });
        }
    }

    Ok(seq)
}

fn read_seq(direct: Option<String>, file: Option<PathBuf>) -> Result<String, AbfvError> {
    match (direct, file) {
        (Some(s), _) => Ok(s),
        (None, Some(p)) => Ok(std::fs::read_to_string(p)?),
        (None, None) => unreachable!("should have never gotten here!"),
    }
}

fn run(args: Args) -> Result<(), AbfvError> {
    info!(?args, "Starting");

    let heavy = clean_and_validate(
        &read_seq(args.heavy, args.heavy_file)?,
        "VH",
        args.allow_unknown,
    )?;

    let light = clean_and_validate(
        &read_seq(args.light, args.light_file)?,
        "VL",
        args.allow_unknown,
    )?;

    info!("Inputs validated");

    let out_dir = args.out_dir;

    let mut predict = match args.predict {
        Some(PredictCmd::Predict(p)) => p,
        None => PredictArgs::default(),
    };

    let freesasa = match predict.freesasa.take() {
        Some(FreesasaCmd::Freesasa(f)) => f,
        None => FreesasaArgs::default(),
    };

    // 1. predict structure -> calls workers/predict.py -> out/complex.pdb (complex chain H/L)
    let complex_pdb = predict_structure(&predict, &out_dir, &heavy, &light)?;
    info!(path = %complex_pdb.display(), "Predicted Fv structure");

    // 2. split chains -> produces heavy.pdb / light.pdb (isolated chains)
    info!("Splitting complex into single-chain PDBs");
    let heavy_pdb = split_chain(&complex_pdb, 'H', "heavy.pdb")?;
    // info!(path = %dst.display(), "wrote chain {chain}");
    let light_pdb = split_chain(&complex_pdb, 'L', "light.pdb")?;
    info!(heavy = %heavy_pdb.display(), light = %light_pdb.display(), "Wote split chains");

    // 3. call FreeSASA on each chain -> produces per-residue side-chain SASA (complex, heavy, light)
    for (pdb_in_file, rsa_out_file) in [
        (&complex_pdb, "complex.rsa"),
        (&heavy_pdb, "heavy.rsa"),
        (&light_pdb, "light.rsa"),
    ] {
        info!(pdb = %pdb_in_file.display(), "Running FreeSASA");
        let rsa = run_freesasa(&freesasa, pdb_in_file, &out_dir.join(rsa_out_file))?;
        info!(path = %rsa.display(), "Wrote RSA file");
    }

    // 4. dSASA + contacts -> metric @ threshold
    delta_sasa(
        out_dir.join("complex.rsa"),
        out_dir.join("heavy.rsa"),
        out_dir.join("light.rsa"),
        args.threshold,
    )?;

    // 5. write contacts.csv / contacts.json
    // 6. visualize           -> workers/visualize.py (TODO --no-viz)

    Ok(())
}

fn main() -> Result<(), AbfvError> {
    let args = Args::parse();
    init_tracing(&args.log);
    run(args)?;
    Ok(())
}

fn delta_sasa(
    complex_rsa: PathBuf,
    heavy_rsa: PathBuf,
    light_rsa: PathBuf,
    contact_threshold: f64,
) -> Result<(), AbfvError> {
    let complex = parse_rsa(&complex_rsa)?;

    let mut isolated = parse_rsa(&heavy_rsa)?;
    isolated.extend(parse_rsa(&light_rsa)?);

    assert_eq!(
        complex.len(),
        isolated.len(),
        "isolated and complex should be equal!"
    );

    let mut contact_rows: Vec<ContactRow> = vec![];

    for (key @ (chain, residue_name, residue_number), complex_residue) in &complex {
        // println!("complex residue: {complex_residue:?}");
        // println!("iso residue: {iso_residue:?}");

        let iso_residue = isolated.get(key).ok_or(AbfvError::Data {
            why: format!("isolated map does not contain residue: {:?}", key),
        })?;

        let delta = iso_residue.side_chain_absolute - complex_residue.side_chain_absolute;

        // first
        let contact_metric = if delta > 0.0 {
            delta / iso_residue.side_chain_absolute
        } else {
            f64::NAN
        };

        contact_rows.push(ContactRow {
            chain: *chain,
            residue_name: residue_name.to_string(),
            residue_number: *residue_number,
            iso_side_chain_absolute: iso_residue.side_chain_absolute,
            complex_side_chain_absolute: complex_residue.side_chain_absolute,
            contact_metric,
            is_contact: contact_metric >= contact_threshold,
        });
    }

    Ok(())
}

fn parse_rsa(rsa: &Path) -> Result<HashMap<ResidueKey, ResidueSasa>, AbfvError> {
    let text = fs::read_to_string(rsa)?;

    let err = |line_no: usize, reason: String| AbfvError::Parse {
        what: format!("RSA file {} (line {line_no})", rsa.display()),
        why: reason,
    };

    // RES resname chain resnum  all(abs rel)  side(abs rel)  main(..) nonpolar(..) polar(..)
    // tokens:  0    1     2      3     4   5      6    7      ...
    text.lines()
        .enumerate()
        .filter(|(_, l)| l.starts_with("RES "))
        .map(|(i, l)| {
            let line_no = i + 1;
            let cols: Vec<&str> = l.split_whitespace().collect();

            if cols.len() < 8 {
                return Err(err(
                    line_no,
                    format!("expected at least 8 columns, found {}: {l:?}", cols.len()),
                ));
            }

            let residue_name = cols[1].to_string();

            let chain = cols[2]
                .chars()
                .next()
                .ok_or_else(|| err(line_no, format!("empty chain id (column 3) in {l:?}")))?;

            let residue_number = cols[3]
                .parse::<u32>()
                .map_err(|e| err(line_no, format!("residue number '{}': {e}", cols[3])))?;

            let side_chain_absolute = cols[6]
                .parse::<f64>()
                .map_err(|e| err(line_no, format!("side-chain ABS '{}': {e}", cols[6])))?;

            let key = (chain, residue_name.clone(), residue_number);

            Ok((
                key,
                ResidueSasa {
                    side_chain_absolute,
                },
            ))
        })
        .collect()
}

fn run_freesasa(args: &FreesasaArgs, in_pdb: &Path, out_rsa: &Path) -> Result<PathBuf, AbfvError> {
    let output = Command::new(&args.binary)
        .arg(format!("--format={}", args.format))
        .arg(format!("--output={}", out_rsa.display()))
        .arg(in_pdb)
        .output()?;

    if !output.status.success() {
        return Err(AbfvError::Subprocess {
            tool: args.binary.display().to_string(),
            code: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    Ok(out_rsa.to_path_buf())
}

fn split_chain(complex_pdb: &Path, chain: char, name: &str) -> Result<PathBuf, AbfvError> {
    let pdb = std::fs::read_to_string(complex_pdb)?;
    let dst = complex_pdb.with_file_name(name);

    let mut out = String::new();
    for line in pdb.lines() {
        let kept =
            (line.starts_with("ATOM") || line.starts_with("HETATM") || line.starts_with("TER"))
                && line.as_bytes().get(21) == Some(&(chain as u8));

        if kept {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.push_str("END\n");

    std::fs::write(&dst, out)?;
    Ok(dst)
}

fn predict_structure(
    predict: &PredictArgs,
    out_dir: &Path,
    heavy: &str,
    light: &str,
) -> Result<PathBuf, AbfvError> {
    info!("Predicting Fv structure with ABodyBuilder3");

    let status = Command::new(&predict.python)
        .arg(&predict.script)
        .arg(light)
        .arg(heavy)
        .arg("--checkpoint")
        .arg(&predict.checkpoint)
        .arg("--out-dir")
        .arg(out_dir)
        .arg("--out-file")
        .arg(&predict.out_file)
        .status()?;

    if !status.success() {
        return Err(AbfvError::Subprocess {
            tool: predict.script.display().to_string(),
            code: status.code().unwrap_or(-1),
            stderr: "see output above".into(),
        });
    }

    Ok(out_dir.join(&predict.out_file))
}

fn init_tracing(filter: &str) {
    let filter = EnvFilter::try_new(filter).unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .init();
}

use std::path::PathBuf;
use std::process::Command;
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, prelude::*};

use clap::{Parser, Subcommand, ValueEnum};
use thiserror::Error;

const DEFAULT_PYTHON: &str = "/home/filip/CloudStation/Python/abodybuilder3/.venv/bin/python";
const DEFAULT_SCRIPT: &str = "workers/predict.py";
const DEFAULT_CHECKPOINT: &str =
    "/home/filip/CloudStation/Python/abodybuilder3/output/plddt-loss/best_second_stage.ckpt";
const DEFAULT_OUT_DIR: &str = "out";
const DEFAULT_OUT_FILE: &str = "complex.pdb";

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Metric {
    /// Relative drop in side-chain SASA: (iso − cplx) / iso ≥ threshold%.
    First,
    /// Drop in RSA percentage points: rel_iso − rel_cplx ≥ threshold.
    Second,
    /// Report and flag under both.
    Both,
}

#[derive(Parser, Debug)]
#[command(
    name = "abfv",
    version,
    about = "Antibody Fv VH–VL interface detection (wraps ABodyBuilder3 + FreeSASA)"
)]
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

    /// Contact metric (which reading of the >= 10% rule).
    #[arg(long, value_enum, default_value_t = Metric::Both)]
    metric: Metric,

    /// Contact threshold (percent for `first`, percentage points for `second`).
    #[arg(long, default_value_t = 10.0)]
    threshold: f64,

    /// Allow `X` (unknown) residues. Off by default: `X` → a structural gap in the model.
    #[arg(long)]
    allow_unknown: bool,

    /// Predictor backend + its options (defaults to `predict` with ABodyBuilder3).
    #[command(subcommand)]
    predictor: Option<Predictor>,

    // /// Keep intermediate single-chain PDBs.
    // #[arg(long)]
    // keep_intermediate: bool,

    // /// Skip the visualization step.
    // #[arg(long)]
    // no_viz: bool,

    // /// Verbose logging to stderr.
    // #[arg(short, long)]
    // verbose: bool,
    /// Tracing filter that honors standard RUST_LOG, with sensible defaults
    #[arg(long, env = "RUST_LOG", default_value = "info,api=debug,common=debug")]
    log: String,
}

/// Structure-prediction backend (each variant is one predictor + its options).
#[derive(Subcommand, Debug)]
enum Predictor {
    /// Fold the Fv with the ABodyBuilder3 worker script.
    Predict(PredictArgs),
}

/// Options for the ABodyBuilder3 worker (`workers/predict.py`).
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

    /// Output directory.
    #[arg(long, value_name = "DIR", default_value = DEFAULT_OUT_DIR)]
    out_dir: PathBuf,

    /// Output PDB file name (written under `--out-dir`).
    #[arg(long, value_name = "FILE", default_value = DEFAULT_OUT_FILE)]
    out_file: String,
}

impl Default for PredictArgs {
    fn default() -> Self {
        Self {
            python: DEFAULT_PYTHON.into(),
            script: DEFAULT_SCRIPT.into(),
            checkpoint: DEFAULT_CHECKPOINT.into(),
            out_dir: DEFAULT_OUT_DIR.into(),
            out_file: DEFAULT_OUT_FILE.into(),
        }
    }
}

#[derive(Debug, Error)]
// #[allow(dead_code)] // TODO
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

    #[error("failed to parse {what}: {reason}")]
    #[allow(dead_code)] // TODO: constructed once FreeSASA/contacts parsing lands
    Parse { what: String, reason: String },
}

/// The 20 standard amino acids (one-letter codes).
const STANDARD_AA: &str = "ACDEFGHIKLMNPQRSTVWY";

/// Normalize then validate a raw sequence
fn clean_and_validate(
    raw: &str,
    chain: &'static str,
    allow_unknown: bool,
) -> Result<String, AbfvError> {
    // normalize
    let mut seq: String = raw
        .lines()
        // drop FASTA header(s)
        .filter(|l| !l.starts_with('>'))
        .flat_map(|l| l.chars())
        // drop whitespaces
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        // just being nice
        .to_uppercase();

    // drop stop marker
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

    let predict = match args.predictor {
        Some(Predictor::Predict(p)) => p,
        None => PredictArgs::default(),
    };

    // 1. predict structure   -> workers/predict.py -> out/complex.pdb (chains H/L)
    let complex_pdb = predict_structure(&predict, &heavy, &light)?;
    info!(path = %complex_pdb.display(), "Predicted Fv structure");

    // 2. split chains        -> heavy.pdb / light.pdb
    let (heavy_pdb, light_pdb) = split_chains(complex_pdb.clone())?;
    info!(heavy = %heavy_pdb.display(), light = %light_pdb.display(), "Split chains");

    // 3. FreeSASA x3         -> per-residue side-chain SASA (complex, heavy, light)
    run_freesasa(complex_pdb)?;
    run_freesasa(heavy_pdb)?;
    run_freesasa(light_pdb)?;
    // 4. ΔSASA + contacts    -> metric first/second @ threshold
    // 5. write contacts.csv / contacts.json
    // 6. visualize           -> workers/visualize.py (unless --no-viz)

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_tracing(&args.log);
    run(args)?;
    Ok(())
}

fn run_freesasa(pdb: PathBuf) -> Result<PathBuf, AbfvError> {
    todo!()
}

fn split_chains(complex_pdb: PathBuf) -> Result<(PathBuf, PathBuf), AbfvError> {
    info!("Splitting complex into single-chain PDBs");

    let pdb = std::fs::read_to_string(&complex_pdb)?;

    // Write the records for one chain to `<name>` next to the complex; return its path.
    let write_chain = |chain: char, name: &str| -> Result<PathBuf, AbfvError> {
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
        info!(path = %dst.display(), "wrote chain {chain}");
        Ok(dst)
    };

    let heavy_pdb = write_chain('H', "heavy.pdb")?;
    let light_pdb = write_chain('L', "light.pdb")?;

    Ok((heavy_pdb, light_pdb))
}

fn predict_structure(
    predict: &PredictArgs,
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
        .arg(&predict.out_dir)
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

    Ok(predict.out_dir.join(&predict.out_file))
}

fn init_tracing(filter: &str) {
    let filter = EnvFilter::try_new(filter).unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .init();
}

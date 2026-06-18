use std::path::{Path, PathBuf};
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
const DEFAULT_FREESASA: &str = "/home/filip/CloudStation/Python/freesasa/src/freesasa";
const DEFAULT_FORMAT: &str = "rsa";

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Metric {
    /// Relative drop in side-chain SASA: (iso − cplx) / iso >= threshold.
    First,
    /// Drop in RSA percentage points: rel_iso − rel_cplx >= threshold.
    Second,
    /// Report and flag under both.
    Both,
}

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

    /// Contact metric (which reading of the >= 10% rule).
    #[arg(long, value_enum, default_value_t = Metric::Both)]
    metric: Metric,

    /// Contact threshold (percent for `first`, percentage points for `second`).
    #[arg(long, default_value_t = 10.0)]
    threshold: f64,

    /// Allow `X` (unknown) residues. Off by default.
    #[arg(long)]
    allow_unknown: bool,

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

/// `predict` subcommand keyword (chains into `freesasa`).
#[derive(Subcommand, Debug)]
enum PredictCmd {
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

    /// `freesasa` step + its options (omit to use defaults).
    #[command(subcommand)]
    freesasa: Option<FreesasaCmd>,
}

/// `freesasa` subcommand keyword (nested under `predict`).
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
            out_dir: DEFAULT_OUT_DIR.into(),
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

    #[error("failed to parse {what}: {reason}")]
    #[allow(dead_code)] // TODO: constructed once FreeSASA/contacts parsing lands
    Parse { what: String, reason: String },
}

const STANDARD_AA: &str = "ACDEFGHIKLMNPQRSTVWY";

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

    let mut predict = match args.predict {
        Some(PredictCmd::Predict(p)) => p,
        None => PredictArgs::default(),
    };

    let freesasa = match predict.freesasa.take() {
        Some(FreesasaCmd::Freesasa(f)) => f,
        None => FreesasaArgs::default(),
    };

    // 1. predict structure   -> workers/predict.py -> out/complex.pdb (chains H/L)
    let complex_pdb = predict_structure(&predict, &heavy, &light)?;
    info!(path = %complex_pdb.display(), "Predicted Fv structure");

    // 2. split chains        -> heavy.pdb / light.pdb
    info!("Splitting complex into single-chain PDBs");
    let heavy_pdb = split_chain(&complex_pdb, 'H', "heavy.pdb")?;
    // info!(path = %dst.display(), "wrote chain {chain}");
    let light_pdb = split_chain(&complex_pdb, 'L', "light.pdb")?;
    info!(heavy = %heavy_pdb.display(), light = %light_pdb.display(), "Split chains");

    // 3. FreeSASA x3         -> per-residue side-chain SASA (complex, heavy, light)
    for pdb in [&complex_pdb, &heavy_pdb, &light_pdb] {
        info!(pdb = %pdb.display(), "Running FreeSASA");
        let rsa = run_freesasa(&freesasa, pdb)?;
        info!(path = %rsa.display(), "wrote RSA file");
    }
    // 4. ΔSASA + contacts    -> metric first/second @ threshold
    // 5. write contacts.csv / contacts.json
    // 6. visualize           -> workers/visualize.py (TODO --no-viz)

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_tracing(&args.log);
    run(args)?;
    Ok(())
}

fn run_freesasa(fs: &FreesasaArgs, pdb: &Path) -> Result<PathBuf, AbfvError> {
    let output = Command::new(&fs.binary)
        .arg(format!("--format={}", fs.format))
        .arg(pdb)
        .output()?;

    if !output.status.success() {
        return Err(AbfvError::Subprocess {
            tool: fs.binary.display().to_string(),
            code: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    let rsa_path = pdb.with_extension("rsa");
    std::fs::write(&rsa_path, &output.stdout)?;

    Ok(rsa_path)
}

/// Write the records for one `chain` of `complex_pdb` to `name` next to it (the
/// 'isolated' reference state); returns the written path. Keeps ATOM/HETATM/TER
/// records whose chain id (column 22, 0-based index 21) matches, then a trailing `END`.
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

//! abfv - antibody Fv VH-VL interface detection.
//!
//! Steps:
//! - Argument parsing + input sanitization;
//! - predict;
//! - split;
//! - FreeSASA;
//! - ΔSASA/contacts;
//! - visualize;

use std::path::PathBuf;
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, prelude::*};

use clap::{Parser, ValueEnum};
use thiserror::Error;

/// Interpretation of the README's "side-chain accessibility changes by ≥10%".
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

    /// Output directory.
    #[arg(long, value_name = "DIR", default_value = "out")]
    out_dir: PathBuf,

    /// Allow `X` (unknown) residues. Off by default: `X` → a structural gap in the model.
    #[arg(long)]
    allow_unknown: bool,

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

#[derive(Debug, Error)]
#[allow(dead_code)] // TODO
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
    Parse { what: String, reason: String },
}

/// The 20 standard amino acids (one-letter codes).
const STANDARD_AA: &str = "ACDEFGHIKLMNPQRSTVWY";

/// Normalize then validate a raw sequence
///
/// *Normalize*: drop FASTA header lines, strip all whitespace, uppercase, drop a trailing
/// `*` stop marker.
///
/// *Validate*: non-empty, alphabet \in 20 AAs (+ `X` only if `allow_unknown`),
/// reporting the offending character and 1-based position if
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
                reason: format!(
                    "'X' at position {pos} (unknown residue → structural gap; \
                     pass --allow-unknown to permit)"
                ),
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

    // if args.verbose {
    //     eprintln!("VH: {} residues", heavy.len());
    //     eprintln!("VL: {} residues", light.len());
    //     eprintln!(
    //         "metric={:?} threshold={} out_dir={}",
    //         args.metric,
    //         args.threshold,
    //         args.out_dir.display()
    //     );
    // }

    // Pipeline (to implement; see mvp.ipynb for the reference flow):
    //   1. predict structure   → workers/predict.py → out/complex.pdb (chains H/L)
    //   2. split chains        → heavy.pdb / light.pdb
    //   3. FreeSASA ×3         → per-residue side-chain SASA (complex, heavy, light)
    //   4. ΔSASA + contacts    → metric first/second @ threshold
    //   5. write contacts.csv / contacts.json
    //   6. visualize           → workers/visualize.py (unless --no-viz)

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_tracing(&args.log);
    run(args)?;
    Ok(())
}

fn init_tracing(filter: &str) {
    let filter = EnvFilter::try_new(filter).unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .init();
}

use std::{
    collections::HashMap,
    fmt::Write,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use clap::{Parser, Subcommand};
use thiserror::Error;
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, prelude::*};

// ONNX inference path (preprocess -> ort -> pdb), replacing the Python worker.
mod e2e;
mod inference;
mod pdb;
mod preprocess;

const STANDARD_AA: &str = "ACDEFGHIKLMNPQRSTVWY";

// For host-specific use a local `.env` file (see `.env.example`)
const DEFAULT_PYTHON: &str = "python3";
// ABodyBuilder3 exported to ONNX (see `workers/export_onnx.py`) + the
// onnxruntime shared lib we dlopen at runtime (ort `load-dynamic`).
const DEFAULT_ONNX: &str = "vendored/abb3.onnx";
const DEFAULT_ORT_DYLIB: &str = "vendored/onnxruntime/lib/libonnxruntime.so";
const DEFAULT_OUT_DIR: &str = "out";
const DEFAULT_OUT_FILE: &str = "complex.pdb";
const DEFAULT_FREESASA: &str = "vendored/freesasa";
const DEFAULT_FORMAT: &str = "rsa";
const DEFAULT_OUT_CSV: &str = "contacts.csv";
const DEFAULT_VISUALIZE: &str = "workers/visualize.py";
const DEFAULT_PLOT_FILE: &str = "dsasa_barplot.png";
const CONTACT_METRIC: &str = "contact_metric_abs";

#[derive(Parser, Debug)]
#[command(name = "abfv")]
struct Args {
    /// Heavy-chain (VH) amino-acid sequence.
    #[arg(
        long,
        env = "ABFV_HEAVY",
        value_name = "SEQ",
        required_unless_present = "heavy_file"
    )]
    heavy: Option<String>,

    /// File with the VH sequence (plain text or FASTA).
    #[arg(long, value_name = "PATH", conflicts_with = "heavy")]
    heavy_file: Option<PathBuf>,

    /// Light-chain (VL) amino-acid sequence.
    #[arg(
        long,
        env = "ABFV_LIGHT",
        value_name = "SEQ",
        required_unless_present = "light_file"
    )]
    light: Option<String>,

    /// File with the VL sequence (plain text or FASTA).
    #[arg(long, value_name = "PATH", conflicts_with = "light")]
    light_file: Option<PathBuf>,

    /// Contact threshold (lives in [0.0 - 1.0])
    #[arg(long, default_value_t = 0.1, value_parser = parse_unit_interval, allow_hyphen_values = true)]
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

    /// File with the VL sequence (plain text or FASTA).
    #[arg(long, value_name = "OUT_CSV", default_value = DEFAULT_OUT_CSV)]
    contacts_out_csv_file: PathBuf,

    /// Tracing filter that honors standard RUST_LOG, with sensible defaults
    #[arg(long, env = "RUST_LOG", default_value = "info,api=debug,common=debug")]
    log: String,
}

#[derive(Subcommand, Debug)]
enum PredictCmd {
    /// Fold the Fv with the ABodyBuilder3 worker script.
    Predict(PredictArgs),
}

#[derive(clap::Args, Debug)]
struct PredictArgs {
    /// ABodyBuilder3 model exported to ONNX.
    #[arg(long, env = "ABFV_ONNX", value_name = "PATH", default_value = DEFAULT_ONNX)]
    onnx: PathBuf,

    /// libonnxruntime shared library (dlopen'd at runtime via ort load-dynamic).
    #[arg(long, env = "ABFV_ORT_DYLIB", value_name = "PATH", default_value = DEFAULT_ORT_DYLIB)]
    ort_dylib: PathBuf,

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
            onnx: env_or("ABFV_ONNX", DEFAULT_ONNX),
            ort_dylib: env_or("ABFV_ORT_DYLIB", DEFAULT_ORT_DYLIB),
            out_file: DEFAULT_OUT_FILE.into(),
            freesasa: None,
        }
    }
}

#[derive(clap::Args, Debug)]
struct FreesasaArgs {
    /// FreeSASA binary.
    #[arg(long, env = "ABFV_FREESASA", value_name = "PATH", default_value = DEFAULT_FREESASA)]
    binary: PathBuf,

    /// Output format, passed as `--format=<FORMAT>`.
    #[arg(long, default_value = DEFAULT_FORMAT)]
    format: String,

    /// `visualize` step + its options (omit to use defaults).
    #[command(subcommand)]
    visualize: Option<VisualizeCmd>,
}

impl Default for FreesasaArgs {
    fn default() -> Self {
        Self {
            binary: env_or("ABFV_FREESASA", DEFAULT_FREESASA),
            format: DEFAULT_FORMAT.into(),
            visualize: None,
        }
    }
}

#[derive(Subcommand, Debug)]
enum VisualizeCmd {
    /// Render the per-residue dSASA bar charts with the matplotlib worker.
    Visualize(VisualizeArgs),
}

/// Options for the matplotlib plot shell-out (`workers/visualize.py`).
#[derive(clap::Args, Debug)]
struct VisualizeArgs {
    /// Python interpreter (defaults to the ABodyBuilder3 venv).
    #[arg(long, env = "ABFV_PYTHON", value_name = "PATH", default_value = DEFAULT_PYTHON)]
    python: PathBuf,

    /// Worker script that renders the plot.
    #[arg(long, env = "ABFV_VISUALIZE", value_name = "PATH", default_value = DEFAULT_VISUALIZE)]
    script: PathBuf,

    /// Output PNG file name (written under the top-level `--out-dir`).
    #[arg(long, value_name = "FILE", default_value = DEFAULT_PLOT_FILE)]
    out_file: String,

    /// Contact metric (column name in the input csv file)
    #[arg(long, value_name = "CONTACT_METRIC", default_value = CONTACT_METRIC)]
    contact_metric: String,
}

impl Default for VisualizeArgs {
    fn default() -> Self {
        Self {
            python: env_or("ABFV_PYTHON", DEFAULT_PYTHON),
            script: env_or("ABFV_VISUALIZE", DEFAULT_VISUALIZE),
            out_file: DEFAULT_PLOT_FILE.into(),
            contact_metric: CONTACT_METRIC.into(),
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

    #[error("ONNX inference failed: {0}")]
    Inference(#[from] inference::InferError),

    #[error("PDB write failed: {0}")]
    Pdb(#[from] pdb::PdbError),
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
        if c == 'X' {
            if allow_unknown {
                continue;
            }

            return Err(AbfvError::InvalidSequence {
                chain,
                reason: format!("'X' at position {i} (pass --allow-unknown to permit)"),
            });
        }

        if !STANDARD_AA.contains(c) {
            return Err(AbfvError::InvalidSequence {
                chain,
                reason: format!("invalid residue '{c}' at position {i}"),
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
    info!("Starting\n{args:#?}");

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

    let mut freesasa = match predict.freesasa.take() {
        Some(FreesasaCmd::Freesasa(f)) => f,
        None => FreesasaArgs::default(),
    };

    let visualize = match freesasa.visualize.take() {
        Some(VisualizeCmd::Visualize(v)) => v,
        None => VisualizeArgs::default(),
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
    let contacts = delta_sasa(
        out_dir.join("complex.rsa"),
        out_dir.join("heavy.rsa"),
        out_dir.join("light.rsa"),
        args.threshold,
    )?;

    // 5. write contacts.csv
    let csv = &out_dir.join(args.contacts_out_csv_file);
    write_csv(csv, &contacts)?;
    info!(path = %csv.display(), "Wrote CSV file");

    // 6. visualize -> workers/visualize.py  (TODO --no-viz)
    let plot = run_visualize(&visualize, &out_dir, csv, args.threshold)?;
    info!(path = %plot.display(), "Wrote plot");

    Ok(())
}

fn main() -> Result<(), AbfvError> {
    let _ = dotenvy::dotenv();
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
) -> Result<Vec<ContactRow>, AbfvError> {
    let complex = parse_rsa(&complex_rsa)?;

    let mut isolated = parse_rsa(&heavy_rsa)?;
    isolated.extend(parse_rsa(&light_rsa)?);

    assert_eq!(
        complex.len(),
        isolated.len(),
        "isolated and complex should be equal!"
    );

    let mut contacts: Vec<ContactRow> = vec![];

    for (key @ (chain, residue_name, residue_number), complex_residue) in &complex {
        let iso_residue = isolated.get(key).ok_or(AbfvError::Data {
            why: format!("isolated map does not contain residue: {:?}", key),
        })?;

        let delta = iso_residue.side_chain_absolute - complex_residue.side_chain_absolute;

        let contact_metric = if delta > 0.0 {
            delta / iso_residue.side_chain_absolute
        } else {
            f64::NAN
        };

        contacts.push(ContactRow {
            chain: *chain,
            residue_name: residue_name.to_string(),
            residue_number: *residue_number,
            iso_side_chain_absolute: iso_residue.side_chain_absolute,
            complex_side_chain_absolute: complex_residue.side_chain_absolute,
            contact_metric,
            is_contact: contact_metric >= contact_threshold,
        });
    }

    Ok(contacts)
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
            let cols: Vec<&str> = l.split_whitespace().collect();

            if cols.len() < 8 {
                return Err(err(
                    i,
                    format!("expected at least 8 columns, found {}: {l:?}", cols.len()),
                ));
            }

            let residue_name = cols[1].to_string();

            let chain = cols[2]
                .chars()
                .next()
                .ok_or_else(|| err(i, format!("empty chain id (column 3) in {l:?}")))?;

            let residue_number = cols[3]
                .parse::<u32>()
                .map_err(|e| err(i, format!("residue number '{}': {e}", cols[3])))?;

            let side_chain_absolute = cols[6]
                .parse::<f64>()
                .map_err(|e| err(i, format!("side-chain ABS '{}': {e}", cols[6])))?;

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
    info!(onnx = %predict.onnx.display(), "Predicting Fv structure with ABodyBuilder3 (ONNX)");

    // ort `load-dynamic` resolves libonnxruntime via ORT_DYLIB_PATH on first
    // use. Honor an externally-set value; otherwise point at the configured lib.
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        std::env::set_var("ORT_DYLIB_PATH", &predict.ort_dylib);
    }

    let input = preprocess::string_to_input(heavy, light);
    info!(n = input.n, n_heavy = input.n_heavy, "Built model input");

    let mut predictor = inference::Predictor::load(&predict.onnx)?;
    let output = predictor.predict(&input)?;

    let complex_pdb = out_dir.join(&predict.out_file);
    pdb::write_complex(&output, &input, &complex_pdb)?;

    Ok(complex_pdb)
}

fn run_visualize(
    args: &VisualizeArgs,
    out_dir: &Path,
    contacts_csv: &Path,
    threshold: f64,
) -> Result<PathBuf, AbfvError> {
    info!("Rendering contact plots with matplotlib");

    let status = Command::new(&args.python)
        .arg(&args.script)
        .arg(contacts_csv)
        .arg("--metric")
        .arg(&args.contact_metric)
        .arg("--threshold")
        .arg(threshold.to_string())
        .arg("--out-dir")
        .arg(out_dir)
        .arg("--out-file")
        .arg(&args.out_file)
        .status()?;

    if !status.success() {
        return Err(AbfvError::Subprocess {
            tool: args.script.display().to_string(),
            code: status.code().unwrap_or(-1),
            stderr: "see output above".into(),
        });
    }

    Ok(out_dir.join(&args.out_file))
}

fn write_csv(path: &Path, rows: &[ContactRow]) -> Result<(), AbfvError> {
    let mut out = format!(
        "chain,residue_number,residue_name,iso_side_chain_absolute,complex_side_chain_absolute,{CONTACT_METRIC},is_contact\n"
    );

    for r in rows {
        _ = writeln!(
            out,
            "{},{},{},{:.3},{:.3},{:.3},{}",
            r.chain,
            r.residue_number,
            r.residue_name,
            r.iso_side_chain_absolute,
            r.complex_side_chain_absolute,
            r.contact_metric,
            r.is_contact,
        );
    }

    fs::write(path, out)?;

    Ok(())
}

fn init_tracing(filter: &str) {
    let filter = EnvFilter::try_new(filter).unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .init();
}

/// Parse a CLI float and require it to be a fraction in the closed unit interval.
fn parse_unit_interval(s: &str) -> Result<f64, String> {
    let v: f64 = s
        .parse()
        .map_err(|_| format!("`{s}` is not a valid number"))?;

    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("must be in [0.0, 1.0], got {v}"))
    }
}

/// Read an env var, falling back to a compile-time default.
fn env_or(key: &str, default: &str) -> PathBuf {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .into()
}

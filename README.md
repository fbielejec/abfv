# ABFV

## Instructions

1. Input: two protein sequences for examples  (EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPHYYGSSHWYFDVWGQGTLVTVSS, an antibody heavy chain - WGQGTLVTVSS) and (DIQMTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK, a light chain - FGQGTKVEIK), i.e. an antibody Fv.
2. Generate a 3D structure from the sequences, e.g. with ABodyBuilder3 (https://github.com/Exscientia/abodybuilder3).
3. Use FreeSASA (https://github.com/mittinatten/freesasa) to determine which amino acids of the chains are in contact.
4. Contact definition: residues whose side-chain surface accessibility (SASA) changes by at least 10% in the complex vs. computed for the isolated chains.
5. Visualize the results.
6. Wrap the whole thing in Docker and push to GitHub.

## Docker

The whole pipeline (the `abfv` binary orchestrator, ABodyBuilder3, FreeSASA, and the
matplotlib visualizer) is packaged into a single self-contained image (torch is CPU-only!)

### Build

> First build needs network!

```bash
make docker-build
```

Resulting image is ~2.1 GB.

### Run

```bash
make docker-run        # runs the bundled example Fv, writes to ./out
```

Outputs are written to the mounted `out/` directory: `contacts.csv` (per-residue dSASA contact metric) and `dsasa_barplot.png`.

#### Passing inputs

Passing other sequences:

```bash
docker run --rm -v "$PWD/out:/work/out" abfv \
  --heavy EVQ...VSS \
  --light DIQ...EIK
```

Each chain can be provided using one of the three ways:

| Method | Heavy | Light |
| --- | --- | --- |
| Inline sequence | `--heavy <SEQ>` | `--light <SEQ>` |
| FASTA / text file | `--heavy-file <PATH>` | `--light-file <PATH>` |
| Environment variable | `-e ABFV_HEAVY=<SEQ>` | `-e ABFV_LIGHT=<SEQ>` |

# Python env holding ABodyBuilder3 + Jupyter (override: make notebook VENV=/path/to/venv)
VENV ?= /home/filip/CloudStation/Python/abodybuilder3/.venv

# ABodyBuilder3 checkpoint (override: make predict CKPT=/path/to/model.ckpt)
CKPT ?= vendored/best_second_stage.ckpt

# FreeSASA binary (override: make freesasa FREESASA=/path/to/freesasa)
FREESASA ?= vendored/freesasa

# Hardcoded example Fv chains (verbatim from examples/light.fasta and examples/heavy.fasta)
LIGHT := DIQMTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK
HEAVY := EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPHYYGSSHWYFDVWGQGTLVTVSS

# Hardcoded: csv file wioth contacts metric calculated per residue
CONTACTS := out/contacts.csv

.PHONY: help
help: # Show help for each of the Makefile recipes
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: clippy
clippy: # Lint Rust sources
	cargo clippy --all-targets -- --no-deps -D warnings

.PHONY: fmt
fmt: # Format Rust sources
	cargo +nightly fmt --all

.PHONY: fmt-check
fmt-check: # Check formatting
	cargo +nightly fmt --all -- --check

.PHONY: test
test: # Run tests with verbose output
	cargo test --verbose -- --nocapture

.PHONY: notebook
notebook: # Launch JupyterLab from the venv and open mvp.ipynb in the browser
	$(VENV)/bin/jupyter lab mvp.ipynb

.PHONY: watch
watch: # Watch for changes and run clippy
	cargo watch -s 'cargo clippy' -c

.PHONY: release
release: # Build release binary
	cargo build --release

.PHONY: ort
ort: # Fetch the pinned onnxruntime shared lib into vendored/onnxruntime (not in git)
	./vendored/fetch-onnxruntime.sh

.PHONY: export-onnx
export-onnx: # Re-export ABodyBuilder3 to vendored/abb3.onnx (verifies vs PyTorch)
	$(VENV)/bin/python workers/export_onnx.py --checkpoint "$(CKPT)"

.PHONY: predict
predict: # [reference] Predict via the Python ABodyBuilder3 worker (host venv)
	$(VENV)/bin/python workers/predict.py "$(LIGHT)" "$(HEAVY)" --checkpoint "$(CKPT)"

.PHONY: run
run: ort # Run the abfv Rust CLI on the hardcoded example chains (ONNX inference)
	cargo run -- --heavy "$(HEAVY)" --light "$(LIGHT)" predict freesasa --binary "$(FREESASA)"

.PHONY: freesasa
freesasa: # Run FreeSASA (rsa) on the three pipeline PDBs
	$(FREESASA) --format=rsa out/complex.pdb --output=out/complex.rsa
	$(FREESASA) --format=rsa out/heavy.pdb --output=out/heavy.rsa
	$(FREESASA) --format=rsa out/light.pdb --output=out/light.rsa

.PHONY: visualize
visualize: # Run visualize script
	$(VENV)/bin/python workers/visualize.py "$(CONTACTS)"

.PHONY: docker-build
docker-build: # Build the self-contained abfv docker image
	DOCKER_BUILDKIT=1 docker build -t abfv .

.PHONY: docker-run
docker-run: # Run the dockerized pipeline (uses bundled example chains)
	mkdir -p out
	docker run --rm -v "$(PWD)/out:/work/out" abfv \
	  --heavy-file /opt/abfv/examples/heavy.fasta \
	  --light-file /opt/abfv/examples/light.fasta

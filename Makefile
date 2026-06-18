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

.PHONY: watch
watch: # Watch for changes and run clippy
	cargo watch -s 'cargo clippy' -c

.PHONY: release
release: # Build release binary
	cargo build --release

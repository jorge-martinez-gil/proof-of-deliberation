# Proof-of-Deliberation -- reproducibility entry points.
# `make paper` regenerates every number/table/macro from the experiment
# outputs and recompiles the paper.
#
# Per-run logs are NOT committed to keep the repository lean: populate
# `out_pod_unified/` first with `make experiments` (or `pod-experiments`),
# then run `make paper`.
#
# Exact toolchain: Python 3.10.12, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3,
# scikit-learn 1.7.2, matplotlib 3.10.9 (see requirements-lock.txt).

PYTHON ?= python3
IN     ?= out_pod_unified
TEX    ?= trimmed.tex

.PHONY: paper stats check pdf experiments realdata verify clean

## paper: regenerate derived stats + macros, verify, and build the PDF (~1 min)
paper: stats check pdf

## stats: rebuild statistical + real-user macros from the per-run CSVs in $(IN)
stats:
	$(PYTHON) scripts/build_stats_report.py --in $(IN)
	$(PYTHON) scripts/realuser_stats.py

## check: fail if any reported claim/macro drifts from the regenerated data
check:
	$(PYTHON) scripts/check_claims.py
	$(PYTHON) scripts/sync_macros.py --in $(IN)

## pdf: compile the manuscript (requires a LaTeX toolchain with IEEEtran)
pdf:
	latexmk -pdf -interaction=nonstopmode $(TEX)

## experiments: re-run the RAW streaming experiments (needs network/OpenML; hours)
experiments:
	pod-experiments --datasets synth,elec2,gas,covertype,airlines --out $(IN)

## realdata: re-run the real-user / ATC analyses from raw session logs
realdata:
	pod-realdata --in data_real --out out_real

## verify: unit tests
verify:
	pytest -q

## clean: remove LaTeX build artifacts
clean:
	latexmk -C $(TEX) || true

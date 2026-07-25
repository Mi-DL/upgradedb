# Formal ULTRA zero-shot reference

This protocol evaluates one fixed, externally pretrained `ultra_4g` checkpoint
on all six UpgradeBench chains and all three tasks. It is a separate evidence
regime from the five-seed, benchmark-trained PyKEEN and NBFNet references: no
UpgradeBench label is used for checkpoint selection, training, fine-tuning, or
calibration, but inference does use each target chain's early graph. The result
is therefore an **external pretrained zero-shot reference**, not graph-free
cold start and not a fair-compute model-family contest.

## Frozen inputs and environment

The committed controller and config bind the checkpoint SHA-256, the disclosed
four-graph pretraining mixture, upstream commits, six candidate cohorts, raw
BACI source identity, native backend, task aggregation, value budgets, and
reporting rule. The checkpoint training seed is not disclosed; the fixed value
`1024` controls inference only. The formal path requires CUDA, native
`torch_scatter`, and ULTRA's native `rspmm` extension. Compatibility shims and
the PyG message-passing fallback used by the feasibility smoke are forbidden.

Before execution, provide the permission-gated BACI archive through the normal
`VCU_RAW` data-root mechanism and place the hash-matching checkpoint and
vendored upstream source at the paths declared by
`configs/v2_ultra_formal.json`. Do not publish the raw archive, checkpoint,
vendored tree, JIT cache, host logs, component receipts, or score files.

## Irreversible phase order

Run the phases in order from the repository root:

```bash
python tools/v2_ultra_formal.py freeze

python tools/v2_ultra_formal.py score-chain --chain sheep --device cuda:0
python tools/v2_ultra_formal.py score-chain --chain cotton --device cuda:0
python tools/v2_ultra_formal.py score-chain --chain aluminium --device cuda:0
python tools/v2_ultra_formal.py score-chain --chain nickel --device cuda:0
python tools/v2_ultra_formal.py score-chain --chain cocoa --device cuda:0
python tools/v2_ultra_formal.py score-chain --chain oilseed-soy --device cuda:0

python tools/v2_ultra_formal.py seal-scores
python tools/v2_ultra_formal.py evaluate
python tools/v2_ultra_formal.py verify --level evaluation
```

The six `score-chain` commands may run on separate compatible GPUs after the
same frozen manifest and runtime receipt have been established. Every chain
must use the same exact software/native-extension environment. Scoring reads
only candidate identities and protocol metadata. `evaluate` is the first phase
allowed to parse target labels, and it refuses to run until all six score
components have been verified and sealed.

Sheep is scored twice in the same process, device, and model instance. Both
complete A+B score files must pass the configured score tolerance, and all six
derived metrics must agree within the prespecified absolute tolerance. This is
a same-environment repeatability sentinel, not a claim of cross-hardware
bitwise determinism or training-seed uncertainty.

## Public result promotion

The formal output tree remains private. After `verify --level evaluation`
passes, generate the only public result surface:

```bash
python tools/summarize_v2_ultra_results.py --check-only
python tools/summarize_v2_ultra_results.py
python tools/summarize_v2_ultra_results.py --verify-output
```

The sanitizer emits exactly 18 chain-task records plus recomputed six-chain
means, trained-reference comparisons, the abstract eligibility gate, formal
receipt hashes, and the sheep repeatability result. Its public-only verifier
does not open scores, BACI, checkpoints, or the private formal tree. The public
summary forbids fair-compute, champion, and statistical-significance claims;
all comparisons are descriptive over the six selected chains.

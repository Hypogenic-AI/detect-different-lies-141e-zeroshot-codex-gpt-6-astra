# Detecting different errors

A completed, locally executed research project on whether lie-detection and error-detection probes distinguish incentive-conditioned misreporting from epistemic errors.

**Paper:** [compiled PDF](paper_draft/main.pdf) · [LaTeX entry point](paper_draft/main.tex)

## What was run

- **Model:** Qwen2.5-7B-Instruct, unquantized bfloat16, on one NVIDIA RTX A6000.
- **Questions:** 1,600 unique MMLU test questions across 57 subjects; 800/300/500 question-disjoint train/validation/test split.
- **Inference:** 14,400 one-token answer selections: four neutral elicitation variants plus independent neutral, score-pressure, social-pressure, explicit-honest, and explicit-deceptive conditions.
- **Knowledge proxies:** four correct elicitations = stable-correct (K); four identical wrong answers = stable-wrong (W); other patterns = inconsistent (U).
- **Probes:** standardized regularized logistic regression on four layers, at both the last prompt token and answer token. Primary layer 14 was fixed before inference. Instruction, neutral-error, pressure, cross-prompt, same-pressure, and an exploratory neutral-knowledge probe were tested.
- **Controls:** same-pressure correct answers and errors, preferred-answer conflicts, question-only text, answer letters, surface features, uncertainty, prompt condition, pre-answer states, and post-hoc preference-adherence checks.

MMLU replaced open-ended TriviaQA to permit exact scoring and answer-order controls. This means the project studies constrained-choice errors, including reasoning failures, rather than free-text hallucination alone. The pressure prompts state rewards/preferences but do not instruct falsehood. No real reward is delivered and no intentional lying is verified.

## Main findings

The neutral checks classified 887 questions as K, 534 as U, and 179 as W.

| Reporting condition | Errors | K errors | U errors | W errors |
|---|---:|---:|---:|---:|
| Neutral | 514 | 11 | 327 | 176 |
| Score pressure | 650 | 133 (20.5%) | 352 (54.2%) | 165 (25.4%) |
| Social pressure | 685 | 162 (23.6%) | 364 (53.1%) | 159 (23.2%) |

Percentages are shares of that condition's errors. Requiring the independent neutral evaluation also to be correct leaves 125 and 154 K correct-to-wrong transitions under score and social pressure. The model also resisted pressure: on 671 K questions with a wrong preferred answer, it remained correct on 540 score-pressure and 509 social-pressure responses.

On held-out questions at layer 14:

- The instructed-deception probe separates K social-pressure errors from U neutral errors perfectly (AUROC **1.000**), but **prompt-condition identity alone does too**.
- When both false classes receive social pressure, its AUROC is **0.621** (95% question-bootstrap interval 0.53–0.71).
- A probe trained directly on same-pressure errors reaches **0.663** (0.57–0.75); its pre-answer counterpart reaches **0.692** (0.60–0.78).
- The neutral-error probe preferentially scores U errors: AUROC **0.343** (0.25–0.44) with K errors positive.
- In a post-hoc restriction where both classes select the preferred answer, instructed-deception AUROC is **0.537** and direct same-pressure AUROC is **0.590**. A neutral-knowledge probe retains signal (**0.681**).

These results demonstrate evaluation shortcuts and limited specificity, not the impossibility of useful monitoring. Later layers sometimes perform better. Knowledge strata are observational; U pressure errors are not verified honest errors. The percentages above are **not estimates of the natural prevalence of intentional lies**.

## Reproduce

Run from the repository root. An NVIDIA GPU and a system LaTeX installation (`pdflatex`, `bibtex`, standard packages) are needed for the complete pipeline. The tested GPU had 48 GB VRAM. The model download is approximately 14 GB; raw experimental records/activations occupy approximately 360 MB. No API calls or API keys are needed for inference; the model and dataset are public.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Downloads pinned snapshots and reconstructs the exact question sample.
.venv/bin/python src/prepare.py

# Resumable inference: keeps completed batches. Keep --batch-size 12 unchanged.
.venv/bin/python src/run_model.py --batch-size 12

# CPU analysis, validation, conditional checks, and document artifacts.
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python src/analyze.py
OPENBLAS_NUM_THREADS=2 .venv/bin/python src/validate.py
OPENBLAS_NUM_THREADS=2 .venv/bin/python src/robustness.py
.venv/bin/python src/examples.py
.venv/bin/python src/report.py
bash src/build_paper.sh
.venv/bin/python src/audit_artifacts.py
```

To reproduce analysis and the PDF from the supplied raw results, skip `prepare.py` and `run_model.py`; no GPU/model download is then necessary. `src/literature.py` independently refreshes primary-source bibliographic metadata. The paper's prose is retained as source; numerical tables/macros and figures are rebuilt by the scripts.

The inference runner resumes only the identical sample, batching, and configuration. For a genuinely new experiment, use a separate copy of the workspace or move `results/raw/` first. The engineering pilot in `results/pilot/` is excluded from analysis. Initial PyTorch 2.14.1 failed because a compiler was unavailable; all reported inference uses PyTorch 2.6.0. Direct dependencies are pinned in `requirements.txt`; the complete observed environment, including unused dependencies left from that failed setup, is recorded in `results/environment_freeze.txt`.

## Files and audit trail

- `results/design.md`: design decisions and explicitly dated-by-stage exploratory additions.
- `results/provenance.json`, `questions.json`, `deduplication.json`: immutable revisions, exact sample, permutations, preferences, splits, and duplicate-removal audit.
- `results/raw/batch_*.json`: actual prompts, predictions, original-answer identities, logits, probabilities, and scoring. Matching `.npz` files hold prompt/answer states at layers 7/14/21/28. Belief-only batches have no activation file.
- `results/responses.csv`, `behavior.json`, `paired_effects.json`: behavioral labels, compositions, and paired effects.
- `results/probe_metrics.csv`, `test_scores.csv`, `probe_training.csv`, `probes/`: all probe estimates, test scores, training counts, and fitted coefficients/scalers.
- `results/robustness.csv`, `position_differences.csv`, `flag_rates.csv`: conditional checks, paired position comparisons, and threshold-transfer results.
- `results/validation.json`, `results/raw/cache_verification.json`: scoring/statistical and cached-inference checks.
- `results/artifact_manifest.json`, `artifact_audit.json`: file hashes and final document/provenance checks.
- `logs/`: successful runs plus failed setup logs; `paper_draft/figures/`: figures generated from the data.

Question-level bootstrap intervals condition on the fitted probes; they do not include retraining uncertainty. The full paper discusses proxy labels, benchmark ambiguity, contamination, prompt-induced compliance, and the distinction between causal mechanisms and behavioral categories. No git commit, push, or remote modification was performed.

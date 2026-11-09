# Stress-testing weight-space programming

In [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/), Takehi et al. report that GPT-6 Astra can read and edit neural network weights directly. Their headline edit took Qwen3.5-0.8B from 7.9% to 48.1% on MATH-500.

I wanted to know what that means for mechanistic interpretability, so I rebuilt the edit from their [published logs](https://github.com/RikiyaT/lm_understand-logs), tested it against simple alternatives, and ran related experiments on circuits, backdoors and sycophancy. Everything is in one script, and the full run took 87 minutes on four rented RTX 5090s.

## Top findings

1. **The published edit's scores replicate.** On my own run it scored 54.0% on MATH-500 and 68.0% on GSM8K, against the paper's 48.1% and 67.1%. Answers cut off at the token limit fell from 91.4% to 16.4%.

2. **Turning thinking off gets most of the same gain, with no edit.** The post reports Qwen with thinking on only. With thinking off, the unedited model reached 48.8% combined, against 61.0% for the edit and 9.2% for the unedited model with thinking on. That is about 76% of the edit's gain, at about one-seventh of the tokens. The edit still wins by 12.2 points (roughly +6 to +18), so it is a real improvement, but much smaller than 7.9% → 48.1% suggests.

3. **A simple method finds a similar direction, but removing it from other layers did nothing.** A difference-of-means direction, found with no agent, has |cosine| 0.62 and 0.58 with Astra's edit in layers 15 and 19, where chance is about 0.03. Yet removing it from the layers my method chose (7 and 8) did nothing: 9.2%, the same as unedited. Removing it from all 24 layers made things worse on practice problems (0% correct, 100% cut off). I did not test this direction at layers 15 and 19, and that is the obvious next experiment.

4. **Weights can reveal sycophancy, but only if you know what to look for.** Comparing a weight change with a known template separated sycophantic models well once 25% of labels were sycophantic (AUC 0.89). Just measuring how far the weights moved stayed near chance until 100% (0.45–0.55, then 0.75). At 10% of labels, behaviour had already shifted from 0.74 to 0.91 while the template's AUC was only 0.64, which was not significant.

5. **Perfect accuracy can hide a changing circuit.** A hand-built network stayed 100% accurate with no noise at all, while swapping a single known unit stopped giving the predicted answer up to 25% of the time. After a task change, the now-unneeded AND unit lost its correlation with AND (1.00 → 0.00) within 500 steps, while cosine similarity to the starting weights was still 0.996.

6. **At 481 edited weights, backdoor detection looks strong only when the defender knows the attack.** With 481 edited weights, a classifier trained on the same kind of attack reached AUC 0.99, while plain weight statistics reached 0.68. That 0.99 is probably optimistic for real use.

**The pattern across these:** measuring how much weights changed told me little unless the change was large. Knowing the right direction, and the right place, told me a lot. If that holds up, the hard part of weight-space work is knowing where and what to look for, not the size of the change.

## Results

### Idea 1: a simple baseline against Astra's edit

250 MATH-500 and 250 GSM8K problems, one sample each, using the paper's sampling settings.

| Arm | MATH-500 | GSM8K | Combined | Cut at token limit | Mean tokens |
|---|---|---|---|---|---|
| Unedited, thinking on | 8.4% (paper 7.9) | 10.0% (paper 13.8) | 9.2% | 91.4% | 30,438 |
| **Astra's 50-pass edit** | **54.0%** (paper 48.1) | **68.0%** (paper 67.1) | **61.0%** | 16.4% | 16,531 |
| **Unedited, thinking off** | **45.2%** | **52.4%** | **48.8%** | 18.6% | 2,448 |
| Difference-of-means, layers 7 and 8 | 5.2% | 13.2% | 9.2% | 91.8% | 30,458 |
| Random direction, same layers | 7.2% | 12.4% | 9.8% | 90.8% | 30,264 |

- My numbers are within about two standard errors of the paper's, on a subset of its problems and with fewer samples.
- Thinking off uses the model card's non-thinking settings and an 8,192-token cap, as in the authors' evaluation code, so it differs from the other arms in more than the flag.
- The range for edit minus thinking-off is my unpaired estimate. I did not compute a paired one.
- The authors' logs add context: 10 and 25 passes reached only 12.3% and 14.1% on MATH-500, and the 48.1% comes from one 50-pass run.

### Idea 4: how much sycophancy shows in the weights?

SmolLM2-135M, 8 seeds, 300 fine-tuning steps. AUC compares sycophant/control pairs with control/control pairs, and 0.5 is chance.

| Sycophantic labels | Sycophancy rate (control 0.74) | Template on weights | Template on activations | Distance only |
|---|---|---|---|---|
| 5% | 0.81 | 0.62 | 0.42 | 0.53 |
| 10% | 0.91 | 0.64 | 0.17 | 0.48 |
| 25% | 0.94 | **0.89** | 0.20 | 0.45 |
| 50% | 0.99 | **0.98** | 0.31 | 0.55 |
| 100% | 1.00 | **1.00** | **1.00** | 0.75 |

### Idea 5: a known circuit under further training

A hand-compiled network for `XOR(x0,x1) OR AND(x2,x3)`, 50 seeds, 3,000 steps. The swap score is the share of single-unit swaps that give the answer the circuit predicts.

| Condition | Original-task accuracy | XOR swap | AND swap | XOR correlation | AND correlation | Cosine to start |
|---|---|---|---|---|---|---|
| Same task | 1.000 | 0.867 | 0.755 | 0.970 | 0.915 | 0.984 |
| Label noise | 1.000 | 0.870 | 0.850 | 0.929 | 0.948 | 0.990 |
| Weight noise | 1.000 | 0.782 | 0.769 | 0.962 | 0.902 | 0.988 |
| Related task change | 0.625 | n/a | n/a | 0.828 | 0.004 | 0.988 |
| Random start | 1.000 | 0.000 | 0.006 | -0.027 | 0.138 | 0.449 |

The unrelated-task condition reached only 55% on its target, about chance, so I leave it out here. It is in the raw results.

### Idea 6: backdoors made by editing weights

Digit-classifier MLPs, 40 clean and 40 backdoored networks per budget, with an optimiser as the attacker.

| Edited weights | Working backdoors | Weight statistics AUC | Classifier AUC | Trigger inversion AUC |
|---|---|---|---|---|
| 4 | 0/40 | - | - | - |
| 48 | 0/40 | - | - | - |
| 481 | 40/40 | 0.68 | 0.99 | 0.69 |
| 4,810 | 40/40 | 1.00 | 1.00 | 0.36 |

## Limits

- **LLM experiments did not run.** Idea 3 (do explanations predict edits?), idea 2 (which tools help an audit?) and idea 6's LLM arm were skipped because the model setting was missing. That code has only run against stand-ins, so nothing here tests an LLM's ability to read weights.
- **Idea 1:** layers were chosen on only 64 practice problems, and the separation score barely varied across the top layers (2.65–2.81). The overlap uses the top singular vector of Astra's edit, and I did not measure how much of the edit that vector captures. I also measured only math, not the post's other benchmarks.
- **Idea 4:** this is a different, smaller model than the post's Gemma, and the control already prefers the suggestion 74% of the time. All strengths share the same null pairs, and p-values are not corrected for multiple comparisons. The below-chance activation scores are unexplained.
- **Idea 5:** swap scores settle on nearly identical values across seeds (0.87 appears in several conditions), which looks like a structural effect rather than gradual drift. Single-unit swaps may also understate a job that spread across units, and Adam may cause some of the movement. Swap scores are not meaningful after a task change, so the table omits them. I have not resolved any of this.
- **Idea 6:** the tiny-budget attacks failed on attack success, normal accuracy or both, so this does not show small backdoors are impossible. Trigger inversion scoring below chance suggests my implementation or settings need work.
- These are exploratory runs and have not been peer reviewed.

## Reproduce

You need a Linux machine with a GPU and about 40 GB of free disk. The script installs into its own `mi_venv` folder and uses every GPU it finds.

```bash
python mi_sprint.py 2>&1 | tee run.log
```

Results appear in [`results/SUMMARY.md`](results/SUMMARY.md). To include the LLM experiments, set a key first:

```bash
# Claude
export ANTHROPIC_API_KEY=your_key

# Or any OpenAI-compatible provider
export OPENAI_API_KEY=your_key OPENAI_BASE_URL=provider_url MI_LLM_MODEL=model_name MI_PRICE_IN=input_price MI_PRICE_OUT=output_price
```

LLM spending stops at $30 by default, based on those prices. To run selected ideas after the first install, use `mi_venv/bin/python mi_sprint.py run 5 6`.

## Repository

```
mi_sprint.py     all six experiments, the parallel runner and the provider support
results/         the raw output of the run reported above
  SUMMARY.md     the automatic report
  idea*.json     full numbers
  logs/          per-idea logs, with progress bars removed
```

I corrected some report wording in the script after this run, including idea 5's automatic "Read" line, which wrongly blamed label noise. [`results/`](results/) keeps the original text, and this README has the corrected interpretation.

## What I would do next

1. Remove the difference-of-means direction at layers 15 and 19, and compare it with Astra's edit.
2. Compare the edit and thinking-off on the same problems, with matched settings and the post's other benchmarks.
3. Run the LLM experiments, including Astra if I can get access.
4. Retest idea 5 with SGD and multi-unit swaps.

## Credits

The original work, edit and evaluation settings are by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab. The edit is downloaded at run time from [their logs](https://github.com/RikiyaT/lm_understand-logs), not stored here.

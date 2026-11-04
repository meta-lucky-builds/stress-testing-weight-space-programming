# Results

> **Note added after the run:** the report below is unchanged. No LLM was called, despite the `LLM:` line in the header. Ideas 3 and 2 were skipped because the model setting was missing, not the API key. Idea 1's verdict applies only to layers 7 and 8. Idea 5's "Read" line wrongly blames label noise, because the no-noise control changed just as much. See the README for the corrected interpretation.

Total time 87.3 min on 4 GPU(s). LLM: openai, claude-opus-5-5.

## Idea 5

50 seeds, 3000 training steps. Interchange accuracy (IIA) = how often swapping that one unit between inputs gives the answer the circuit predicts. 1.0 means the unit still does its job.

| Condition | Original-task acc | Target acc | XOR IIA, original unit | XOR IIA, best unit | AND IIA, original unit | AND IIA, best unit |
|---|---|---|---|---|---|---|
| same_task | 1.000 | 1.000 | 0.867 | 0.867 | 0.755 | 0.755 |
| label_noise | 1.000 | 1.000 | 0.870 | 0.870 | 0.850 | 0.850 |
| weight_noise | 1.000 | 1.000 | 0.782 | 0.782 | 0.769 | 0.769 |
| related_shift | 0.625 | 1.000 | 0.757 | 0.757 | 0.834 | 0.834 |
| unrelated_shift | 0.732 | 0.554 | 0.740 | 0.740 | 0.675 | 0.675 |
| random_init_control | 1.000 | 1.000 | 0.000 | 0.008 | 0.006 | 0.011 |

**Read:** label noise kept the behaviour but the original units lost their role: behaviour alone would hide this drift.

## Idea 6

40 clean and 40 backdoored networks per budget; clean test accuracy before attack 0.987. AUC 0.5 is chance. Backdoors that did not work are left out of detection.

| Edited weights | Attacks working | Attack success | Clean acc | Weight stats AUC | Meta-classifier AUC | Trigger inversion AUC |
|---|---|---|---|---|---|---|
| 4 | 0/40 | 0.45 | 0.827 | - | - | - |
| 48 | 0/40 | 0.81 | 0.772 | - | - | - |
| 481 | 40/40 | 1.00 | 0.982 | 0.68 (p=0.003) | 0.99 (p=0.000) | 0.69 (p=0.001) |
| 4810 | 40/40 | 1.00 | 0.986 | 1.00 (p=0.000) | 1.00 (p=0.000) | 0.36 (p=0.985) |

LLM arm skipped.

The attacker here is an optimiser, not an LLM: this tests the defences, not an LLM's ability to attack.

## Idea 3

Skipped: needs an LLM API key.

## Idea 2

Skipped: needs an LLM API key.

## Idea 4

Model HuggingFaceTB/SmolLM2-135M-Instruct. 8 seeds, 300 fine-tuning steps per run. The score of a pair is its similarity to a template from other seeds at full strength; AUC compares sycophant pairs with control pairs (0.5 is chance). With 8 seeds, the smallest possible one-sided p is about 1/12870.

| Sycophantic labels | Sycophancy rate | Control rate | Weights AUC | Activations AUC | Distance only AUC |
|---|---|---|---|---|---|
| 5% | 0.81 | 0.74 | 0.62 (p=0.231) | 0.42 (p=0.724) | 0.53 |
| 10% | 0.91 | 0.74 | 0.64 (p=0.190) | 0.17 (p=0.993) | 0.48 |
| 25% | 0.94 | 0.74 | 0.89 (p=0.003) | 0.20 (p=0.982) | 0.45 |
| 50% | 0.99 | 0.74 | 0.98 (p=0.000) | 0.31 (p=0.902) | 0.55 |
| 100% | 1.00 | 0.74 | 1.00 (p=0.000) | 1.00 (p=0.000) | 0.75 |

## Idea 1

Problems: 250 MATH-500 and 250 GSM8K, one sample each, paper's sampling. Baseline chosen on 64 practice problems: top2, layers [7, 8].

| Arm | MATH-500 | GSM8K | Combined | Cut at limit | Mean tokens |
|---|---|---|---|---|---|
| unedited | 8.4% (paper 7.9) | 10.0% (paper 13.8) | 9.2% | 91.4% | 30438 |
| astra_50_pass | 54.0% (paper 48.1) | 68.0% (paper 67.1) | 61.0% | 16.4% | 16531 |
| baseline_dom | 5.2% | 13.2% | 9.2% | 91.8% | 30458 |
| random_direction | 7.2% | 12.4% | 9.8% | 90.8% | 30264 |
| thinking_off | 45.2% | 52.4% | 48.8% | 18.6% | 2448 |

Baseline minus Astra: -51.8%, 95% CI [-56.2%, -47.4%]. Random direction minus unedited: +0.6%.
Overlap of the baseline direction with Astra's edit (|cosine|): {"15": {"same_layer": 0.617, "best_other_layer": 0.617, "random_expected": 0.031}, "19": {"same_layer": 0.576, "best_other_layer": 0.576, "random_expected": 0.031}}

**Verdict:** The simple baseline FAILS: this direction alone does not give the effect.

The thinking_off arm answers a different question: how much of the deficit comes from using thinking mode at all.

## GPU use

GPU 0: 84% busy, peak 31 GB, GPU 1: 78% busy, peak 30 GB, GPU 2: 64% busy, peak 30 GB, GPU 3: 63% busy, peak 30 GB

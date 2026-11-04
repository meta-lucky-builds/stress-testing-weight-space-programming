
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

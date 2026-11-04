
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

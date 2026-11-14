# Lightweight Phishing URL Detection

Experiment files for the MSc thesis *A Lightweight Deep Learning Model for Phishing URL Detection: Improving the Deployability of Optimized URL Intelligence Frameworks*, by Patrick Amegavi, Kwame Nkrumah University of Science and Technology (KNUST), 2026.

The study rebuilds the ADUIN phishing URL detector (Gobinath and Manikandan, *Scientific Reports* 16, 15794, 2026) on the PhiUSIIL dataset, then makes it smaller with pruning, quantization and knowledge distillation, and checks what happens to detection of phishing from domains the model has never seen.

## What is in this repository

| Folder | Contents |
|---|---|
| `data/` | The train, validation and test splits (70/15/15, 165,056 / 35,369 / 35,370 rows), the 24 URL-only feature names used by Baseline U, and the min-max scalers for each feature regime (F = 50 features, C = 49, U = 24). |
| `models/baselines/` | The reproduced model in all three regimes, three seeds each (42, 43, 44). |
| `models/pruning/` | One-shot and iterative magnitude-pruned versions of Baseline U at 50 to 95 per cent sparsity. |
| `models/quantization/` | Baseline U after dynamic 8-bit post-training quantization. |
| `models/distillation/` | The four student networks (S-192, S-64, S-16, S-LIN) and S-16 quantized to 8 bits. |
| `results/` | Per-seed test results for every experiment. |
| `code/` | Model definitions, a script that reloads the saved models and checks them against the recorded results, and the teacher control experiment. |

## Main results (mean of three seeds)

| Model | Parameters | Size | ms per URL (CPU) | Accuracy | Unseen-domain recall |
|---|---|---|---|---|---|
| Baseline U (reproduced ADUIN, 24 features) | 96,385 | 0.391 MB | 0.879 | 99.66% | 99.53% |
| Distilled S-16 (recommended) | 545 | 0.0054 MB | 0.049 | 99.66% | 99.53% |
| Distilled S-LIN | 25 | 0.0021 MB | 0.015 | 99.00% | 98.71% |

The control experiment in `results/teacher_control_results.csv` shows that S-16 and S-LIN reach practically the same results when trained on the labels alone, without the teacher.

A note on the benchmark: PhiUSIIL scores above 99 per cent with every feature set tested, so these results show that compression is safe on this dataset. They do not on their own show the same holds on harder, real-world phishing.

## Reproducing the results

```bash
pip install -r requirements.txt
python code/evaluate_saved_models.py   # reloads saved models, compares with results/
python code/teacher_control.py         # reruns the with/without-teacher control (about 6 minutes on a CPU)
```

The models expect inputs in the order listed in `data/features_U.txt`, scaled with `data/scaler_min_U.npy` and `data/scaler_rng_U.npy`. Label 1 is legitimate and label 0 is phishing; the models output the probability that a URL is legitimate.

The original training was done in Google Colab (Python 3.12, PyTorch 2.11, Tesla T4). The training notebook will be added to `notebook/`.

## Data

The PhiUSIIL dataset is by Prasad and Chandra (2024), *Computers & Security* 136, 103545, and is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset). The splits in `data/` are derived from it.

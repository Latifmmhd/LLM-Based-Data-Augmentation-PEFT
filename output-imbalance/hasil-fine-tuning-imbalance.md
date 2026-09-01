# Hasil Fine-Tuning Emotion Classification dengan PEFT (LoRA) - Imbalanced Dataset

## Overview

Laporan ini berisi hasil fine-tuning model klasifikasi emosi menggunakan teknik **Parameter-Efficient Fine-Tuning (PEFT)** dengan metode **LoRA (Low-Rank Adaptation)** pada dataset imbalanced.

- **Source Notebook**: `ft-with-peft.ipynb`
- **Dataset Split**: Train=4956, Val=1062, Test=1062
- **Emosi Classes**: Anger, Fear, Joy, Love, Neutral, Sad (6 kelas)
- **LoRA Configuration**: r=8, lora_alpha=32, lora_dropout=0.1
- **Max Sequence Length**: 128

---

## 1. IndoBERTweet-base-uncased

### Hyperparameter
| Parameter | Value |
|---|---|
| Epochs | 8 |
| Learning Rate | 1e-4 |
| Train Batch Size | 16 |
| Eval Batch Size | 8 |
| Weight Decay | 0.0921 |
| Trainable Parameters | 299,526 (0.27%) |
| Training Time | 8.70 menit |

### Test Results
| Metric | Value |
|---|---|
| Accuracy | 0.7665 |
| Precision (Macro) | 0.7668 |
| Recall (Macro) | 0.7796 |
| F1-Score (Macro) | 0.7719 |

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Anger | 0.7638 | 0.8444 | 0.8021 | 180 |
| Fear | 0.7852 | 0.8731 | 0.8269 | 134 |
| Joy | 0.8154 | 0.7644 | 0.7891 | 208 |
| Love | 0.7248 | 0.7822 | 0.7524 | 101 |
| Neutral | 0.7343 | 0.6838 | 0.7082 | 291 |
| Sad | 0.7770 | 0.7297 | 0.7526 | 148 |

### Confusion Matrix
![Confusion Matrix - IndoBERTweet](/image/cm_indobertweet.png)

---

## 2. XLM-RoBERTa-large

### Hyperparameter
| Parameter | Value |
|---|---|
| Epochs | 10 |
| Learning Rate | 3e-4 |
| Train Batch Size | 16 |
| Eval Batch Size | 8 |
| Weight Decay | 0.0533 |
| Trainable Parameters | 1,842,182 (0.33%) |
| Training Time | 19.91 menit |

### Test Results
| Metric | Value |
|---|---|
| Accuracy | 0.7985 |
| Precision (Macro) | 0.8000 |
| Recall (Macro) | 0.8046 |
| F1-Score (Macro) | 0.8012 |

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Anger | 0.8362 | 0.8222 | 0.8291 | 180 |
| Fear | 0.8244 | 0.8060 | 0.8151 | 134 |
| Joy | 0.8537 | 0.8413 | 0.8475 | 208 |
| Love | 0.7059 | 0.8317 | 0.7636 | 101 |
| Neutral | 0.7466 | 0.7491 | 0.7479 | 291 |
| Sad | 0.8333 | 0.7770 | 0.8042 | 148 |

### Confusion Matrix
![Confusion Matrix - XLM-RoBERTa-large](/image/cm_xlm_r_large.png)

---

## 3. XLM-RoBERTa-base

### Hyperparameter
| Parameter | Value |
|---|---|
| Epochs | 9 |
| Learning Rate | 3e-4 |
| Train Batch Size | 32 |
| Eval Batch Size | 64 |
| Weight Decay | 0.0277 |
| Trainable Parameters | 890,118 (0.32%) |
| Training Time | 5.02 menit |

### Test Results
| Metric | Value |
|---|---|
| Accuracy | 0.7552 |
| Precision (Macro) | 0.7533 |
| Recall (Macro) | 0.7817 |
| F1-Score (Macro) | 0.7634 |

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Anger | 0.7803 | 0.7500 | 0.7649 | 180 |
| Fear | 0.7073 | 0.8657 | 0.7785 | 134 |
| Joy | 0.8342 | 0.7500 | 0.7899 | 208 |
| Love | 0.7521 | 0.8713 | 0.8073 | 101 |
| Neutral | 0.7602 | 0.6426 | 0.6965 | 291 |
| Sad | 0.6857 | 0.8108 | 0.7430 | 148 |

### Confusion Matrix
![Confusion Matrix - XLM-RoBERTa-base](/image/cm_xlm_r_base.png)

---

## 4. BERT-base-multilingual-cased (mBERT)

### Hyperparameter
| Parameter | Value |
|---|---|
| Epochs | 10 |
| Learning Rate | 1e-4 |
| Train Batch Size | 32 |
| Eval Batch Size | 32 |
| Weight Decay | 0.0767 |
| Trainable Parameters | 299,526 (0.17%) |
| Training Time | 5.80 menit |

### Test Results
| Metric | Value |
|---|---|
| Accuracy | 0.4896 |
| Precision (Macro) | 0.4932 |
| Recall (Macro) | 0.4279 |
| F1-Score (Macro) | 0.4304 |

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Anger | 0.3358 | 0.2556 | 0.2902 | 180 |
| Fear | 0.4403 | 0.4403 | 0.4403 | 134 |
| Joy | 0.6215 | 0.6394 | 0.6303 | 208 |
| Love | 0.6000 | 0.1485 | 0.2381 | 101 |
| Neutral | 0.4855 | 0.7457 | 0.5881 | 291 |
| Sad | 0.4762 | 0.3378 | 0.3953 | 148 |

### Confusion Matrix
![Confusion Matrix - mBERT](/image/cm_mbert.png)

---

## Rekapitulasi Perbandingan Model

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Train Time |
|---|---|---|---|---|---|
| **XLM-RoBERTa-large** | **0.7985** | **0.8000** | **0.8046** | **0.8012** | 19.91 min |
| IndoBERTweet-base | 0.7665 | 0.7668 | 0.7796 | 0.7719 | 8.70 min |
| XLM-RoBERTa-base | 0.7552 | 0.7533 | 0.7817 | 0.7634 | 5.02 min |
| mBERT | 0.4896 | 0.4932 | 0.4279 | 0.4304 | 5.80 min |

## Kesimpulan

1. **XLM-RoBERTa-large** menghasilkan performa terbaik di semua metrik dengan accuracy 79.85% dan F1-score macro 80.12%, namun memerlukan waktu training paling lama (19.91 menit).
2. **IndoBERTweet-base** menempati posisi kedua dengan accuracy 76.65% dan F1-score macro 77.19%, dengan waktu training yang lebih efisien (8.70 menit).
3. **XLM-RoBERTa-base** memberikan performa yang kompetitif (accuracy 75.52%) dengan waktu training paling cepat (5.02 menit).
4. **mBERT** menunjukkan performa terendah (accuracy 48.96%) pada dataset imbalanced ini, kemungkinan karena model ini kurang optimal untuk data tweet berbahasa Indonesia.
5. Semua model menggunakan LoRA dengan rank=8 sehingga hanya melatih sebagian kecil parameter (0.17% - 0.33%), menunjukkan efisiensi PEFT dalam fine-tuning.

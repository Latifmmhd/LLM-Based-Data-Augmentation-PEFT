### 1. IndoBERTweet (`indolem/indobertweet-base-uncased`)
**Best HP:** 3 epochs, lr=3e-4, train_bs=16, eval_bs=32, wd=0.0421

#### Validation per Epoch

| Epoch | Accuracy | Precision | Recall | F1 | Loss |
|:-----:|:--------:|:---------:|:------:|:----:|:----:|
| 1 | 0.7661 | 0.7690 | 0.7710 | 0.7617 | 0.7099 |
| 2 | 0.7714 | 0.7705 | 0.7761 | 0.7680 | 0.6645 |
| 3 | **0.7899** | **0.7890** | **0.7934** | **0.7896** | **0.6318** |

#### Test Set (best model @ epoch 3)

| Accuracy | Precision | Recall | F1 |
|:--------:|:---------:|:------:|:----:|
| **0.8021** | **0.8018** | **0.8030** | **0.7994** |

---

### 2. XLM-RoBERTa Large (`FacebookAI/xlm-roberta-large`)
**Best HP:** 7 epochs, lr=1e-4, train_bs=8, eval_bs=8, wd=0.0996

#### Validation per Epoch

| Epoch | Accuracy | Precision | Recall | F1 |
|:-----:|:--------:|:---------:|:------:|:----:|
| 1 | 0.7624 | 0.7656 | 0.7676 | 0.7558 |
| 2 | 0.7730 | 0.7736 | 0.7765 | 0.7721 |
| 3 | 0.7730 | 0.7734 | 0.7757 | 0.7721 |
| 4 | 0.7905 | 0.7899 | 0.7935 | 0.7888 |
| 5 | 0.7915 | 0.7926 | 0.7944 | 0.7890 |
| 6 | 0.7952 | 0.7946 | 0.7976 | 0.7942 |
| 7 | **0.7974** | **0.7973** | **0.8005** | **0.7957** |

#### Test Set (best model @ epoch 7)

| Accuracy | Precision | Recall | F1 |
|:--------:|:---------:|:------:|:----:|
| **0.8053** | **0.8062** | **0.8054** | **0.8032** |

---

### 3. XLM-RoBERTa Base (`FacebookAI/xlm-roberta-base`)
**Best HP:** 9 epochs, lr=3e-4, train_bs=8, eval_bs=64, wd=0.0619

#### Validation per Epoch

| Epoch | Accuracy | Precision | Recall | F1 |
|:-----:|:--------:|:---------:|:------:|:----:|
| 1 | 0.7148 | 0.7360 | 0.7214 | 0.6995 |
| 2 | 0.7688 | 0.7699 | 0.7720 | 0.7685 |
| 3 | 0.7513 | 0.7559 | 0.7541 | 0.7466 |
| 4 | 0.7561 | 0.7644 | 0.7607 | 0.7481 |
| 5 | 0.7825 | 0.7846 | 0.7847 | 0.7796 |
| 6 | 0.7815 | 0.7814 | 0.7847 | 0.7791 |
| 7 | 0.7751 | 0.7782 | 0.7769 | 0.7748 |
| 8 | 0.7873 | 0.7883 | 0.7904 | 0.7848 |
| 9 | **0.7884** | **0.7878** | **0.7911** | **0.7870** |

#### Test Set (best model @ epoch 9)

| Accuracy | Precision | Recall | F1 |
|:--------:|:---------:|:------:|:----:|
| **0.7831** | **0.7810** | **0.7841** | **0.7804** |

---

### 4. mBERT (`google-bert/bert-base-multilingual-cased`)
**Best HP:** 7 epochs, lr=1e-4, train_bs=8, eval_bs=8, wd=0.0966

#### Validation per Epoch

| Epoch | Accuracy | Precision | Recall | F1 | Loss |
|:-----:|:--------:|:---------:|:------:|:----:|:----:|
| 1 | 0.5360 | 0.5254 | 0.5410 | 0.5188 | 1.2509 |
| 2 | 0.6280 | 0.6456 | 0.6345 | 0.6251 | 1.0955 |
| 3 | 0.6714 | 0.6686 | 0.6760 | 0.6688 | 0.9976 |
| 4 | 0.6804 | 0.6779 | 0.6857 | 0.6753 | 0.9904 |
| 5 | 0.6979 | 0.6994 | 0.7029 | 0.6959 | 0.9315 |
| 6 | 0.6989 | 0.6975 | 0.7033 | 0.6975 | 0.9230 |
| 7 | **0.6968** | **0.6958** | **0.7014** | **0.6952** | — |

#### Test Set (best model @ epoch 7)

| Accuracy | Precision | Recall | F1 |
|:--------:|:---------:|:------:|:----:|
| **0.7201** | **0.7172** | **0.7210** | **0.7168** |

---

## Struktur Direktori

```
.
├── README.md                                  # Dokumentasi proyek & sitasi
├── summary.md                                 # File ini
├── requirements.txt                           # Dependensi Python
├── catatan_2026-07-24.md                      # Catatan observasi proyek
│
├── bert_fine_tuning_peft.py                   # Modul inti: LoRA init, tokenisasi, metrik
├── hp_tuning_peft.py                          # Hyperparameter tuning (Optuna)
├── run_hp_tuning_peft.sh                      # Shell script HP tuning 4 model
│
├── split_data.ipynb                           # Notebook split data
├── ft_with_peft_augmented_bnf16.ipynb         # Notebook fine-tuning utama
├── Augmentasi_indonesia_Tweet_Model_Base_bnf16.ipynb  # Notebook augmentasi data
│
├── dataset/
│   ├── all_data_augmented_bnf16.csv           # Dataset augmented (12.600 baris)
│   ├── train.csv                              # Train set (8.820)
│   ├── val.csv                                # Val set (1.890)
│   ├── test.csv                               # Test set (1.890)
│   └── get_data_info.ipynb                    # Statistik dataset
│
├── best_params/                               # Hasil hyperparameter tuning
│   ├── p-indobertweet.json                    # Best HP: IndoBERTweet
│   ├── p-xlm-r-large.json                     # Best HP: XLM-RoBERTa Large
│   ├── p-xlm-r-base.json                      # Best HP: XLM-RoBERTa Base
│   └── p-mbert.json                           # Best HP: mBERT
│
├── p-emotcls-indobertweet-base/               # LoRA adapter terlatih (IndoBERTweet)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── training_args.bin
│   ├── tokenizer.json / tokenizer_config.json
│   ├── special_tokens_map.json / vocab.txt
│   ├── checkpoint-1656/
│   ├── runs/                                  # TensorBoard logs
│   └── README.md
│
├── p-emotcls-xlm-r-large/                     # LoRA adapter terlatih (XLM-RoBERTa Large)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── ...
│
├── p-emotcls-xlm-r-base/                      # LoRA adapter terlatih (XLM-RoBERTa Base)
│   ├── ...
│
├── p-emotcls-bert-base/                       # LoRA adapter terlatih (mBERT)
│   ├── ...
│
└── wandb/                                     # Weights & Biases logging
```

---

## Dependensi (`requirements.txt`)

```
absl-py
accelerate
adapter-transformers
datasets
huggingface-hub
numpy
pandas
peft @ git+https://github.com/huggingface/peft
scikit-learn
torch
transformers
tqdm
```

---

## Alur Kerja Pipeline

```
1. Augmentasi Data
   └── Augmentasi_indonesia_Tweet_Model_Base_bnf16.ipynb
       Input:  7.080 tweet (imbalanced, 6 kelas)
       Proses: Llama 3 8B (bf16) → few-shot paraphrasing
       Output: 12.600 tweet (balanced, @2.100/kelas) → all_data_augmented_bnf16.csv

2. Split Data
   └── split_data.ipynb
       Input:  all_data_augmented_bnf16.csv
       Proses: two-stage train_test_split (random_state=42)
       Output: train.csv (70%), val.csv (15%), test.csv (15%)

3. Hyperparameter Tuning (4 model × 10 trials)
   └── run_hp_tuning_peft.sh → hp_tuning_peft.py
       Input:  train.csv, val.csv
       Proses: Optuna → minimalkan training_loss
       Output: best_params/p-{model}.json

4. Fine-Tuning Utama
   └── ft_with_peft_augmented_bnf16.ipynb
       Input:  train.csv, val.csv, test.csv + best_params/*.json
       Proses: PEFT + LoRA → Trainer → eval per epoch → load best model
       Output: p-emotcls-{model}/ (adapter LoRA + tokenizer + checkpoint)

5. Evaluasi Akhir
   └── Test set (1.890 tweet)
       Metrik: accuracy, precision, recall, F1 (macro)
       Visualisasi: confusion matrix, loss curve
```

---
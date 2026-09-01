import numpy as np
import pandas as pd
import os
import argparse
import json
import optuna


from bert_fine_tuning_peft import (
    model_init,
    preprocessing,
    compute_metrics,
)
from transformers import (
    TrainingArguments,
    Trainer,
)

    
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parse = argparse.ArgumentParser()
parse.add_argument('--model_name', type=str, default=None)
parse.add_argument('--output_dir', type=str, default=None)
parse.add_argument('--num_trials', type=int, default=None)
parse.add_argument('--save_best_params', type=str, default=None)

args = parse.parse_args()

# Dataset preparation
all_data = pd.read_csv('dataset/all_data_augmented_bnf16.csv')
train_data = pd.read_csv('dataset/train.csv')
val_data = pd.read_csv('dataset/val.csv')
test_data = pd.read_csv('dataset/test.csv')


tags = np.unique(all_data['label'])
num_tags = len(tags)
label2id = {t: i for i, t in enumerate(tags)}
id2label = {i: t for i, t in enumerate(tags)}


def objective(trial: optuna.Trial):
    tr_model, tokenizer = model_init(args.model_name)
    train_tokenized, val_tokenized, test_tokenized = preprocessing()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=trial.suggest_int("num_train_epochs", 1, 10),
        learning_rate=trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 2e-5, 3e-5, 5e-5]),
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64]),
        per_device_eval_batch_size=trial.suggest_categorical("per_device_eval_batch_size", [8, 16, 32, 64]),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1)
    )

    trainer = Trainer(
        model=tr_model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    result = trainer.train()
    return result.training_loss

# We want to minimize the loss!
study = optuna.create_study(study_name="hyper-parameter-search")
study.optimize(func=objective, n_trials=args.num_trials)
print("\n")
print("Best value: ",study.best_value)
print("\n")
print("Best params: ",study.best_params)
print("\n")
print("Best trial: ",study.best_trial)
# save best params
filename = args.save_best_params
dir_name = filename.split("/")[0]

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

with open(f"{filename}", "w") as outfile:
    json.dump(study.best_params, outfile)
    print("JSON file created!")

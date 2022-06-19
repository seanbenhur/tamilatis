import json
import logging
import os
import pickle
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer, HfArgumentParser, Trainer,
                          TrainingArguments)

logger = logging.getLogger(__name__)


@dataclass
class DataArgs:
    train_data_path: str = "/content/drive/MyDrive/datasets/tamilatis/train_intent.pkl"  # path to the training dataset
    dev_data_path: str = "/content/drive/MyDrive/datasets/tamilatis/val_intent.pkl"  # path to the validation dataset
    test_data_path: str = "/content/drive/MyDrive/datasets/tamilatis/test_intent.pkl"  # path to the test dataset
    num_labels: int = 23
    test_predictions_file: str = "predictions.csv"


@dataclass
class ModelArgs:
    model_name_or_path: str = "microsoft/xlm-align-base"
    tokenizer_name_or_path: str = "microsoft/xlm-align-base"
    task_name: str = "tamilatis-models"
    run_name: str = "firstrun"


@dataclass
class TrainingArgs:
    output_dir: str = "results"
    num_epochs: int = 5
    fold_num: int = 4
    logging_steps: int = 5000
    load_best_model_at_end: bool = True
    save_strategy: str = "epoch"
    train_bs: int = 32
    dev_bs: int = 32
    learning_rate: int = 5e-5
    evaluation_strategy: str = "epoch"
    scheduler: str = "cosine"
    warmup_steps: int = 0
    weight_decay: float = 0.01
    seed: int = 42
    root_dir: str = "/content/models"


@dataclass
class WandbArgs:
    project_name: str = "tamilatis"
    group_name: str = "single-task-learning-intent-xlm-align"


data_args, model_args, training_args, wandb_args = (
    DataArgs,
    ModelArgs,
    TrainingArgs,
    WandbArgs,
)
tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path, num_labels=data_args.num_labels
)

os.environ["WANDB_PROJECT"] = wandb_args.project_name
os.environ["WANDB_RUN_GROUP"] = wandb_args.group_name


class BuildDataset:
    def __init__(self):
        pass

    def tokenize(self, text):
        """Splits the text and get offsets"""
        text = text.strip()
        tokens = text.split()
        offsets = []
        for token in tokens:
            start_idx = text.find(token)
            end_idx = start_idx + len(token)
            offsets.append([start_idx, end_idx])
        return tokens, offsets

    def convert_to_boi(self, text, annotations):
        """Convert Intent Tags to BOI Tags"""
        tokens, offsets = self.tokenize(text)
        boi_tags = ["O"] * len(tokens)

        for name, value, [start_idx, end_idx] in annotations:
            value = value.strip()
            try:
                token_span = len(value.split())

                start_token_idx = [
                    token_idx
                    for token_idx, (s, e) in enumerate(offsets)
                    if s == start_idx
                ][0]
                end_token_idx = start_token_idx + token_span
                annotation = [name] + ["I" + name[1:]] * (token_span - 1)
                boi_tags[start_token_idx:end_token_idx] = annotation
            except Exception as error:
                pass

        return list(zip(tokens, boi_tags))

    def build_dataset(self, path):
        """Build a TOD dataset"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        boi_data = []
        for text, annotation, intent in tqdm(data):
            boi_item = self.convert_to_boi(text, annotation)
            is_valid = any([True for token, tag in boi_item if tag != "O"])
            wrong_intent = intent[0] == "B" or intent[0] == "I"

            if is_valid and not wrong_intent:
                boi_data.append((boi_item, intent))
        return boi_data


class ATISDataset(Dataset):
    def __init__(self, data, tokenizer, intent_encoder):

        self.data = data
        self.intent_encoder = intent_encoder
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        tokens = [token for token, annotation in self.data[idx][0]]
        tags = [tag for token, tag in self.data[idx][0]]

        intent_name = self.data[idx][1]

        intent_label = self.intent_encoder.transform([intent_name])

        text = " ".join(tokens)

        encoding = self.tokenizer(
            text,
            max_length=60,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)

        return {
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": intent_label.item(),
        }


# Get all tags
annotations = set()
intents = set()
count = 0

data_utils = BuildDataset()
train_data = data_utils.build_dataset(data_args.train_data_path)
valid_data = data_utils.build_dataset(data_args.dev_data_path)
test_data = data_utils.build_dataset(data_args.test_data_path)

annotations, intents, count = set(), set(), 0
for boi_data, intent in train_data:
    if intent[0] == "B" or intent[0] == "I":
        count += 1
    intents.add(intent)
    for token, annotation in boi_data:
        annotations.add(annotation)

for boi_data, intent in valid_data:
    if intent[0] == "B" or intent[0] == "I":
        count += 1
    intents.add(intent)
    for token, annotation in boi_data:
        annotations.add(annotation)
    for boi_data, intent in test_data:
        if intent[0] == "B" or intent[0] == "I":
            count += 1
        intents.add(intent)
        for token, annotation in boi_data:
            annotations.add(annotation)

annotations = list(annotations)
intents = list(intents)

intent_encoder = LabelEncoder()
intent_encoder.fit(intents)
train_ds = ATISDataset(train_data, tokenizer, intent_encoder)

val_ds = ATISDataset(valid_data, tokenizer, intent_encoder)
test_ds = ATISDataset(test_data, tokenizer, intent_encoder)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


training_args = TrainingArguments(
    output_dir=f"{model_args.task_name}/{model_args.run_name}",
    logging_steps=training_args.logging_steps,
    num_train_epochs=training_args.num_epochs,
    learning_rate=training_args.learning_rate,
    per_device_train_batch_size=training_args.train_bs,
    per_device_eval_batch_size=training_args.dev_bs,
    warmup_steps=training_args.warmup_steps,
    weight_decay=training_args.weight_decay,
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)
train_results = trainer.train()
train_metrics = train_results.metrics
eval_metrics = trainer.evaluate(val_ds)
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)
logger.info("***Predictiong for Test File***")
predictions = trainer.predict(test_ds)
y_pred = np.argmax(predictions.predictions, axis=1)
y_pred = intent_encoder.inverse_transform(y_pred)
predictions = intent_encoder.inverse_transform(predictions.label_ids)
pred_df = pd.DataFrame(list(zip(predictions.label_ids, y_pred)))
pred_df.columns = ["Actual", "Preds"]

pred_df = pd.DataFrame(list(zip(predictions, y_pred)))
pred_df.columns = ["Actual", "Preds"]
pred_df.to_csv("xlm_align_base_intent.csv")

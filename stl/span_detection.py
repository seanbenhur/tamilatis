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
from seqeval.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from transformers import (AutoModel, AutoModelForTokenClassification,
                          AutoTokenizer, HfArgumentParser, Trainer,
                          TrainingArguments)

logger = logging.getLogger(__name__)


@dataclass
class DataArgs:
    train_data_path: str = "/content/drive/MyDrive/datasets/tamilatis/train_intent.pkl"  # path to the training dataset
    dev_data_path: str = "/content/drive/MyDrive/datasets/tamilatis/val_intent.pkl"  # path to the validation dataset
    test_data_path: str = "/content/drive/MyDrive/datasets/tamilatis/test_intent.pkl"  # path to the test dataset
    num_labels: int = 78
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
    logging_steps: int = 5000
    load_best_model_at_end: bool = True
    save_strategy: str = "epoch"
    train_bs: int = 32
    dev_bs: int = 32
    learning_rate: int = 5e-5
    evaluation_strategy: str = "epoch"
    scheduler: str = "cosine"
    warmup_steps: int = 100
    weight_decay: float = 0.01
    seed: int = 42
    root_dir: str = "/content/models"


@dataclass
class WandbArgs:
    project_name: str = "tamilatis"
    group_name: str = "single-task-learning-span-xlm-align"


data_args, model_args, training_args, wandb_args = (
    DataArgs,
    ModelArgs,
    TrainingArgs,
    WandbArgs,
)
os.environ["WANDB_PROJECT"] = wandb_args.project_name
os.environ["WANDB_RUN_GROUP"] = wandb_args.group_name


tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
model = AutoModelForTokenClassification.from_pretrained(
    model_args.model_name_or_path, num_labels=data_args.num_labels
)


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
    def __init__(self, data, tokenizer, label_encoder):
        self.data = data
        self.label_encoder = label_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, add_extra_space=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = [token for token, annotation in self.data[idx][0]]
        tags = [tag for token, tag in self.data[idx][0]]

        text = "#".join(tokens)

        encoding = self.tokenizer(
            tokens,
            max_length=60,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
        )

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        word_ids = encoding.word_ids()

        tags = self.label_encoder.transform(tags)

        labels = []
        label_all_tokens = None
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(tags[word_idx])
            else:
                labels.append(tags[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels = torch.tensor(labels)
        tags = tags.tolist()
        tags.extend([-100] * (50 - len(tags)))

        return {
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "tags": tags,
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

label_encoder = LabelEncoder()
label_encoder.fit(annotations)

logger.info("Creating Dataset")

train_ds = ATISDataset(train_data, model_args.tokenizer_name_or_path, label_encoder)

val_ds = ATISDataset(valid_data, model_args.tokenizer_name_or_path, label_encoder)
test_ds = ATISDataset(test_data, model_args.tokenizer_name_or_path, label_encoder)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


training_args = TrainingArguments(
    output_dir=f"{model_args.task_name}/{model_args.run_name}",
    logging_steps=training_args.logging_steps,
    num_train_epochs=5,
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


def get_predictions(sentence):

    inputs = tokenizer(
        sentence.split(),
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=60,
        return_tensors="pt",
    )

    # move to gpu
    ids = inputs["input_ids"].cuda()
    mask = inputs["attention_mask"].cuda()
    # forward pass
    outputs = model(input_ids=ids, attention_mask=mask, labels=None)
    # logits = outputs.logits
    print(outputs.logits)

    active_logits = outputs.logits.view(
        -1, data_args.num_labels
    )  # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(
        active_logits, axis=1
    )  # shape (batch_size*seq_len,) - predictions at the token level

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = label_encoder.inverse_transform(
        [i for i in flattened_predictions.cpu().numpy()]
    )
    wp_preds = list(
        zip(tokens, token_predictions)
    )  # list of tuples. Each tuple = (wordpiece, prediction)

    slot_prediction = []
    for token_pred, mapping in zip(
        wp_preds, inputs["offset_mapping"].squeeze().tolist()
    ):
        # only predictions on first word pieces are important
        if mapping[0] == 0 and mapping[1] != 0 and token_pred[0] != "▁":
            slot_prediction.append(token_pred[1])
        else:
            continue

    return slot_prediction


outputs = []
intents = []

for item, intent in tqdm(test_data):
    tokens = [token for token, tag in item]
    tags = [tag for token, tag in item]
    text = " ".join(tokens)
    slot_preds = get_predictions(text)
    print(slot_preds)
    outputs.append((tags, slot_preds))


for output in tqdm(outputs):
    assert len(output[0]) == len(output[1])

y_true = [output[0] for output in outputs]
y_pred = [output[1] for output in outputs]


print(classification_report(y_true, y_pred))
print(accuracy_score(y_true, y_pred))

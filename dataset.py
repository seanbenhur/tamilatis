import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


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
        return list(zip(tokens, boi_tags))

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
    def __init__(self, data, tokenizer, label_encoder, intent_encoder):
        self.data = data
        self.label_encoder = label_encoder
        self.intent_encoder = intent_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = [token for token, annotation in self.data[idx][0]]
        tags = [tag for token, tag in self.data[idx][0]]

        intent_name = self.data[idx][1]
        intent_label = self.intent_encoder.transform([intent_name])
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
            "intent": intent_label.item(),
            "tags": tags,
        }

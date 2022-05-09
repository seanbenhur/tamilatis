import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


class TamilATISPredictor:
    def __init__(
        self,
        model,
        checkpoint_path,
        tokenizer,
        label_encoder,
        intent_encoder,
        num_labels,
    ):
        self.model = model
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_labels = num_labels
        self.label_encoder = label_encoder
        self.intent_encoder = intent_encoder

    def get_predictions(self, text):

        inputs = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=60,
            return_tensors="pt",
        )
        ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)

        # forward pass
        loss_dict = self.model(input_ids=ids, attention_mask=mask, labels=None)
        slot_logits, intent_logits, slot_loss = (
            loss_dict["dst_logits"],
            loss_dict["intent_loss"],
            loss_dict["dst_loss"],
        )

        active_logits = slot_logits.view(
            -1, self.num_labels
        )  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(
            active_logits, axis=1
        )  # shape (batch_size*seq_len,) - predictions at the token level
        tokens = self.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = self.label_encoder.inverse_transform(
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
        intent_preds = torch.argmax(intent_logits, axis=1)
        intent_preds = self.intent_encoder.inverse_transform(intent_preds.cpu().numpy())

        return intent_preds, slot_prediction

    def predict_test_data(self, test_data):
        outputs = []
        intents = []

        for item, intent in tqdm(test_data):
            try:
                tokens = [token for token, tag in item]
                tags = [tag for token, tag in item]
                text = " ".join(tokens)
                intent_preds, slot_preds = self.get_predictions(text)
                outputs.append((tags, slot_preds))
                intents.append((intent, intent_preds.item()))
            except Exception as error:
                print(error)
        return outputs, intents

    def evaluate(self, outputs, intents):
        for output in tqdm(outputs):
            assert len(output[0]) == len(output[1])
        y_true = [output[0] for output in outputs]
        y_pred = [output[1] for output in outputs]
        from seqeval.metrics import classification_report

        ner_cls_rep = classification_report(y_true, y_pred, output_dict=True)
        from sklearn.metrics import classification_report

        # Compute metrics for intent
        y_true = self.intent_encoder.transform(
            [output[0] for output in intents]
        ).tolist()
        y_pred = self.intent_encoder.transform(
            [output[1] for output in intents]
        ).tolist()

        target_names = self.intent_encoder.classes_.tolist()
        target_names = [target_names[idx] for idx in np.unique(y_true + y_pred)]
        intent_cls_rep = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        )

        return ner_cls_rep, intent_cls_rep

import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModelForTokenClassification


class JointATISModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_name, num_labels, num_intents):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model_config = AutoConfig.from_pretrained(model_name)
        self.intent_head = nn.Linear(self.model_config.hidden_size, num_intents)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids, attention_mask, labels=labels, output_hidden_states=True
        )
        pooled_output = outputs["hidden_states"][-1][:, 0, :]
        intent_logits = self.intent_head(pooled_output)
        return {
            "dst_logits": outputs.logits,
            "intent_loss": intent_logits,
            "dst_loss": outputs.loss,
        }

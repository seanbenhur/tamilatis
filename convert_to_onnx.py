from dataset import BuildDataset, ATISDataset
from model import JointATISModel
from sklearn.preprocessing import LabelEncoder
import logging
import torch
from torch.utils.data import DataLoader
import hydra

logger = logging.getLogger(__name__)


test_data_path = "/content/drive/MyDrive/datasets/tamilatis/test_intent.pkl"
model_path = "/content/drive/MyDrive/models/tamilatis/best_model.bin"
tokenizer_name = "xlm-roberta-base"
model_name = "xlm-roberta-base"
num_labels = 78
num_intents = 23


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    logger.info("Building Dataset")
    data_utils = BuildDataset()
    test_data = data_utils.build_dataset(cfg.dataset.test_path)

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    annotations, intents, count = set(), set(), 0
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

    intent_encoder = LabelEncoder()
    intent_encoder.fit(intents)

    test_ds = ATISDataset(
        test_data, cfg.model.tokenizer_name, label_encoder, intent_encoder
    )

    test_dl = DataLoader(test_ds, batch_size=1, pin_memory=True)
    logging.info("DataLoaders Created")

    example = next(iter(test_dl))

    model = JointATISModel(
        cfg.model.model_name, cfg.model.num_labels, cfg.model.num_intents
    )
    model.load_state_dict(torch.load(cfg.model.test_model,map_location=device))
    logging.info(f"Model loaded from {cfg.model.test_model}")
    model.eval()
    input_ids = example["input_ids"]
    attention_mask = example["attention_mask"]
    labels = example["labels"]

    logging.info("Converting Pytorch Model to ONNX Model")
    torch.onnx.export(
        model,
        (input_ids, attention_mask, labels),
        "model.onnx",
        input_names=["input_ids", "attention_mask", "labels"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "labels": {0: "batch_size", 1: "sequence"},
        },
    )

    logging.info("Model is converted to ONNX")


if __name__ == "__main__":
    main()

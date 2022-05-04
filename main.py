import logging
import os
import pickle

import wandb
import hydra
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from omegaconf.omegaconf import OmegaConf
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler

from dataset import ATISDataset, BuildDataset
from model import JointATISModel
from predict import TamilATISPredictor
from trainer import ATISTrainer

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):

    os.environ['WANDB_PROJECT'] = cfg.wandb.project_name
    os.environ['WANDB_RUN_GROUP'] = cfg.wandb.group_name
    os.environ['WANDB_NAME'] = cfg.wandb.run_name
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    accelerator = Accelerator()
    # Get all tags
    annotations = set()
    intents = set()
    count = 0

    logger.info("Building Dataset")
    data_utils = BuildDataset()
    train_data = data_utils.build_dataset(cfg.dataset.train_path)
    valid_data = data_utils.build_dataset(cfg.dataset.valid_path)
    test_data = data_utils.build_dataset(cfg.dataset.test_path)

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

    # convert string labels to int
    label_encoder = LabelEncoder()
    label_encoder.fit(annotations)

    intent_encoder = LabelEncoder()
    intent_encoder.fit(intents)

    train_ds = ATISDataset(
        train_data, cfg.model.tokenizer_name, label_encoder, intent_encoder
    )
    val_ds = ATISDataset(
        valid_data, cfg.model.tokenizer_name, label_encoder, intent_encoder
    )
    test_ds = ATISDataset(
        test_data, cfg.model.tokenizer_name, label_encoder, intent_encoder
    )

    train_dl = DataLoader(train_ds, batch_size=cfg.training.batch_size, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.training.batch_size * 2, pin_memory=True)
    test_dl = DataLoader(
        test_ds, batch_size=cfg.training.batch_size * 2, pin_memory=True
    )
    logging.info("DataLoaders are created!")

    model = JointATISModel(
        cfg.model.model_name, cfg.model.num_labels, cfg.model.num_intents
    )
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=cfg.training.lr)
    nb_train_steps = int(
        len(train_dl) / cfg.training.batch_size * cfg.training.max_epochs
    )

    scheduler = get_scheduler(
        cfg.training.scheduler,
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=nb_train_steps,
    )

    model, optimizer, train_dl, val_dl = accelerator.prepare(
        model, optimizer, train_dl, val_dl
    )
    # Register the LR scheduler
    accelerator.register_for_checkpointing(scheduler)
    run = wandb.init("tamilatis", "test")
    if cfg.training.do_train:
        trainer = ATISTrainer(
            model,
            optimizer,
            scheduler,
            criterion,
            accelerator,
            cfg.dataset.output_dir,
            cfg.dataset.num_labels,
            cfg.dataset.num_intents,
            run
        )
        best_model, best_loss = trainer.fit(
            cfg.training.max_epochs, train_dl, val_dl, cfg.training.patience
        )
        model_dir = f"{cfg.dataset.output_dir}/model_{best_loss}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        best_model.save_pretrained(model_dir, push_to_hub=False)
        logging.info(
            f"The Best model with validation loss {best_loss} is saved in {model_dir}"
        )
    if cfg.training.do_predict:
        predictor = TamilATISPredictor(
            model,
            cfg.model.test_model,
            cfg.model.tokenizer_name,
            label_encoder,
            intent_encoder,
            cfg.model.num_labels,
        )
        outputs, intents = predictor.predict_test_data(test_data)
        ner_cls_rep, intent_cls_rep = predictor.evaluate(outputs, intents)
        ner_cls_df = pd.DataFrame(ner_cls_rep).transpose()
        intent_cls_df = pd.DataFrame(intent_cls_rep).transpose()
        ner_cls_df.to_csv(cfg.training.ner_cls_path)
        intent_cls_df.to_csv(cfg.training.intent_cls_path)
        logging.info(
            f"Classification reports of intents and slots are saved in {cfg.training.ner_cls_path} and {cfg.training.intent_cls_path}"
        )


if __name__ == "__main__":
    main()

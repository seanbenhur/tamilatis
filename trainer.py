import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchmetrics.functional import accuracy, f1_score, precision, recall
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


class ATISTrainer:
    """A Trainer class consists of utitlity functions for training the model"""
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        accelerate,
        output_dir,
        num_labels,
        num_intents,
        run
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerate
        self.output_dir = output_dir
        self.num_labels = num_labels
        self.num_intents = num_intents

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.run = run
        logging.info(f"Strating Training, outputs are saved in {self.output_dir}")

    def train_step(self, iterator):
        training_progress_bar = tqdm(iterator, desc="training")
        for batch in training_progress_bar:
            input_ids, attention_mask, labels, intents = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
                batch["intent"],
            )
            self.optimizer.zero_grad()
            loss_dict = self.model(input_ids, attention_mask, labels)
            slot_logits, intent_logits, slot_loss = (
                loss_dict["dst_logits"],
                loss_dict["intent_loss"],
                loss_dict["dst_loss"],
            )

            # compute training accuracy for slots
            flattened_target_labels = batch["labels"].view(
                -1
            )  # [batch_size * seq_len, ]
            active_logits = slot_logits.view(
                -1, self.num_labels
            )  # [batch_size* seq_len, num_labels]
            flattened_preds = torch.argmax(
                active_logits, axis=-1
            )  # [batch_size * seq_len,]

            # compute accuracy at active labels
            active_accuracy = (
                batch["labels"].view(-1) != -100
            )  # [batch_size * seq_len, ]

            slot_labels = torch.masked_select(flattened_target_labels, active_accuracy)
            slot_preds = torch.masked_select(flattened_preds, active_accuracy)

            # compute loss for intents
            #use rlw
            intent_loss = self.criterion(intent_logits, batch["intent"])
            weight = F.softmax(torch.randn(1), dim=-1) # RLW is only this!
            intent_loss = torch.sum(intent_loss*weight.cuda())
            intent_preds = torch.argmax(intent_logits, axis=1)
            train_loss = slot_loss + intent_loss
            self.accelerator.backward(train_loss)
            self.optimizer.step()

            if self.scheduler is not None:
              if not self.accelerator.optimizer_step_was_skipped:
                self.scheduler.step()

            if self.scheduler is not None:
              self.scheduler.step()

            intent_acc = accuracy(
                intent_preds, intents, num_classes=self.num_intents, average="weighted"
            )
            intent_f1 = f1_score(
                intent_preds, intents, num_classes=self.num_intents, average="weighted"
            )
            intent_rec = recall(
                intent_preds, intents, num_classes=self.num_intents, average="weighted"
            )
            intent_prec = precision(
                intent_preds, intents, num_classes=self.num_intents, average="weighted"
            )

            slot_acc = accuracy(
                slot_preds, slot_labels, num_classes=self.num_labels, average="weighted"
            )
            slot_f1 = f1_score(
                slot_preds, slot_labels, num_classes=self.num_labels, average="weighted"
            )
            slot_rec = recall(
                slot_preds, slot_labels, num_classes=self.num_labels, average="weighted"
            )
            slot_prec = precision(
                slot_preds, slot_labels, num_classes=self.num_labels, average="weighted"
            )

            self.run.log(
                {
                    "train_loss_step": train_loss.cpu().detach().numpy(),
                    "train_intent_acc_step": intent_acc,
                    "train_intent_f1_step": intent_f1,
                    "train_slot_acc_step": slot_acc,
                    "train_slot_f1_step": slot_f1,
                }
            )
        # logging.info({"train_loss_step": train_loss, "train_intent_acc_step": intent_acc, "train_intent_f1_step": intent_f1, "train_slot_acc_step": slot_acc, "train_slot_f1_step": slot_f1 })

        return {
            "train_loss_epoch": train_loss / len(iterator),
            "train_intent_f1_epoch": intent_f1 / len(iterator),
            "train_intent_acc_epoch": intent_acc / len(iterator),
            "train_slot_f1_epoch": slot_f1 / len(iterator),
            "train_slot_acc_epoch": slot_acc / len(iterator),
        }

    @torch.no_grad()
    def eval_step(self, iterator):
        eval_progress_bar = tqdm(iterator, desc="Evaluating")
        for batch in eval_progress_bar:
            input_ids, attention_mask, labels, intents = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
                batch["intent"],
            )
            loss_dict = self.model(input_ids, attention_mask, labels)
            slot_logits, intent_logits, slot_loss = (
                loss_dict["dst_logits"],
                loss_dict["intent_loss"],
                loss_dict["dst_loss"],
            )
            # compute training accuracy for slots
            flattened_target_labels = batch["labels"].view(
                -1
            )  # [batch_size * seq_len, ]
            active_logits = slot_logits.view(
                -1, self.num_labels
            )  # [batch_size* seq_len, num_labels]
            flattened_preds = torch.argmax(
                active_logits, axis=-1
            )  # [batch_size * seq_len,]

            # compute accuracy at active labels
            active_accuracy = (
                batch["labels"].view(-1) != -100
            )  # [batch_size * seq_len, ]

            slot_labels = torch.masked_select(flattened_target_labels, active_accuracy)
            slot_preds = torch.masked_select(flattened_preds, active_accuracy)

            # compute loss for intents
            intent_loss = self.criterion(intent_logits, batch["intent"])
            weight = F.softmax(torch.randn(1), dim=-1) # RLW is only this!
            intent_loss = torch.sum(intent_loss*weight.cuda())
            
            intent_preds = torch.argmax(intent_logits, axis=1)
            eval_loss = slot_loss + intent_loss

            intent_acc = accuracy(
                intent_preds, intents, num_classes=self.num_intents, average="weighted"
            )
            intent_f1 = f1_score(
                intent_preds, intents, num_classes=self.num_intents, average="weighted"
            )
            intent_rec = recall(
                intent_preds, intents, num_classes=self.num_intents, average="weighted"
            )
            intent_prec = precision(
                intent_preds, intents, num_classes=self.num_intents, average="weighted"
            )

            slot_acc = accuracy(
                slot_preds, slot_labels, num_classes=self.num_labels, average="weighted"
            )
            slot_f1 = f1_score(
                slot_preds, slot_labels, num_classes=self.num_labels, average="weighted"
            )
            slot_rec = recall(
                slot_preds, slot_labels, num_classes=self.num_labels, average="weighted"
            )
            slot_prec = precision(
                slot_preds, slot_labels, num_classes=self.num_labels, average="weighted"
            )

            self.run.log(
                {
                    "eval_loss_step": eval_loss,
                    "eval_intent_acc_step": intent_acc,
                    "eval_intent_f1_step": intent_f1,
                    "eval_slot_acc_step": slot_acc,
                    "eval_slot_f1_step": slot_f1,
                }
            )

        return {
            "eval_loss_epoch": eval_loss / len(iterator),
            "eval_intent_f1_epoch": intent_f1 / len(iterator),
            "eval_intent_acc_epoch": intent_acc / len(iterator),
            "eval_slot_f1_epoch": slot_f1 / len(iterator),
            "eval_slot_acc_epoch": slot_acc / len(iterator),
        }

    def fit(self, n_epochs, train_dataloader, eval_dataloader, patience):
        best_eval_loss = float("inf")
        pbar = trange(n_epochs)

        for epoch in pbar:
            train_metrics_dict = self.train_step(train_dataloader)
            eval_metrics_dict = self.eval_step(eval_dataloader)
            # access all the values from the dicts
            train_loss, eval_loss = (
                train_metrics_dict["train_loss_epoch"],
                eval_metrics_dict["eval_loss_epoch"],
            )
            train_intent_f1, eval_intent_f1 = (
                train_metrics_dict["train_intent_f1_epoch"],
                eval_metrics_dict["eval_intent_f1_epoch"],
            )
            train_intent_acc, eval_intent_acc = (
                train_metrics_dict["train_intent_acc_epoch"],
                eval_metrics_dict["eval_intent_acc_epoch"],
            )
            train_slot_f1, eval_slot_f1 = (
                train_metrics_dict["train_intent_acc_epoch"],
                eval_metrics_dict["eval_intent_acc_epoch"],
            )
            train_slot_acc, eval_slot_acc = (
                train_metrics_dict["train_slot_acc_epoch"],
                eval_metrics_dict["eval_slot_acc_epoch"],
            )


            if eval_loss < best_eval_loss:
                best_model = self.model
                best_eval_loss = eval_loss

                train_logs = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "train_intent_acc": train_intent_acc,
                    "train_intent_f1": train_intent_f1,
                    "eval_intent_f1": eval_intent_f1,
                    "eval_intent_acc": eval_intent_acc,
                    "train_slot_f1": train_slot_f1,
                    "train_slot_acc": train_slot_acc,
                    "lr": {self.optimizer.param_groups[0]["lr"]: 0.2},
                }

                train_logs["patience"] = patience
                logging.info(train_logs)
                logging.info(eval_metrics_dict)

                self.accelerator.wait_for_everyone()
                model = self.accelerator.unwrap_model(self.model)
                self.accelerator.save_state(self.output_dir)
                logging.info(f"Checkpoint is saved in {self.output_dir}")

        return best_model, best_eval_loss

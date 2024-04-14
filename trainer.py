import time
import os
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from monai.data import DataLoader
from tensorboardX import SummaryWriter
import einops


@dataclass
class Trainer:
    model: nn.Module = None
    train_loader: DataLoader = None
    val_loader: DataLoader = None
    optimizer: torch.optim.Optimizer = None
    loss_func: callable = None
    acc_func: callable = None
    scheduler: torch.optim.lr_scheduler = None
    log_dir: str = None
    max_epochs: int = 1000
    val_every: int = 100
    model_inferer: callable = None
    start_epoch: int = 0
    post_label: callable = None
    post_pred: callable = None
    best_val_acc = 0.0
    epoch = start_epoch
    grad_scale: bool = False
    batch_augmentation: callable = None
    orient_axes: str = "h w d"

    def __post_init__(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.scaler = None
        if self.grad_scale:
            print("Using Gradient Scaling")
            self.scaler = GradScaler()

        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create file handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, 'errors.log'))

        # Set the logging level for the file handler
        file_handler.setLevel(logging.DEBUG)

        # Create formatter and add it to the file handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add file handler to the logger
        self.logger.addHandler(file_handler)

        print(f"Saving logs to {self.writer.logdir}")

    def train(self):

        # Save hyper-parameter values
        h_params = {
            "epochs": self.max_epochs,
            "start_epoch": self.start_epoch,
            "batch_size": self.train_loader.batch_size,
            "optimizer": self.optimizer.__class__.__name__,
            "accuracy": self.acc_func.__class__.__name__,
            "loss": self.loss_func.__class__.__name__,
            "val_every": self.val_every,
            "grad_scaler": bool(self.scaler)
        }

        for key, value in h_params.items():
            self.writer.add_text("Hyperparameters", f"{key}: {value}")

        # Start training
        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch

            # Record learning rate at this epoch
            lr = self.scheduler.get_last_lr()[0]
            print(f"{time.ctime()} Epoch: {epoch} Learning Rate: {lr:.6f}")
            self.writer.add_scalar("learning_rate", lr, epoch)
            epoch_time = time.perf_counter()

            # Run training
            train_loss, train_acc = self.train_epoch()

            # Record training metrics
            curr_epoch_str = str(epoch).rjust(len(str(self.max_epochs - 1)), "0")
            print(f"Train Epoch {curr_epoch_str}/{self.max_epochs - 1}  "
                  f"Mean Loss: {train_loss:.4f}  "
                  f"Time: {time.perf_counter() - epoch_time:.2f}s")
            self.writer.add_scalar("train_loss", train_loss, epoch)
            for i, acc in enumerate(train_acc, start=1):
                self.writer.add_scalar(f"train_acc_{i}", acc.item(), epoch)

            # Maybe evaluate validation set
            if (epoch + 1) % self.val_every == 0:
                val_loss, val_acc = self.val_epoch()
                mean_val_acc = val_acc.mean().item()

                # Record validation metrics
                print(f"Validate Epoch {curr_epoch_str}/{self.max_epochs - 1}  "
                      f"Mean Acc: {mean_val_acc:.4f}  "
                      f"Time: {time.perf_counter() - epoch_time:.2f}s")
                self.writer.add_scalar("val_loss", val_loss, epoch)
                for i, acc in enumerate(val_acc, start=1):
                    self.writer.add_scalar(f"val_acc_{i}", acc.item(), epoch)

                # Save model if improved
                if mean_val_acc > self.best_val_acc:
                    print(f"New highest accuracy {self.best_val_acc:.4f} --> {mean_val_acc:.4f}")
                    self.best_val_acc = mean_val_acc

                    self.save_checkpoint(filename="model_best.pt")

            # Step learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Save model otherwise
            # If loaded from checkpoint, accuracy may not improve and would otherwise not be save
            self.save_checkpoint(filename="model_final.pt")

        print(f"Training Finished! Best Accuracy: {self.best_val_acc:.4f}")

    def train_epoch(self):
        self.model.train()
        start_time = time.perf_counter()

        epoch_loss = 0.0
        total_epoch_acc = torch.zeros(1)
        total_not_nans = torch.zeros(1)
        for index, batch in enumerate(self.train_loader):
            # Batch augmentation acts on dict of Tensors - B 1 H W D (Patch Size)
            if self.batch_augmentation is not None:
                batch = self.batch_augmentation(batch)

            batch = self.rearrange_axes(batch)

            data, target = batch["image"], batch["label"]
            data, target = data.cuda(0), target.cuda(0)

            with autocast(dtype=torch.bfloat16):
                self.optimizer.zero_grad()

                logits = self.model(data)  # B 2 H W D (Patch Size)

                if not self.assert_finite(logits, index, msg="LOGITS"):
                    print("MAX DIFF", torch.max(torch.abs(logits[:, 1] - logits[:, 0])))
                    print("# NON FINITE: ", torch.count_nonzero(torch.logical_not(logits.isfinite())))
                    continue

                loss = self.loss_func(logits, target)  # 0-dim tensor

                if not self.assert_finite(loss, index, msg="LOSS"):
                    continue

                # Calculate metrics for train set
                # TODO - only record train metrics some of the time - or only some of the train data
                train_acc, train_not_nans = self.calc_accuracy(y_pred=logits, y=target)

                # Update model
                # Use gradient scaling to prevent underflow
                if self.grad_scale:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            # Update mean metrics
            epoch_loss += loss.item()
            total_epoch_acc = total_epoch_acc + train_acc * train_not_nans
            total_not_nans = total_not_nans + train_not_nans

            curr_epoch_str = str(self.epoch).rjust(len(str(self.max_epochs - 1)), "0")
            curr_index_str = str(index).rjust(len(str(len(self.train_loader) - 1)), "0")
            print(
                f"Train Batch {curr_epoch_str}/{self.max_epochs - 1}  {curr_index_str}/{len(self.train_loader) - 1}  "
                f"Loss: {loss.item():.4f}  "
                f"Time: {time.perf_counter() - start_time:.2f}s"
            )

            start_time = time.perf_counter()
        return epoch_loss / len(self.train_loader), total_epoch_acc / total_not_nans

    def val_epoch(self):
        self.model.eval()
        start_time = time.perf_counter()

        epoch_loss = 0.0
        total_epoch_acc = torch.zeros(1)
        total_not_nans = torch.zeros(1)
        with torch.no_grad():
            for index, batch in enumerate(self.val_loader):
                batch = self.rearrange_axes(batch)

                data, target = batch["image"], batch["label"]
                data, target = data.cuda(0), target.cuda(0)  # 1 1 H W D (Image Size)
                with autocast(dtype=torch.bfloat16):
                    if self.model_inferer is None:
                        logits = self.model(data)  # 1 C H W D (Image Size)
                    else:
                        logits = self.model_inferer(data)  # 1 C H W D (Image Size)

                    # Calculate validation metrics
                    val_loss = self.loss_func(logits, target)
                    val_acc, not_nans = self.calc_accuracy(y_pred=logits, y=target)

                # Update mean metrics
                total_epoch_acc = total_epoch_acc + val_acc * not_nans
                total_not_nans = total_not_nans + not_nans
                epoch_loss += val_loss.item()

                curr_epoch_str = str(self.epoch).rjust(len(str(self.max_epochs - 1)), "0")
                curr_index_str = str(index).rjust(len(str(len(self.val_loader) - 1)), "0")
                print(
                    f"Validate Batch {curr_epoch_str}/{self.max_epochs - 1} {curr_index_str}/{len(self.val_loader) - 1}  "
                    f"Acc: {val_acc.mean().item():.4f}  "
                    f"Not NaNs: {not_nans.sum().item()}  "
                    f"Time: {time.perf_counter() - start_time:.2f}s"
                )

                start_time = time.perf_counter()

        return epoch_loss / len(self.val_loader), total_epoch_acc / total_not_nans

    def calc_accuracy(self, y, y_pred):
        # Transform logits and target for accuracy function
        # e.g. Argmax, Softmax
        if self.post_label is not None:
            y = self.post_label(y)
        if self.post_pred is not None:
            y_pred = self.post_pred(y_pred)

        # Calculate Accuracy
        self.acc_func.reset()
        self.acc_func(y_pred=y_pred, y=y)
        acc, not_nan = self.acc_func.aggregate()

        return acc.cpu(), not_nan.cpu()

    def save_checkpoint(self, filename: str = "model.pt"):
        state_dict = self.model.state_dict()
        save_dict = {"epoch": self.epoch, "best_acc": self.best_val_acc, "state_dict": state_dict}
        if self.optimizer is not None:
            save_dict["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            save_dict["scheduler"] = self.scheduler.state_dict()

        filename = os.path.join(self.log_dir, filename)
        torch.save(save_dict, filename)
        print("Saving checkpoint", filename)

    def assert_finite(self, value, batch_num, msg=None):
        if torch.all(value.isfinite()):
            return True

        non_finite = value[torch.logical_not(value.isfinite())]

        self.logger.error(f"Epoch: {self.epoch}, Batch: {batch_num}, {msg} is {non_finite.tolist()}")
        return False

    def rearrange_axes(self, batch):
        batch["image"] = einops.rearrange(batch["image"], f"b c h w d -> b c {self.orient_axes}")
        batch["label"] = einops.rearrange(batch["label"], f"b c h w d -> b c {self.orient_axes}")

        return batch

import time
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from monai.data import DataLoader
from tensorboardX import SummaryWriter


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

    def __post_init__(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.scaler = None
        if self.grad_scale:
            print("Using Gradient Scaling")
            self.scaler = GradScaler()

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
            "val_every": self.val_every
        }

        for key, value in h_params.items():
            self.writer.add_text("Hyperparameters", f"{key}: {value}")

        if self.scheduler is not None:
            self.scheduler.last_epoch = self.start_epoch

        # Start training
        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"{time.ctime()} Epoch: {epoch} Learning Rate: {lr:.6f}")
            self.writer.add_scalar("learning_rate", lr, epoch)
            epoch_time = time.perf_counter()

            train_loss, train_acc = self.train_epoch()

            curr_epoch_str = str(epoch).rjust(len(str(self.max_epochs - 1)), "0")
            print(f"Train Epoch {curr_epoch_str}/{self.max_epochs - 1}  "
                  f"Mean Loss: {train_loss:.4f}  "
                  f"Time: {time.perf_counter() - epoch_time:.2f}s")
            self.writer.add_scalar("train_loss", train_loss, epoch)
            self.writer.add_scalar("train_acc", train_acc, epoch)

            if (epoch + 1) % self.val_every == 0:
                val_loss, val_acc = self.val_epoch()

                print(f"Validate Epoch {curr_epoch_str}/{self.max_epochs - 1}  "
                      f"Mean Acc: {val_acc:.4f}  "
                      f"Time: {time.perf_counter() - epoch_time:.2f}s")
                self.writer.add_scalar("val_acc", val_acc, epoch)
                self.writer.add_scalar("val_loss", val_loss, epoch)

                if val_acc > self.best_val_acc:
                    print(f"New highest accuracy {self.best_val_acc:.4f} --> {val_acc:.4f}")
                    self.best_val_acc = val_acc

                    self.save_checkpoint(filename="model_best.pt")

            if self.scheduler is not None:
                self.scheduler.step()

            self.save_checkpoint(filename="model_final.pt")

        print(f"Training Finished! Best Accuracy: {self.best_val_acc:.4f}")

    def train_epoch(self):
        self.model.train()
        start_time = time.perf_counter()

        epoch_loss = 0.0
        epoch_acc = 0.0
        for index, batch in enumerate(self.train_loader):
            if self.batch_augmentation is not None:
                batch = self.batch_augmentation(batch)

            data, target = batch["image"], batch["label"]
            data, target = data.cuda(0), target.cuda(0)

            with autocast():
                self.optimizer.zero_grad()

                logits = self.model(data)  # B*P 2 X Y Z
                loss = self.loss_func(logits, target)  # 0-dim tensor

                train_acc, train_not_nans = self.calc_accuracy(y_pred=logits, y=target)

                if self.grad_scale:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            mean_batch_loss = loss.item()
            epoch_loss += mean_batch_loss
            epoch_acc += train_acc

            curr_epoch_str = str(self.epoch).rjust(len(str(self.max_epochs - 1)), "0")
            curr_index_str = str(index).rjust(len(str(len(self.train_loader) - 1)), "0")
            print(
                f"Train Batch {curr_epoch_str}/{self.max_epochs - 1}  {curr_index_str}/{len(self.train_loader) - 1}  "
                f"Loss: {mean_batch_loss:.4f}  "
                f"Time: {time.perf_counter() - start_time:.2f}s"
            )

            start_time = time.perf_counter()
        return epoch_loss / len(self.train_loader), epoch_acc / len(self.train_loader)

    def val_epoch(self):
        self.model.eval()
        start_time = time.perf_counter()

        epoch_acc = 0.0
        epoch_loss = 0.0
        with torch.no_grad():
            for index, batch in enumerate(self.val_loader):
                data, target = batch["image"], batch["label"]
                data, target = data.cuda(0), target.cuda(0)  # 1 1 H W D
                with autocast():
                    if self.model_inferer is None:
                        logits = self.model(data)
                    else:
                        logits = self.model_inferer(data)  # 1 C H W D
                    val_loss = self.loss_func(logits, target)
                    val_acc, not_nans = self.calc_accuracy(y_pred=logits, y=target)

                epoch_acc += val_acc
                epoch_loss += val_loss.item()

                curr_epoch_str = str(self.epoch).rjust(len(str(self.max_epochs - 1)), "0")
                curr_index_str = str(index).rjust(len(str(len(self.val_loader) - 1)), "0")
                print(
                    f"Validate Batch {curr_epoch_str}/{self.max_epochs - 1} {curr_index_str}/{len(self.val_loader) - 1}  "
                    f"Acc: {val_acc:.4f}  "
                    f"Not NaNs: {not_nans}  "
                    f"Time: {time.perf_counter() - start_time:.2f}s"
                )

                start_time = time.perf_counter()

        return epoch_loss / len(self.val_loader), epoch_acc / len(self.val_loader)

    def calc_accuracy(self, y, y_pred):
        if self.post_label is not None:
            y = self.post_label(y)
        if self.post_pred is not None:
            y_pred = self.post_pred(y_pred)

        self.acc_func.reset()
        self.acc_func(y_pred=y_pred, y=y)
        acc, val_not_nan = self.acc_func.aggregate()

        return acc.item(), int(val_not_nan.item())

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

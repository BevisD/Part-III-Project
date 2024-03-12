import time
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from monai.data import DataLoader, decollate_batch
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
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    best_val_acc = 0.0
    epoch = start_epoch

    def train(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            print(time.ctime(), "Epoch:", epoch)
            epoch_time = time.perf_counter()

            train_loss, train_acc = self.train_epoch()

            print(f"Train Epoch {epoch}/{self.max_epochs - 1}\t"
                  f"Mean Loss: {train_loss:.4f}\t"
                  f"Time: {time.perf_counter() - epoch_time:.2f}s")
            self.writer.add_scalar("train_loss", train_loss, epoch)
            self.writer.add_scalar("train_acc", train_acc, epoch)

            if (epoch + 1) % self.val_every == 0:
                val_loss, val_acc = self.val_epoch()

                print(f"Validate Epoch {epoch}/{self.max_epochs - 1}\t"
                      f"Mean Acc: {val_acc:.4f}\t"
                      f"Time: {time.perf_counter() - epoch_time:.2f}s")
                self.writer.add_scalar("val_acc", val_acc, epoch)
                self.writer.add_scalar("val_loss", val_loss, epoch)

                if val_acc > self.best_val_acc:
                    print(f"New highest accuracy {self.best_val_acc:.4f} --> {val_acc:.4f}")
                    self.best_val_acc = val_acc

                    self.save_checkpoint(filename="model_best.pt")

            self.scheduler.step()
            self.save_checkpoint(filename="model_final.pt")

        print(f"Training Finished! Best Accuracy: {self.best_val_acc:.4f}")

    def train_epoch(self):
        self.model.train()
        start_time = time.perf_counter()

        epoch_loss = 0.0
        epoch_acc = 0.0
        for index, batch in enumerate(self.train_loader):
            data, target = batch["image"], batch["label"]
            data, target = data.cuda(0), target.cuda(0)

            with autocast():
                self.optimizer.zero_grad()

                logits = self.model(data)  # B*P 2 X Y Z
                loss = self.loss_func(logits, target)  # 0-dim tensor

                train_acc, train_not_nans = self.calc_accuracy(target, logits)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            mean_batch_loss = loss.item()
            epoch_loss += mean_batch_loss
            epoch_acc += train_acc

            print(
                f"Train Batch {self.epoch}/{self.max_epochs - 1}\t{index}/{len(self.train_loader)}\t"
                f"Loss: {mean_batch_loss:.4f}\t"
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
                data, target = data.cuda(0), target.cuda(0)  # B*P 1 H W D

                with autocast():
                    if self.model_inferer is None:
                        logits = self.model(data)
                    else:
                        logits = self.model_inferer(data)  # B*P C H W D

                val_acc, not_nans = self.calc_accuracy(target, logits)
                val_loss = self.loss_func(logits, target)

                epoch_acc += val_acc
                epoch_loss += val_loss
                print(
                    f"Validate Batch {self.epoch}/{self.max_epochs} {index}/{len(self.val_loader)}\t"
                    f"Acc: {val_acc:.4f}\t"
                    f"Not NaNs: {not_nans}\t"
                    f"Time: {time.perf_counter() - start_time:.2f}s"
                )

                start_time = time.perf_counter()

        return epoch_loss / len(self.val_loader), epoch_acc / len(self.val_loader)

    def calc_accuracy(self, target, logits):
        labels_list = decollate_batch(target)  # [C H W D]
        outputs_list = decollate_batch(logits)  # [C H W D]

        labels_converted = [self.post_label(label_tensor) for label_tensor in labels_list]
        output_converted = [self.post_pred(pred_tensor) for pred_tensor in outputs_list]

        self.acc_func.reset()
        self.acc_func(y_pred=output_converted, y=labels_converted)
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

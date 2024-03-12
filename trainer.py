import time
import os

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from monai.data import DataLoader, decollate_batch
from tensorboardX import SummaryWriter


def train_epoch(model: nn.Module,
                loader: DataLoader,
                loss_func,
                acc_func,
                post_label,
                post_pred,
                optimizer,
                scaler,
                epoch,
                max_epochs) -> tuple[float, float]:
    model.train()
    start_time = time.perf_counter()

    epoch_loss = 0.0
    epoch_acc = 0.0
    for index, batch in enumerate(loader):
        data, target = batch["image"], batch["label"]
        data, target = data.cuda(0), target.cuda(0)

        with autocast():
            optimizer.zero_grad()

            logits = model(data)  # B*P 2 X Y Z
            loss = loss_func(logits, target)  # 0-dim tensor

            train_acc, train_not_nans = calc_accuracy(target, logits, acc_func, post_pred, post_label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        mean_batch_loss = loss.item()
        epoch_loss += mean_batch_loss
        epoch_acc += train_acc

        print(
            f"Train Batch {epoch}/{max_epochs - 1}\t{index}/{len(loader)}\t"
            f"Loss: {mean_batch_loss:.4f}\t"
            f"Time: {time.perf_counter() - start_time:.2f}s"
        )

        start_time = time.perf_counter()
    return epoch_loss / len(loader), epoch_acc / len(loader)


def val_epoch(model: nn.Module,
              loader,
              loss_func,
              acc_func,
              post_label,
              post_pred,
              epoch,
              max_epochs,
              model_inferer=None) -> tuple[float, float]:
    model.eval()
    start_time = time.perf_counter()

    epoch_acc = 0.0
    epoch_loss = 0.0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            data, target = batch["image"], batch["label"]
            data, target = data.cuda(0), target.cuda(0)  # B*P 1 H W D

            with autocast():
                if model_inferer is None:
                    logits = model(data)
                else:
                    logits = model_inferer(data)  # B*P C H W D

            val_acc, not_nans = calc_accuracy(target, logits, acc_func, post_pred, post_label)
            val_loss = loss_func(logits, target)

            epoch_acc += val_acc
            epoch_loss += val_loss
            print(
                f"Validate Batch {epoch}/{max_epochs} {index}/{len(loader)}\t"
                f"Acc: {val_acc:.4f}\t"
                f"Not NaNs: {not_nans}\t"
                f"Time: {time.perf_counter() - start_time:.2f}s"
            )

            start_time = time.perf_counter()

    return epoch_loss / len(loader), epoch_acc / len(loader)


def calc_accuracy(target, logits, acc_func, post_pred, post_label):
    labels_list = decollate_batch(target)  # [C H W D]
    outputs_list = decollate_batch(logits)  # [C H W D]

    labels_converted = [post_label(label_tensor) for label_tensor in labels_list]
    output_converted = [post_pred(pred_tensor) for pred_tensor in outputs_list]

    acc_func.reset()
    acc_func(y_pred=output_converted, y=labels_converted)
    acc, val_not_nan = acc_func.aggregate()

    return acc.item(), int(val_not_nan.item())


def save_checkpoint(model: nn.Module,
                    epoch: int,
                    filename: str = "model.pt",
                    best_acc: float = 0,
                    logdir: str = "runs/test",
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()

    filename = os.path.join(logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_func,
                 acc_func,
                 scheduler,
                 log_dir,
                 max_epochs: int,
                 val_every: int,
                 model_inferer,
                 start_epoch: int = 0,
                 post_label=None,
                 post_pred=None,
                 ) -> None:
    best_test_acc = 0.0
    scaler = GradScaler()

    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.perf_counter()

        train_loss, train_acc = train_epoch(
            model=model,
            loader=train_loader,
            loss_func=loss_func,
            post_label=post_label,
            post_pred=post_pred,
            acc_func=acc_func,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            max_epochs=max_epochs
        )
        print(f"Train Epoch {epoch}/{max_epochs - 1}\t"
              f"Mean Loss: {train_loss:.4f}\t"
              f"Time: {time.perf_counter() - epoch_time:.2f}s")
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)

        if (epoch + 1) % val_every == 0:
            val_loss, val_acc = val_epoch(
                model=model,
                loader=val_loader,
                model_inferer=model_inferer,
                loss_func=loss_func,
                acc_func=acc_func,
                post_label=post_label,
                post_pred=post_pred,
                epoch=epoch,
                max_epochs=max_epochs
            )

            print(f"Validate Epoch {epoch}/{max_epochs - 1}\t"
                  f"Mean Acc: {val_acc:.4f}\t"
                  f"Time: {time.perf_counter() - epoch_time:.2f}s")
            writer.add_scalar("val_acc", val_acc, epoch)
            writer.add_scalar("val_loss", val_loss, epoch)

            if val_acc > best_test_acc:
                print(f"New highest accuracy {best_test_acc:.4f} --> {val_acc:.4f}")
                best_test_acc = val_acc

                save_checkpoint(model, epoch,
                                best_acc=best_test_acc,
                                filename="model_best.pt")

        scheduler.step()
        save_checkpoint(model, epoch,
                        best_acc=best_test_acc,
                        filename="model_final.pt")

    print(f"Training Finished! Best Accuracy: {best_test_acc:.4f}")

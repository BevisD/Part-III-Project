import time
import os

import monai.inferers
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from monai.data import DataLoader, decollate_batch


def train_epoch(model: nn.Module,
                loader: DataLoader,
                loss_func,
                optimizer,
                scaler,
                batch_size,
                epoch,
                max_epochs) -> float:
    model.train()
    start_time = time.time()

    epoch_loss = 0.0
    for index, batch in enumerate(loader):
        data, target = batch
        data, target = data.cuda(), target.cuda()

        with autocast():
            optimizer.zero_grad()

            logits = model(data)
            loss = loss_func(logits, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        mean_batch_loss = loss.item() / batch_size
        epoch_loss += loss.item()
        print(
            f"Epoch {epoch}/{max_epochs}\t{index}/{len(loader)}\t"
            f"Loss: {mean_batch_loss:.4f}\t"
            f"Time: {time.time() - start_time:.2f}"
        )

        start_time = time.time()
    return epoch_loss / len(loader)


def test_epoch(model: nn.Module,
               loader,
               model_inferer,
               acc_func,
               post_label,
               post_pred,
               epoch,
               max_epochs) -> float:
    model.eval()
    start_time = time.time()

    epoch_acc = 0.0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            data, target = batch
            data, target = data.cuda(), target.cuda()

            with autocast:
                logits = model_inferer(data)

            test_labels_list = decollate_batch(target)
            test_outputs_list = decollate_batch(logits)

            val_labels_converted = [post_label(test_label_tensor) for test_label_tensor in test_labels_list]
            val_output_converted = [post_pred(test_pred_tensor) for test_pred_tensor in test_outputs_list]

            acc_func.reset()
            acc_func(y_pred=val_output_converted, y=val_labels_converted)
            acc, not_nans = acc_func.aggregate()

            mean_batch_acc = acc / not_nans
            epoch_acc += mean_batch_acc

            print(
                f"Epoch {epoch}/{max_epochs}\t{index}/{len(loader)}\t"
                f"Acc: {mean_batch_acc:.4f}\t"
                f"Time: {time.time() - start_time:.2f}"
            )

            start_time = time.time()

    return epoch_acc / len(loader)


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
                 test_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_func,
                 acc_func,
                 scheduler,
                 max_epochs: int,
                 batch_size: int,
                 val_every: int,
                 model_inferer,
                 start_epoch: int = 0,
                 post_label=None,
                 post_pred=None
                 ) -> None:

    best_test_acc = 0.0
    scaler = GradScaler()
    for epoch in range(start_epoch, max_epochs):
        train_epoch(model=model,
                    loader=train_loader,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    scaler=scaler,
                    batch_size=batch_size,
                    epoch=epoch,
                    max_epochs=max_epochs)

        if (epoch + 1) % val_every == 0:
            test_acc = test_epoch(model=model,
                                  loader=test_loader,
                                  model_inferer=model_inferer,
                                  acc_func=acc_func,
                                  post_label=post_label,
                                  post_pred=post_pred,
                                  epoch=epoch,
                                  max_epochs=max_epochs)

            if test_acc > best_test_acc:
                print(f"BEST ACCURACY {best_test_acc:.4f} --> {test_acc:.4f}")
                best_test_acc = test_acc

                save_checkpoint(model, epoch,
                                best_acc=best_test_acc,
                                filename="model_best.pt")

    save_checkpoint(model,
                    epoch=max_epochs-1,
                    best_acc=best_test_acc,
                    filename="model_final.pt")
    print(f"Training Finished! Best Accuracy: {best_test_acc:.4f}")

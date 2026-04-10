"""Training loops for HeatNet models."""

from __future__ import annotations

import torch
from tqdm import tqdm


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    save_path,
    scheduler=None,
    uses_depth=False,
):
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            training=True,
            uses_depth=uses_depth,
            scheduler=scheduler,
        )
        val_loss = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            epoch=epoch,
            training=False,
            uses_depth=uses_depth,
            scheduler=None,
        )

        if scheduler is not None and not getattr(scheduler, "step_per_batch", False):
            scheduler.step()

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} (val_loss={val_loss:.4f})")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            }
        )

    return history


def build_scheduler(name, optimizer, epochs, train_loader, base_lr=1e-4, max_lr=5e-4):
    name = (name or "").lower()
    if not name or name == "none":
        return None
    if name == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR

        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )
        scheduler.step_per_batch = True
        return scheduler
    if name == "polynomial":
        from torch.optim.lr_scheduler import PolynomialLR

        scheduler = PolynomialLR(optimizer, total_iters=epochs, power=1.0)
        scheduler.step_per_batch = False
        return scheduler
    raise ValueError(f"Unsupported scheduler: {name}")


def _run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
    training,
    uses_depth,
    scheduler=None,
):
    mode = "Train" if training else "Val"
    progress = tqdm(loader, desc=f"Epoch {epoch} [{mode}]", leave=False)

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    for batch in progress:
        if uses_depth:
            images, depth, heatmaps_gt = batch
            images = images.to(device)
            depth = depth.to(device)
            heatmaps_gt = heatmaps_gt.to(device)
        else:
            images, heatmaps_gt = batch
            images = images.to(device)
            heatmaps_gt = heatmaps_gt.to(device)
            depth = None

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            if uses_depth:
                heatmaps_pred = model(images, depth)
            else:
                heatmaps_pred = model(images)
            loss = criterion(heatmaps_pred, heatmaps_gt)

            if training:
                loss.backward()
                optimizer.step()
                if scheduler is not None and getattr(scheduler, "step_per_batch", False):
                    scheduler.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        progress.set_postfix(loss=loss.item())

    return total_loss / len(loader.dataset)

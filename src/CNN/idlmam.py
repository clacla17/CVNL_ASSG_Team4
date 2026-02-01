import time
from typing import Dict, Callable, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.autonotebook import tqdm


def moveTo(obj, device):
    if hasattr(obj, "to"):
        return obj.to(device)
    if isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    if isinstance(obj, set):
        return set(moveTo(list(obj), device))
    if isinstance(obj, dict):
        to_ret = {}
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    return obj


def run_epoch(
    model,
    optimizer,
    data_loader,
    loss_func,
    device,
    results,
    score_funcs,
    prefix="",
    desc=None,
):
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()

    for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        y_hat = model(inputs)
        loss = loss_func(y_hat, labels)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())

    end = time.time()

    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    results[prefix + " loss"].append(np.mean(running_loss))
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append(score_func(y_true, y_pred))
        except Exception:
            results[prefix + " " + name].append(float("NaN"))

    return end - start


def train_network(
    model,
    loss_func,
    train_loader,
    val_loader=None,
    test_loader=None,
    score_funcs: Optional[Dict[str, Callable]] = None,
    epochs=50,
    device="cpu",
    checkpoint_file=None,
    lr_schedule=None,
    optimizer=None,
    disable_tqdm=False,
):
    if score_funcs is None:
        score_funcs = {}

    to_track = ["epoch", "total time", "train loss"]
    if val_loader is not None:
        to_track.append("val loss")
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score)
        if val_loader is not None:
            to_track.append("val " + eval_score)
        if test_loader is not None:
            to_track.append("test " + eval_score)

    total_train_time = 0
    results = {item: [] for item in to_track}

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters())
        del_opt = True
    else:
        del_opt = False

    model.to(device)

    for epoch in tqdm(range(epochs), desc="Epoch", disable=disable_tqdm):
        model = model.train()
        total_train_time += run_epoch(
            model,
            optimizer,
            train_loader,
            loss_func,
            device,
            results,
            score_funcs,
            prefix="train",
            desc="Training",
        )

        results["epoch"].append(epoch)
        results["total time"].append(total_train_time)

        if val_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(
                    model,
                    optimizer,
                    val_loader,
                    loss_func,
                    device,
                    results,
                    score_funcs,
                    prefix="val",
                    desc="Validating",
                )

        if lr_schedule is not None:
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val loss"][-1])
            else:
                lr_schedule.step()

        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(
                    model,
                    optimizer,
                    test_loader,
                    loss_func,
                    device,
                    results,
                    score_funcs,
                    prefix="test",
                    desc="Testing",
                )

        if checkpoint_file is not None:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "results": results,
                },
                checkpoint_file,
            )

    if del_opt:
        del optimizer

    return pd.DataFrame.from_dict(results)

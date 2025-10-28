from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from DiffVNet import DiffVNet
from data import DatasetAllTasks, get_StrongAug
from utils import (
    GaussianSmoothing,
    fetch_data,
    poly_lr,
    seed_worker,
    sigmoid_rampup,
)
from utils.config import Config
from utils.loss import DC_and_CE_loss


# -----------------------------------------------------------------------------
# Hyper-parameters for the example.
# -----------------------------------------------------------------------------
MAX_EPOCHS = 50
CONSISTENCY_WEIGHT = 1.0
CONSISTENCY_RAMPUP_EPOCHS = 30
BASE_LR = 1e-3
PSEUDO_TEMP = 1.0
PSEUDO_MIN_PROB = 1e-6
MIXUP_ALPHA = 0.5
MIXUP_PROB = 0.5
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5


def _set_logging(snapshot_path: Path) -> None:
    snapshot_path.mkdir(parents=True, exist_ok=True)
    log_file = snapshot_path / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _fuse_probabilities(p_u_xi: torch.Tensor, p_u_psi: torch.Tensor, p_u_fake: torch.Tensor) -> torch.Tensor:
    """Fuse three probability maps using entropy-based weights."""

    u_xi = -torch.sum(p_u_xi * torch.log(p_u_xi.clamp_min(1e-12)), dim=1) / p_u_xi.size(1)
    u_psi = -torch.sum(p_u_psi * torch.log(p_u_psi.clamp_min(1e-12)), dim=1) / p_u_psi.size(1)
    u_fake = -torch.sum(p_u_fake * torch.log(p_u_fake.clamp_min(1e-12)), dim=1) / p_u_fake.size(1)

    entropy_stack = torch.stack([u_xi, u_psi, u_fake], dim=1)
    weight_logits = -entropy_stack / max(PSEUDO_TEMP, 1e-6)
    fusion_weights = torch.softmax(weight_logits, dim=1)

    prob_stack = torch.stack([p_u_xi, p_u_psi, p_u_fake], dim=1)
    prob_fused = torch.sum(fusion_weights.unsqueeze(2) * prob_stack, dim=1)
    prob_fused = torch.clamp(prob_fused, min=PSEUDO_MIN_PROB)
    prob_fused = prob_fused / torch.clamp(prob_fused.sum(dim=1, keepdim=True), min=1e-12)

    return prob_fused


def _consistency_weight(epoch: int) -> float:
    ramp = sigmoid_rampup(epoch, CONSISTENCY_RAMPUP_EPOCHS)
    return CONSISTENCY_WEIGHT * ramp


def train_example(task: str = "mmwhs_mr2ct") -> None:
    """Run a pared-down semi-supervised training loop for DiffVNet."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DiffVNet example requires a CUDA-enabled device.")

    config = Config(task)
    snapshot_path = Path("./logs/example_run")
    _set_logging(snapshot_path)
    logging.info("Starting DiffVNet example training for task '%s'", task)

    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark = True

    model = DiffVNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization="batchnorm",
        has_dropout=True,
        dropout_rate=0.5,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=BASE_LR,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True,
    )

    supervised_loss = DC_and_CE_loss()
    unsupervised_loss = DC_and_CE_loss()
    smoothing = GaussianSmoothing(config.num_cls, kernel_size=3, sigma=1).to(device)

    augment = get_StrongAug(
        patch_size=config.patch_size,
        sample_num=3,
        p_per_sample=0.7,
        mixup_alpha=MIXUP_ALPHA,
        mixup_prob=MIXUP_PROB,
        cutmix_alpha=CUTMIX_ALPHA,
        cutmix_prob=CUTMIX_PROB,
    )

    labeled_dataset = DatasetAllTasks(
        split="train_labeled",
        num_cls=config.num_cls,
        task=task,
        transform=augment,
        unlabeled=False,
    )

    unlabeled_dataset = DatasetAllTasks(
        split="train_unlabeled",
        num_cls=config.num_cls,
        task=task,
        transform=augment,
        unlabeled=True,
    )

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        lr = poly_lr(epoch - 1, MAX_EPOCHS, BASE_LR, 0.9)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        consistency_w = _consistency_weight(epoch - 1)
        epoch_sup_loss = 0.0
        epoch_unsup_loss = 0.0
        steps = 0

        loader_iter = zip(labeled_loader, unlabeled_loader)
        for batch_l, batch_u in tqdm(loader_iter, desc=f"Epoch {epoch}/{MAX_EPOCHS}"):
            image_l, label_l = fetch_data(batch_l, labeled=True)
            image_u = fetch_data(batch_u, labeled=False)

            optimizer.zero_grad(set_to_none=True)

            logits_sup = model(image=image_l, pred_type="D_theta_u")
            loss_sup = supervised_loss(logits_sup, label_l.long())

            with torch.no_grad():
                p_u_xi = smoothing(F.gumbel_softmax(model(image=image_u, pred_type="ddim_sample"), dim=1))
                p_u_psi = F.softmax(model(image=image_u, pred_type="D_psi_l"), dim=1)
                p_u_fake = F.softmax(model(image=image_u, pred_type="fake"), dim=1)
                prob_fused = _fuse_probabilities(p_u_xi, p_u_psi, p_u_fake)
                pseudo_label = torch.argmax(prob_fused, dim=1, keepdim=True).long()

            logits_unsup = model(image=image_u, pred_type="D_theta_u")
            loss_unsup = unsupervised_loss(logits_unsup, pseudo_label.detach())

            loss = loss_sup + consistency_w * loss_unsup
            loss.backward()
            optimizer.step()

            epoch_sup_loss += loss_sup.item()
            epoch_unsup_loss += loss_unsup.item()
            steps += 1

        avg_sup = epoch_sup_loss / max(steps, 1)
        avg_unsup = epoch_unsup_loss / max(steps, 1)
        logging.info(
            "Epoch %03d | lr %.5f | mu %.3f | sup %.4f | unsup %.4f",
            epoch,
            lr,
            consistency_w,
            avg_sup,
            avg_unsup,
        )

        torch.save(
            {"epoch": epoch, "state_dict": model.state_dict()},
            snapshot_path / "latest.ckpt",
        )


if __name__ == "__main__":
    train_example()

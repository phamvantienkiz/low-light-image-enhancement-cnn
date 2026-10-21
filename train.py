"""
Main Training Pipeline for Low-Light Image Enhancement.

Supports:
- Model architectures: 'attn_unet' (default), 'compact_unet', 'plain_cnn'
- Dataset splits: 'initial_500' or 'extended' with quarantine slices (+50 each)
- Loss: Composite L1 + SSIM loss
- Metrics: PSNR and SSIM tracked per epoch
- Checkpoint management: 'best_model.pth' and 'last_model.pth'
- Seamless execution on Google Colab (GPU) and local machine (CPU verification)
"""

import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models import PlainCNN, CompactUNet, LightweightAttnUNet
from src.datasets import load_dataset_from_manifest
from src.losses import CompositeLoss
from src.metrics import calculate_psnr, calculate_ssim
from src.utils import set_seed, count_parameters, save_checkpoint, save_image_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLIE Lightweight CNN")
    parser.add_argument(
        "--model",
        type=str,
        default="attn_unet",
        choices=["attn_unet", "compact_unet", "plain_cnn"],
        help="Model architecture",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="docs/ai/split_manifest.json",
        help="Path to split_manifest.json",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="Root path containing datasets/",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="initial_250",
        choices=["initial_250", "initial_500", "extended"],
        help="Training mode: 'initial_250' (250 pairs default) or 'extended' (adds quarantine slices)",
    )
    parser.add_argument(
        "--quarantine_slices",
        nargs="*",
        default=[],
        help="Quarantine slices to add in extended mode, e.g. slice_1 slice_2",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=256, help="Training patch crop size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--l1_weight", type=float, default=1.0, help="Weight for L1 pixel loss")
    parser.add_argument("--ssim_weight", type=float, default=0.2, help="Weight for SSIM loss")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (0 for safe Windows/Colab execution)",
    )
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'")
    return parser.parse_args()


def build_model(model_name: str) -> torch.nn.Module:
    if model_name == "attn_unet":
        return LightweightAttnUNet(in_channels=3, out_channels=3, base_ch=16, use_residual=True)
    elif model_name == "compact_unet":
        return CompactUNet(in_channels=3, out_channels=3, base_ch=16)
    elif model_name == "plain_cnn":
        return PlainCNN(in_channels=3, hidden_channels=64, out_channels=3)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def evaluate(
    model: torch.nn.Module, val_loader: DataLoader, device: torch.device
) -> tuple[float, float, float]:
    model.eval()
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    l1_fn = torch.nn.L1Loss()

    with torch.no_grad():
        for low, high, _ in val_loader:
            low = low.to(device)
            high = high.to(device)
            enhanced = model(low)

            bs = low.size(0)
            total_l1 += l1_fn(enhanced, high).item() * bs
            total_psnr += calculate_psnr(enhanced, high) * bs
            total_ssim += calculate_ssim(enhanced, high) * bs
            num_samples += bs

    avg_l1 = total_l1 / max(1, num_samples)
    avg_psnr = total_psnr / max(1, num_samples)
    avg_ssim = total_ssim / max(1, num_samples)
    return avg_l1, avg_psnr, avg_ssim


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print(" " * 15 + "LOW-LIGHT IMAGE ENHANCEMENT TRAINER")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Train Mode: {args.train_mode} (Slices: {args.quarantine_slices})")
    print(f"Hyperparameters: LR={args.lr}, BatchSize={args.batch_size}, PatchSize={args.patch_size}, Epochs={args.epochs}")

    # 1. Build Model
    model = build_model(args.model).to(device)
    param_count = count_parameters(model)
    print(f"Model Parameters: {param_count:,} ({param_count / 1e6:.3f}M)")

    # 2. Prepare DataLoaders
    manifest_path = Path(args.manifest)
    data_root = Path(args.data_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    train_ds, val_ds = load_dataset_from_manifest(
        manifest_path=manifest_path,
        data_root=data_root,
        mode=args.train_mode,
        quarantine_slices=args.quarantine_slices,
        patch_size=args.patch_size,
    )
    print(f"Dataset Loaded: Train Pairs = {len(train_ds)}, Validation Pairs = {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Setup Optimizer, Scheduler, and Loss
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CompositeLoss(l1_weight=args.l1_weight, ssim_weight=args.ssim_weight).to(device)

    save_dir = Path(args.save_dir) / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    best_psnr = -1.0
    history = []

    print("-" * 70)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val L1':<10} | {'Val PSNR (dB)':<14} | {'Val SSIM':<10} | {'Time (s)':<8}")
    print("-" * 70)

    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        total_batches = 0

        for low, high, _ in train_loader:
            low = low.to(device)
            high = high.to(device)

            optimizer.zero_grad()
            pred = model(low)
            loss, _ = criterion(pred, high)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        scheduler.step()
        train_loss = running_loss / max(1, total_batches)

        # Validation every epoch
        val_l1, val_psnr, val_ssim = evaluate(model, val_loader, device)
        epoch_duration = time.time() - epoch_start

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "best_psnr": best_psnr,
            "args": vars(args),
        }
        save_checkpoint(checkpoint, is_best=is_best, save_dir=save_dir)

        # Log entry
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_l1": round(val_l1, 5),
            "val_psnr": round(val_psnr, 2),
            "val_ssim": round(val_ssim, 4),
            "lr": round(scheduler.get_last_lr()[0], 7),
            "epoch_time_sec": round(epoch_duration, 1),
        }
        history.append(entry)

        print(
            f"{epoch:<8} | {train_loss:<12.5f} | {val_l1:<10.5f} | "
            f"{val_psnr:<14.2f} | {val_ssim:<10.4f} | {epoch_duration:<8.1f}"
            + (" *" if is_best else "")
        )

    total_training_time = time.time() - start_train_time
    print("=" * 70)
    print(f"Training Complete in {total_training_time / 60:.1f} minutes.")
    print(f"Best Validation PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints and logs saved at: {save_dir}")

    # Save log history to JSON
    with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({"best_psnr": best_psnr, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()

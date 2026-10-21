"""
Evaluation and Inference Script for Low-Light Image Enhancement.

Evaluates trained checkpoints or traditional baselines on:
1. Fixed Validation Set (200 pairs from split manifest)
2. Benchmark LOLv1 eval15 (15 pairs)
3. Benchmark LOL-v2 Real Test (100 pairs)
4. ExDark (unpaired real-world night photography - saves visual outputs)
"""

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

from src.models import PlainCNN, CompactUNet, LightweightAttnUNet
from src.baselines import apply_gamma_correction, apply_clahe
from src.datasets.paired_lol import PairedLowLightDataset
from src.metrics import calculate_psnr, calculate_ssim
from src.utils import save_image_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLIE Models")
    parser.add_argument(
        "--method",
        type=str,
        default="attn_unet",
        choices=["attn_unet", "compact_unet", "plain_cnn", "gamma", "clahe"],
        help="Model or baseline method",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument(
        "--manifest",
        type=str,
        default="docs/ai/split_manifest.json",
        help="Path to split_manifest.json",
    )
    parser.add_argument("--data_root", type=str, default=".", help="Root path containing datasets/")
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "lolv1_eval15", "lolv2_test", "exdark"],
        help="Dataset split to evaluate",
    )
    parser.add_argument("--save_visuals", action="store_true", help="Save enhanced output images")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save visual outputs")
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'")
    return parser.parse_args()


def load_model(method: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    if method == "attn_unet":
        model = LightweightAttnUNet(in_channels=3, out_channels=3, base_ch=16, use_residual=True)
    elif method == "compact_unet":
        model = CompactUNet(in_channels=3, out_channels=3, base_ch=16)
    elif method == "plain_cnn":
        model = PlainCNN(in_channels=3, hidden_channels=64, out_channels=3)
    else:
        raise ValueError(f"Deep learning checkpoint not required for traditional method: {method}")

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint weights from {checkpoint_path}")
    else:
        print(f"Warning: Running {method} with uninitialized weights (or missing checkpoint).")

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    output_dir = Path(args.output_dir) / f"{args.method}_{args.split}"
    if args.save_visuals:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Handle Unpaired ExDark evaluation
    if args.split == "exdark":
        exdark_dir = data_root / manifest["external_benchmarks"]["exdark_dir"]
        exdark_files = list(exdark_dir.glob("**/*.jpg")) + list(exdark_dir.glob("**/*.png"))
        print(f"Evaluating on ExDark: Found {len(exdark_files)} images.")

        if args.method in ["attn_unet", "compact_unet", "plain_cnn"]:
            model = load_model(args.method, args.checkpoint, device)

        for img_path in exdark_files[:50]:  # Evaluate on first 50 for demo
            img = Image.open(img_path).convert("RGB")
            t_img = TF.to_tensor(img).unsqueeze(0).to(device)

            if args.method == "gamma":
                enhanced = apply_gamma_correction(t_img, gamma=0.4)
            elif args.method == "clahe":
                arr = t_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                enhanced_arr = apply_clahe(arr)
                enhanced = torch.from_numpy(enhanced_arr).permute(2, 0, 1).unsqueeze(0).to(device)
            else:
                with torch.no_grad():
                    enhanced = model(t_img)

            if args.save_visuals:
                comparison = torch.cat([t_img, enhanced], dim=3)
                save_image_tensor(comparison.squeeze(0), output_dir / f"{img_path.stem}_enhanced.png")

        print(f"ExDark visual inference completed. Outputs saved to {output_dir}")
        return

    # 2. Handle Paired Evaluation (Validation, LOLv1 eval15, LOLv2 Test)
    if args.split == "validation":
        pairs = manifest["validation_set"]
    elif args.split == "lolv1_eval15":
        pairs = manifest["external_benchmarks"]["lolv1_eval15"]
    elif args.split == "lolv2_test":
        pairs = manifest["external_benchmarks"]["lolv2_real_test"]
    else:
        raise ValueError(f"Unknown split: {args.split}")

    dataset = PairedLowLightDataset(data_root=data_root, pairs=pairs, patch_size=None, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Evaluating {args.method} on {args.split} ({len(dataset)} pairs)...")

    if args.method in ["attn_unet", "compact_unet", "plain_cnn"]:
        model = load_model(args.method, args.checkpoint, device)

    total_psnr = 0.0
    total_ssim = 0.0

    for idx, (low, high, item_id) in enumerate(loader):
        low = low.to(device)
        high = high.to(device)

        if args.method == "gamma":
            enhanced = apply_gamma_correction(low, gamma=0.4)
        elif args.method == "clahe":
            arr = low.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced_arr = apply_clahe(arr)
            enhanced = torch.from_numpy(enhanced_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            with torch.no_grad():
                enhanced = model(low)

        psnr_val = calculate_psnr(enhanced, high)
        ssim_val = calculate_ssim(enhanced, high)
        total_psnr += psnr_val
        total_ssim += ssim_val

        if args.save_visuals and idx < 20:
            comparison = torch.cat([low, enhanced, high], dim=3)
            save_image_tensor(comparison.squeeze(0), output_dir / f"{item_id[0]}_comparison.png")

    avg_psnr = total_psnr / len(dataset)
    avg_ssim = total_ssim / len(dataset)

    print("=" * 60)
    print(f"Evaluation Results for [{args.method}] on [{args.split}]:")
    print(f"  * Average PSNR : {avg_psnr:.2f} dB")
    print(f"  * Average SSIM : {avg_ssim:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

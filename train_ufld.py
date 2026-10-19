"""Fine-tune UFLD (CULane ResNet18) on auto-labeled MetaDrive frames.

The ONNX model is converted to a trainable PyTorch module (onnx2torch),
fine-tuned with UFLD's row-anchor classification loss plus an L1 term on the
expected grid position (what ufld.UFLD actually decodes), evaluated against
the original weights on held-out maps, and exported back to ONNX so the
existing inference path (ufld.UFLD, CoreML) runs it unchanged.

Usage (MetaDrive venv, which has torch):
    .venv-md/bin/python train_ufld.py --data data/md_lanes --epochs 6 --out models/culane_18_md.onnx
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx2torch import convert

from ufld import GRID, IN_H, IN_W, MEAN, MODEL_PATH, ROW_ANCHORS, STD

COL_W = (IN_W - 1) / (GRID - 1)


def cell_to_x(c):
    return (c + 1) * COL_W - 1


def x_to_cell(x):
    return np.clip(np.round((x + 1) / COL_W - 1), 0, GRID - 1).astype(np.int64)


class LaneSet(torch.utils.data.Dataset):
    def __init__(self, root, idx, augment):
        self.root, self.idx, self.augment = Path(root), idx, augment
        self.labels = np.load(self.root / "labels.npz")["labels"]
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]
        img = cv2.imread(str(self.root / "img" / f"{k:05d}.jpg"))
        lab = self.labels[k].copy()
        if self.augment:
            if self.rng.random() < 0.5:   # mirror: x -> W-1-x and the lane slots swap sides
                img = img[:, ::-1]
                vis = lab < GRID
                mirrored = np.where(vis, x_to_cell(IN_W - 1 - cell_to_x(lab)), GRID)
                lab = mirrored[:, ::-1]
            alpha = self.rng.uniform(0.7, 1.3)    # contrast
            beta = self.rng.uniform(-30, 30)      # brightness
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255)
        img = cv2.resize(np.ascontiguousarray(img).astype(np.uint8), (IN_W, IN_H))
        x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(np.ascontiguousarray(lab))


def loss_fn(out, lab):
    ce = F.cross_entropy(out, lab)
    prob = out[:, :GRID].softmax(1)
    loc = (prob * torch.arange(GRID, device=out.device, dtype=out.dtype)[None, :, None, None]).sum(1)
    vis = lab < GRID
    l1 = (loc - lab.float()).abs()[vis].mean() if vis.any() else out.sum() * 0
    return ce + 0.05 * l1


@torch.no_grad()
def evaluate(net, loader, device):
    """Ego-lane x error in pixels on visible rows, and lane/no-lane agreement."""
    net.eval()
    errs, agree, n = [], 0, 0
    for x, lab in loader:
        out = net(x.to(device)).float().cpu()
        prob = out[:, :GRID].softmax(1)
        loc = (prob * torch.arange(GRID)[None, :, None, None]).sum(1)
        pred_vis = out.argmax(1) != GRID
        vis = lab < GRID
        agree += (pred_vis == vis).sum().item()
        n += vis.numel()
        ego = torch.zeros_like(vis)
        ego[..., 1:3] = True
        m = vis & pred_vis & ego
        errs.append(((loc - lab.float()) * COL_W).abs()[m])
    e = torch.cat(errs).numpy()
    return dict(ego_px_mean=float(e.mean()), ego_px_p95=float(np.percentile(e, 95)), exist_acc=agree / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/md_lanes")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default="models/culane_18_md.onnx")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    n = len(np.load(Path(args.data) / "labels.npz")["labels"])
    n_val = int(n * args.val_frac)
    train_idx, val_idx = np.arange(n - n_val), np.arange(n - n_val, n)  # later frames = unseen maps
    tr = torch.utils.data.DataLoader(LaneSet(args.data, train_idx, True), batch_size=args.batch, shuffle=True,
                                     num_workers=4, drop_last=True, persistent_workers=True)
    va = torch.utils.data.DataLoader(LaneSet(args.data, val_idx, False), batch_size=32, num_workers=4)

    net = convert(onnx.load(str(MODEL_PATH))).to(device)
    base = evaluate(net, va, device)
    print(f"original CULane model on held-out sim maps: {base}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr), pct_start=0.1)
    for ep in range(args.epochs):
        net.train()
        t0, tot = time.perf_counter(), 0.0
        for x, lab in tr:
            loss = loss_fn(net(x.to(device)), lab.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        m = evaluate(net, va, device)
        print(f"epoch {ep + 1}/{args.epochs}  loss {tot / len(tr):.3f}  val {m}  ({time.perf_counter() - t0:.0f}s)")

    net.eval().cpu()
    torch.onnx.export(net, torch.zeros(1, 3, IN_H, IN_W), args.out, input_names=["input.1"], output_names=["out"],
                      opset_version=13, dynamo=False)
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()

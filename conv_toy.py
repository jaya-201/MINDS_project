"""
Toy replication of "Detecting Generated Images by Fitting Natural Image Distributions"
(ConV and F-ConV, NeurIPS 2025) using a pretrained DINOv2.

ConV   (training-free):  score(x) = 1 - mean_k cos( phi(x), phi(h_k(x)) )      [Eq. 12/13]
                         phi = DINOv2 CLS feature, h = flip + color jitter + gaussian blur
F-ConV (needs fake data): small RealNVP on top of DINOv2 features trained with Eq. 15.

Folder layout expected:
    real_dir/                 natural images (any sub-folders)
    fake_dir/                 EITHER images directly (one generator called "fake")
                              OR one sub-folder per generator: fake_dir/sd15/, fake_dir/biggan/, ...

Example:
    python conv_toy.py --real_dir data/real --fake_dir data/fake --model dinov2_vitb14 --n_views 10
    python conv_toy.py --real_dir data/real --fake_dir data/fake --train_gen sd15   # F-ConV trained on sd15 only
    python conv_toy.py ... --aug rotation          # ablation: transformation DINOv2 was NOT trained to ignore

Requirements: torch torchvision pillow scikit-learn matplotlib tqdm
"""
import argparse
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def list_images(d):
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT)


def collect(real_dir, fake_dir, max_per_group, seed):
    """
    Collect images and assign train/test splits by filename stem.

    Matching IDs across real/fake folders are always assigned to the same
    split, preventing paired source images from crossing the train/test
    boundary. max_per_group limits the number of unique IDs selected.
    """
    rng = random.Random(seed)

    def index_images(paths, group_name):
        indexed = {}
        for path in paths:
            image_id = path.stem
            if image_id in indexed:
                raise ValueError(
                    f"Duplicate image ID '{image_id}' in group '{group_name}'. "
                    "Expected one image per ID per group."
                )
            indexed[image_id] = path
        return indexed

    groups = {"real": index_images(list_images(real_dir), "real")}

    fake_dir = Path(fake_dir)
    if not fake_dir.exists():
        raise FileNotFoundError(f"Fake image directory does not exist: {fake_dir}")

    direct = [
        p for p in fake_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXT
    ]
    if direct:
        groups["fake"] = index_images(direct, "fake")

    for sub in sorted(p for p in fake_dir.iterdir() if p.is_dir()):
        imgs = list_images(sub)
        if imgs:
            groups[sub.name] = index_images(imgs, sub.name)

    all_ids = sorted(set().union(*(set(group.keys()) for group in groups.values())))
    if not all_ids:
        raise ValueError("No images found in the supplied real/fake directories.")

    # Select IDs once globally, so matching images remain paired.
    if max_per_group is not None and max_per_group > 0 and len(all_ids) > max_per_group:
        rng.shuffle(all_ids)
        selected_ids = all_ids[:max_per_group]
    else:
        selected_ids = all_ids[:]
        rng.shuffle(selected_ids)

    # A single split assignment per source-image ID, shared by every group.
    split_by_id = {}
    n_train = len(selected_ids) // 2
    for i, image_id in enumerate(selected_ids):
        split_by_id[image_id] = 0 if i < n_train else 1

    gen_names = list(groups.keys())  # index 0 == "real"
    items = []  # (path, gen_id, split): 0 = train, 1 = test

    for gid, name in enumerate(gen_names):
        for image_id in selected_ids:
            path = groups[name].get(image_id)
            if path is not None:
                items.append((str(path), gid, split_by_id[image_id]))

    # Show whether filename stems actually identify paired source images.
    real_ids = set(groups["real"].keys())
    for name, group in groups.items():
        if name == "real":
            continue
        shared = len(real_ids & set(group.keys()))
        fraction = shared / max(1, len(group))
        print(
            f"pairing with real for '{name}': {shared} shared IDs "
            f"({fraction:.1%} of this group)"
        )

    return items, gen_names


class ViewsDataset(Dataset):
    """Returns the original image and n randomly transformed views h_k(x)."""

    def __init__(self, items, n_views, aug="paper"):
        self.items, self.n = items, n_views
        self.base = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                               T.CenterCrop(224)])
        self.norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
        if aug == "paper":  # Appendix E: factors [0.88,1.12], [0.88,1.12], [0.94,1.06], [0.97,1.03]
            self.h = T.Compose([
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.03),
                T.GaussianBlur(9, sigma=(0.7, 1.0)),
            ])
        elif aug == "rotation":  # Table 3 ablation
            self.h = T.RandomRotation(45)
        else:
            raise ValueError(aug)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, gid, _ = self.items[i]
        img = self.base(Image.open(path).convert("RGB"))
        x = self.norm(img)
        views = torch.stack([self.norm(self.h(img)) for _ in range(self.n)])
        return x, views, gid


@torch.no_grad()
def extract(model, items, args, device):
    loader = DataLoader(ViewsDataset(items, args.n_views, args.aug), batch_size=args.batch,
                        num_workers=args.workers, shuffle=False)
    feats, afeats, gens = [], [], []
    use_amp = device.type == "cuda"
    for x, views, g in tqdm(loader, desc="DINOv2 features"):
        B, n = views.shape[:2]
        inp = torch.cat([x, views.flatten(0, 1)]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model(inp)  # CLS embedding, (B*(n+1), D)
        out = out.float().cpu()
        feats.append(out[:B])
        afeats.append(out[B:].view(B, n, -1))
        gens.append(g)
    return torch.cat(feats), torch.cat(afeats), torch.cat(gens)


# ----------------------------------------------------------------------------
# ConV score
# ----------------------------------------------------------------------------
def mean_cos(v, va, k=None):
    """Mean cosine similarity between feature of x (N,D) and its views (N,n,D)."""
    if k is not None:
        va = va[:, :k]
    return F.cosine_similarity(v.unsqueeze(1), va, dim=-1).mean(1)


def evaluate(score, gens, splits, gen_names):
    """score: higher => more likely generated. Returns {generator: (AUROC, AP)} on test split."""
    te = splits == 1
    real = te & (gens == 0)
    res = {}
    for g in range(1, len(gen_names)):
        m = real | (te & (gens == g))
        y = (gens[m] > 0).astype(int)
        res[gen_names[g]] = (100 * roc_auc_score(y, score[m]), 100 * average_precision_score(y, score[m]))
    res["AVERAGE"] = tuple(np.mean([r for r in res.values()], axis=0))
    return res


def print_table(title, res):
    print(f"\n=== {title} ===")
    print(f"{'generator':<20}{'AUROC':>8}{'AP':>8}")
    for k, (a, p) in res.items():
        print(f"{k:<20}{a:8.2f}{p:8.2f}")


# ----------------------------------------------------------------------------
# RealNVP (F-ConV)
# ----------------------------------------------------------------------------
class Coupling(nn.Module):
    def __init__(self, dim, hidden, flip):
        super().__init__()
        self.flip, self.d1 = flip, dim // 2
        d2 = dim - self.d1
        in_d, out_d = (d2, self.d1) if flip else (self.d1, d2)
        self.net = nn.Sequential(nn.Linear(in_d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2 * out_d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        x1, x2 = x[:, :self.d1], x[:, self.d1:]
        if self.flip:
            x1, x2 = x2, x1  # x1 = conditioner, x2 = transformed half
        s, t = self.net(x1).chunk(2, dim=1)
        s = torch.tanh(s)
        y2 = x2 * torch.exp(s) + t
        out = torch.cat([y2, x1], 1) if self.flip else torch.cat([x1, y2], 1)
        return out, s.sum(1)


class RealNVP(nn.Module):
    def __init__(self, dim, hidden=512, n_blocks=2):
        super().__init__()
        self.layers = nn.ModuleList([Coupling(dim, hidden, flip=bool(i % 2)) for i in range(n_blocks)])
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))

    def forward(self, x):
        x = (x - self.mu) / self.sd  # fixed standardisation (constant log-det ignored)
        logdet = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld
        return x, logdet


def gauss_logp(z, logdet):
    """log p(v) = log N(z;0,I) + log|det df/dv|   (Eq. 14)."""
    return (-0.5 * (z ** 2 + math.log(2 * math.pi))).sum(1) + logdet


def train_flow(v, va, is_fake, args, device):
    """Eq. 15.  Shaping loss: raise log p on natural, lower it on generated.
    Consistency loss: cos(f(v), f(T(v))) up for natural, down for generated."""
    D, n = v.shape[1], va.shape[1]
    flow = RealNVP(D, args.flow_hidden, n_blocks=2).to(device)
    real_v = v[~is_fake]
    flow.mu.copy_(real_v.mean(0))
    flow.sd.copy_(real_v.std(0) + 1e-6)
    vr, ar = v[~is_fake].to(device), va[~is_fake].to(device)
    vg, ag = v[is_fake].to(device), va[is_fake].to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=args.flow_lr, betas=(0.9, 0.99), weight_decay=0.01)
    B = args.flow_batch

    def draw(vv, aa):
        i = torch.randint(len(vv), (B,), device=device)
        k = torch.randint(n, (B,), device=device)
        return vv[i], aa[i, k]

    for step in range(1, args.flow_steps + 1):
        xr, xrT = draw(vr, ar)
        xg, xgT = draw(vg, ag)
        zr, ldr = flow(xr)
        zrT, _ = flow(xrT)
        zg, ldg = flow(xg)
        zgT, _ = flow(xgT)
        lpr, lpg = gauss_logp(zr, ldr) / D, gauss_logp(zg, ldg) / D  # per-dimension
        # "+E_g log p(v)" is unbounded below, so we use a hinge: only push generated
        # likelihood until it is `margin` below the natural one (deviation from the paper, for stability)
        shaping = -lpr.mean() + F.relu(lpg - (lpr.detach().mean() - args.margin)).mean()
        consistency = -F.cosine_similarity(zr, zrT).mean() + F.cosine_similarity(zg, zgT).mean()
        loss = shaping + args.lam * consistency
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, args.flow_steps // 5) == 0 or step == 1:
            print(f"  step {step:5d}  loss {loss.item():8.4f}  shaping {shaping.item():8.4f}  "
                  f"consistency {consistency.item():8.4f}")
    return flow.eval()


@torch.no_grad()
def flow_scores(flow, v, va, device, bs=512):
    """Returns (1 - flow-space cosine consistency, negative log-likelihood), both 'higher = generated'."""
    inc, nll = [], []
    for i in range(0, len(v), bs):
        vb, ab = v[i:i + bs].to(device), va[i:i + bs].to(device)
        B, n, D = ab.shape
        z, ld = flow(vb)
        za, _ = flow(ab.reshape(B * n, D))
        cos = F.cosine_similarity(z.unsqueeze(1), za.view(B, n, D), dim=-1).mean(1)
        inc.append((1 - cos).cpu())
        nll.append((-gauss_logp(z, ld) / D).cpu())
    return torch.cat(inc).numpy(), torch.cat(nll).numpy()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--model", default="dinov2_vitb14",
                    choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                    help="paper uses dinov2_vitl14")
    ap.add_argument("--n_views", type=int, default=10, help="paper uses n=20")
    ap.add_argument("--aug", default="paper", choices=["paper", "rotation"])
    ap.add_argument("--max_per_group", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None, help="path to cache extracted features (.pt)")
    ap.add_argument("--no_flow", action="store_true", help="skip F-ConV")
    ap.add_argument("--train_gen", default=None, help="generator used to train the flow (default: first)")
    ap.add_argument("--flow_steps", type=int, default=2000)
    ap.add_argument("--flow_batch", type=int, default=128)
    ap.add_argument("--flow_lr", type=float, default=1e-4)
    ap.add_argument("--flow_hidden", type=int, default=512)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of consistency loss")
    ap.add_argument("--out", default="conv_hist.png")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    items, gen_names = collect(args.real_dir, args.fake_dir, args.max_per_group, args.seed)
    print("groups:", {n: sum(1 for it in items if it[1] == i) for i, n in enumerate(gen_names)})
    splits = np.array([it[2] for it in items])

    cache = Path(args.cache or f"cache_{args.model}_{args.n_views}_{args.aug}.pt")
    if cache.exists():
        print("loading cached features:", cache)
        d = torch.load(cache)
        v, va, gens = d["v"], d["va"], d["gens"]
    else:
        model = torch.hub.load("facebookresearch/dinov2", args.model).to(device).eval()
        v, va, gens = extract(model, items, args, device)
        torch.save({"v": v, "va": va, "gens": gens}, cache)
    gens_np = gens.numpy()

    # ---------------- ConV (training-free) ----------------
    sim = mean_cos(v, va)  # (N,)  high => natural
    print_table(f"ConV  ({args.model}, n={args.n_views}, aug={args.aug})",
                evaluate((1 - sim).numpy(), gens_np, splits, gen_names))

    # Fig. 5 analogue: AUROC vs number of random transformations
    print("\nAverage AUROC vs number of views k (Fig. 5 analogue):")
    for k in sorted({1, 2, 5, args.n_views // 2, args.n_views} - {0}):
        r = evaluate((1 - mean_cos(v, va, k)).numpy(), gens_np, splits, gen_names)["AVERAGE"]
        print(f"  k={k:3d}  AUROC {r[0]:.2f}  AP {r[1]:.2f}")

    # Fig. 7 analogue: cosine similarity histograms
    plt.figure(figsize=(6, 3.5))
    for gid, name in enumerate(gen_names):
        plt.hist(sim.numpy()[gens_np == gid], bins=40, alpha=0.5, density=True, label=name)
    plt.xlabel("cosine similarity  cos(phi(x), phi(h(x)))")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("saved histogram to", args.out)

    # ---------------- F-ConV ----------------
    if args.no_flow or len(gen_names) < 2:
        return
    tg = gen_names.index(args.train_gen) if args.train_gen else 1
    print(f"\nTraining RealNVP on natural vs '{gen_names[tg]}' (train split only)")
    tr = (splits == 0) & ((gens_np == 0) | (gens_np == tg))
    tr_t = torch.from_numpy(tr)
    is_fake = torch.from_numpy(gens_np[tr] > 0)
    flow = train_flow(v[tr_t], va[tr_t], is_fake, args, device)

    inc, nll = flow_scores(flow, v, va, device)
    ref = tr & (gens_np == 0)  # standardise scores with natural-train statistics
    z = lambda s: (s - s[ref].mean()) / (s[ref].std() + 1e-8)
    print_table("F-ConV: flow-space inconsistency only", evaluate(inc, gens_np, splits, gen_names))
    print_table("F-ConV: flow likelihood only", evaluate(nll, gens_np, splits, gen_names))
    print_table(f"F-ConV: combined (trained on '{gen_names[tg]}')",
                evaluate(z(inc) + z(nll), gens_np, splits, gen_names))


if __name__ == "__main__":
    main()
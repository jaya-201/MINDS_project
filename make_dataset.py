"""
Build a small real-vs-generated dataset for conv_toy.py.

Real images : a random subset of COCO val2017 (the same source DRCT-2M uses for natural images).
Fake images : generated with Stable Diffusion from the COCO *captions of those same real images*,
              so real and fake have matching content (reduces a "different subject matter" shortcut).

Usage (two steps):
    python make_dataset.py real --data_dir data --n 300
    python make_dataset.py fake --data_dir data --name sd15    --n 300
    python make_dataset.py fake --data_dir data --name sdturbo --n 300     # 2nd generator (unseen-generator test)

Resulting layout (exactly what conv_toy.py expects):
    data/real/*.jpg
    data/fake/sd15/*.jpg
    data/fake/sdturbo/*.jpg

Then:
    python conv_toy.py --real_dir data/real --fake_dir data/fake --train_gen sd15

Requirements:
    real : pillow tqdm                     (no GPU, ~250 MB annotation download + n small images)
    fake : torch diffusers transformers accelerate safetensors   (GPU strongly recommended)
"""
import argparse
import json
import random
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
CAP_MEMBER = "annotations/captions_val2017.json"

# name -> (huggingface model id, inference steps, guidance scale). Override with --model_id/--steps/--guidance.
# NOTE: Hugging Face model repos get moved/renamed; if a download fails, check the model id on huggingface.co.
PRESETS = {
    "sd15": ("stable-diffusion-v1-5/stable-diffusion-v1-5", 30, 7.5),
    "sdturbo": ("stabilityai/sd-turbo", 2, 0.0),
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def http_get(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:  # noqa
            if i == retries - 1:
                raise e


def download_file(url, dst):
    with tqdm(unit="B", unit_scale=True, desc=f"downloading {Path(dst).name}") as bar:
        def hook(_, bs, total):
            bar.total = total
            bar.update(bs)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        opener = urllib.request.build_opener()
        opener.addheaders = list(req.header_items())
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, dst, hook)


def prep(img, size):
    """Resize shorter side to `size`, then centre-crop to size x size (applied identically to real and fake)."""
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return img.crop((l, t, l + size, t + size))


def save_jpg(img, path, quality):
    # Same format + quality for real and fake, so the detector can't just learn "PNG vs JPEG".
    img.save(path, "JPEG", quality=quality)


# ----------------------------------------------------------------------------
# step 1: real images
# ----------------------------------------------------------------------------
def cmd_real(a):
    data = Path(a.data_dir)
    (data / "real").mkdir(parents=True, exist_ok=True)
    cap_json = data / "captions_val2017.json"
    if not cap_json.exists():
        zpath = data / "annotations_trainval2017.zip"
        download_file(ANN_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            cap_json.write_bytes(z.read(CAP_MEMBER))
        zpath.unlink()

    meta = json.loads(cap_json.read_text())
    captions = {}
    for c in meta["annotations"]:
        captions.setdefault(c["image_id"], c["caption"].strip())  # first caption per image
    images = sorted(meta["images"], key=lambda d: d["id"])
    random.Random(a.seed).shuffle(images)

    prompts, done = {}, 0
    for im in tqdm(images, desc="real images", total=a.n):
        if done >= a.n:
            break
        try:
            raw = http_get(im["coco_url"])
            from io import BytesIO
            img = prep(Image.open(BytesIO(raw)), a.size)
        except Exception as e:  # noqa
            print("skip", im["id"], e)
            continue
        save_jpg(img, data / "real" / f"{im['id']:012d}.jpg", a.jpeg_quality)
        prompts[f"{im['id']:012d}"] = captions[im["id"]]
        done += 1
    (data / "prompts.json").write_text(json.dumps(prompts, indent=1))
    print(f"saved {done} real images to {data / 'real'} and captions to {data / 'prompts.json'}")


# ----------------------------------------------------------------------------
# step 2: fake images
# ----------------------------------------------------------------------------
def cmd_fake(a):
    import torch
    from diffusers import AutoPipelineForText2Image

    data = Path(a.data_dir)
    prompts = json.loads((data / "prompts.json").read_text())
    ids = sorted(prompts)[: a.n]
    out = data / "fake" / a.name
    out.mkdir(parents=True, exist_ok=True)

    model_id, steps, guidance = PRESETS.get(a.name, (None, 30, 7.5))
    model_id = a.model_id or model_id
    steps = a.steps if a.steps is not None else steps
    guidance = a.guidance if a.guidance is not None else guidance
    if model_id is None:
        raise SystemExit("unknown --name; pass --model_id explicitly")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found. Generation on CPU is very slow; consider Google Colab (free T4).")
    kw = {"safety_checker": None} if a.name == "sd15" else {}  # avoid all-black 'NSFW' outputs
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32, **kw).to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"generator={a.name} model={model_id} steps={steps} guidance={guidance}")

    todo = [i for i in ids if not (out / f"{i}.jpg").exists()]  # resumable
    for s in tqdm(range(0, len(todo), a.batch), desc=f"fake/{a.name}"):
        chunk = todo[s:s + a.batch]
        gens = [torch.Generator(device).manual_seed(a.seed + int(i)) for i in chunk]
        imgs = pipe([prompts[i] for i in chunk], num_inference_steps=steps, guidance_scale=guidance,
                    height=a.size, width=a.size, generator=gens).images
        for i, im in zip(chunk, imgs):
            save_jpg(prep(im, a.size), out / f"{i}.jpg", a.jpeg_quality)
    print(f"done: {len(list(out.glob('*.jpg')))} images in {out}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("real", "fake"):
        p = sub.add_parser(name)
        p.add_argument("--data_dir", default="data")
        p.add_argument("--n", type=int, default=300)
        p.add_argument("--size", type=int, default=512)
        p.add_argument("--jpeg_quality", type=int, default=95)
        p.add_argument("--seed", type=int, default=0)
    f = sub.choices["fake"]
    f.add_argument("--name", default="sd15", help="output folder / preset: sd15 or sdturbo (or any name + --model_id)")
    f.add_argument("--model_id", default=None)
    f.add_argument("--steps", type=int, default=None)
    f.add_argument("--guidance", type=float, default=None)
    f.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    cmd_real(a) if a.cmd == "real" else cmd_fake(a)
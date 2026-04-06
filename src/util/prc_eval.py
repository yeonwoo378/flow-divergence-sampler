# util/prc_eval.py
from __future__ import annotations

import os
import shutil
import zipfile
import hashlib
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


IMG_EXTS_DEFAULT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _atomic_write_from_zip(zip_path: str, member: str, out_path: str) -> None:
    """Extract one member from zip to out_path atomically."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"

    with zipfile.ZipFile(zip_path, "r") as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(
                f'NPZ "{zip_path}" does not contain member "{member}". '
                f"Available members (first 20): {zf.namelist()[:20]}"
            )
        with zf.open(member, "r") as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

    os.replace(tmp_path, out_path)


def _default_extracted_npy_path(npz_path: str, arr_name: str, cache_dir: str) -> str:
    """
    Make a stable cache filename:
    - include basename
    - include key name
    - include md5 of absolute path to avoid collisions
    """
    npz_path = _abspath(npz_path)
    h = hashlib.md5(npz_path.encode("utf-8")).hexdigest()[:10]
    base = os.path.splitext(os.path.basename(npz_path))[0]
    return os.path.join(cache_dir, f"{base}_{arr_name}_{h}.npy")


def ensure_npy_extracted_from_npz(
    npz_path: str,
    arr_name: str = "arr_0",
    cache_dir: Optional[str] = None,
) -> str:
    """
    For large OpenAI VIRTUAL_*.npz, loading with np.load() can eat RAM (~GBs).
    We extract {arr_name}.npy once, then memory-map it.
    """
    npz_path = _abspath(npz_path)
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(npz_path), ".npz_cache")

    out_npy = _default_extracted_npy_path(npz_path, arr_name, cache_dir)
    if os.path.exists(out_npy):
        return out_npy

    # Extract arr_name.npy from the zip (.npz is zip)
    member = f"{arr_name}.npy"
    _atomic_write_from_zip(npz_path, member, out_npy)
    return out_npy


class NpzImagesDataset(Dataset):
    """
    Dataset for OpenAI guided-diffusion reference batch format.
    - expects images in arr_0 by default (NHWC uint8 in [0,255]).
    - returns torch.uint8 CHW in [0,255], which torch-fidelity expects.  :contentReference[oaicite:3]{index=3}
    """

    def __init__(
        self,
        npz_path: str,
        arr_name: str = "arr_0",
        cache_dir: Optional[str] = None,
        max_items: Optional[int] = None,
        seed: int = 0,
        force_channel_first: bool = True,
    ):
        super().__init__()
        self.npz_path = _abspath(npz_path)
        self.arr_name = arr_name
        self.force_channel_first = force_channel_first

        # Extract .npy to disk and memmap for low RAM usage
        npy_path = ensure_npy_extracted_from_npz(self.npz_path, arr_name=self.arr_name, cache_dir=cache_dir)
        self._arr = np.load(npy_path, mmap_mode="r")  # memmap ndarray

        n = int(self._arr.shape[0])
        if (max_items is None) or (max_items >= n):
            self._indices = np.arange(n, dtype=np.int64)
        else:
            raise ValueError(f"do not sample")
            rng = np.random.RandomState(seed)
            self._indices = rng.choice(n, size=int(max_items), replace=False).astype(np.int64)
            self._indices.sort()  # stable order

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        i = int(self._indices[idx])
        img = self._arr[i]  # typically HWC uint8

        if img.ndim != 3:
            raise ValueError(f"Expected 3D image array, got shape={img.shape} from {self.npz_path}:{self.arr_name}")

        # Convert to CHW
        if self.force_channel_first:
            # HWC -> CHW
            if img.shape[-1] in (1, 3):
                img = np.transpose(img, (2, 0, 1))
            elif img.shape[0] in (1, 3):
                # already CHW
                pass
            else:
                raise ValueError(f"Cannot infer channel dimension for shape={img.shape}")
        img = np.ascontiguousarray(img)

        t = torch.from_numpy(img)
        if t.dtype != torch.uint8:
            # be robust: handle float [0,1] or [0,255]
            t = t.to(torch.float32)
            t = torch.clamp(t, 0.0, 255.0)
            if float(t.max()) <= 1.0:
                t = torch.round(t * 255.0)
            else:
                t = torch.round(t)
            t = t.to(torch.uint8)

        return t


def list_image_files(
    root_dir: str,
    recursive: bool = True,
    exts: Sequence[str] = IMG_EXTS_DEFAULT,
) -> List[str]:
    root_dir = _abspath(root_dir)
    exts_l = set([e.lower() for e in exts])

    out: List[str] = []
    if recursive:
        for dp, _, fnames in os.walk(root_dir):
            for fn in fnames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in exts_l:
                    out.append(os.path.join(dp, fn))
    else:
        for fn in os.listdir(root_dir):
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts_l:
                out.append(os.path.join(root_dir, fn))

    out.sort()
    return out


class ImageFilesDataset(Dataset):
    """
    Map-style dataset for a list of image files.
    Returns torch.uint8 CHW in [0,255] (PIL->tensor).
    """

    def __init__(self, files: Sequence[str]):
        super().__init__()
        self.files = list(files)
        if len(self.files) == 0:
            raise ValueError("ImageFilesDataset got empty file list")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return pil_to_tensor(img)  # uint8 CHW


def subsample_files(
    files: Sequence[str],
    max_items: Optional[int],
    seed: int = 0,
) -> List[str]:
    raise ValueError(f"do not sample")
    files = list(files)
    if (max_items is None) or (max_items >= len(files)):
        return files
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(files), size=int(max_items), replace=False)
    idx.sort()
    return [files[i] for i in idx]


def calculate_precision_recall_with_virtual_imagenet_npz(
    gen_images_dir: str,
    virtual_npz_path: str,
    *,
    # sampling controls
    prc_num_gen: Optional[int] = None,     # None => use all generated images found
    prc_num_ref: Optional[int] = None,     # None => use all reference images in NPZ (OpenAI: 10k)
    seed: int = 0,
    # torch-fidelity knobs
    batch_size: int = 64,                  # feature extraction batch size
    prc_batch_size: int =10000,           # distance block batch size
    prc_neighborhood: int = 3,             # k in k-NN
    save_cpu_ram: bool = False,
    cuda: bool = True,
    verbose: bool = False,
    # caching
    cache: bool = True,
    cache_root: Optional[str] = None,
    npz_cache_dir: Optional[str] = None,
) -> dict:
    """
    Compute precision/recall (PRC) between:
      - input1: generated images directory
      - input2: OpenAI VIRTUAL_imagenet256_labeled.npz (arr_0 images)
    """

    import torch_fidelity

    # Reference images from NPZ (typically 10k for PR/Recall in OpenAI ref batch) :contentReference[oaicite:4]{index=4}
    ref_ds = NpzImagesDataset(
        npz_path=virtual_npz_path,
        arr_name="arr_0",
        cache_dir=npz_cache_dir,
        max_items=prc_num_ref,
        seed=seed,
    )

    # Generated images from directory
    gen_files = list_image_files(gen_images_dir, recursive=True)
    # gen_files = subsample_files(gen_files, prc_num_gen, seed=seed)
    gen_ds = ImageFilesDataset(gen_files)

    # IMPORTANT:
    # - OpenAI evaluator computes PR on Inception pool_3 activations (2048-d). :contentReference[oaicite:5]{index=5}
    # - torch-fidelity's PRC default FE may be VGG16, so override to inception-v3-compat.
    prc_metrics = torch_fidelity.calculate_metrics(
        input1=gen_ds,
        input2=ref_ds,
        cuda=cuda,
        # verbose=verbose,
        cache=cache,
        cache_root=cache_root,

        # compute only PRC here
        prc=True,
        isc=False,
        fid=False,
        kid=False,

        # match OpenAI-style PR feature space (Inception pool_3 / 2048)
        feature_extractor="inception-v3-compat",
        feature_layer_prc="2048",

        # PRC algorithm params
        prc_neighborhood=prc_neighborhood,
        prc_batch_size=prc_batch_size,

        # memory/perf
        save_cpu_ram=False,
        batch_size=batch_size,

        # make reference caching stable across runs (optional but recommended)
        input2_cache_name=f"virtual_imagenet_prc_ref_{os.path.basename(_abspath(virtual_npz_path))}",
    )
    return prc_metrics
"""
Dataset utilities for Vision-Transformer continual learning.

Supported benchmarks
--------------------
* **Split CIFAR-100** – 100 classes split into N tasks × (100/N) classes.
  Default: 10 tasks × 10 classes.  Images are 32×32 but resized to 224×224.

* **Split CUB-200**   – 200 fine-grained bird classes split into N tasks.
  Default: 10 tasks × 20 classes.  Needs the raw CUB-200-2011 archive.

Both return plain ``torch.utils.data.DataLoader`` objects grouped by task,
matching the dict-of-dataloaders interface used by ``CL_Base_Model``.

Usage
-----
    from utils.data.vit_data_utils import build_split_cifar100, build_split_cub200

    train_tasks, val_tasks, test_tasks, class_masks = build_split_cifar100(
        data_root="./data/cifar100",
        n_tasks=10,
        batch_size=64,
    )
"""

import os
import math
import subprocess
import shutil
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader


# ---------------------------------------------------------------------------
# Robust downloader (wget with resume, falls back to torchvision)
# ---------------------------------------------------------------------------

_CIFAR100_URL  = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
_CIFAR100_FILE = "cifar-100-python.tar.gz"
_CIFAR100_MD5  = "eb9058c3a382ffc7106e4002c42a8d85"   # official MD5


def _wget_download(url: str, dest_dir: str, filename: str) -> bool:
    """
    Download ``url`` into ``dest_dir/filename`` using wget with:
    * ``--continue``  – resume interrupted downloads
    * ``--tries=5``   – retry on transient failures
    * ``--timeout=30`` – per-connection timeout

    Returns True on success, False if wget is not installed.
    """
    if shutil.which("wget") is None:
        return False

    dest_path = os.path.join(dest_dir, filename)
    os.makedirs(dest_dir, exist_ok=True)

    cmd = [
        "wget",
        "--continue",
        "--tries=5",
        "--timeout=30",
        "--show-progress",
        "-O", dest_path,
        url,
    ]
    print(f"[vit_data_utils] Downloading via wget: {url}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def _ensure_cifar100(data_root: str):
    """
    Make sure the CIFAR-100 archive is fully present and intact before
    torchvision tries to load it.  Cleans up partial downloads automatically.
    """
    archive = os.path.join(data_root, _CIFAR100_FILE)
    extracted = os.path.join(data_root, "cifar-100-python")

    # Already extracted – nothing to do
    if os.path.isdir(extracted):
        return

    # Check if existing archive is the right size (≈161 MB)
    if os.path.isfile(archive):
        size_mb = os.path.getsize(archive) / (1024 ** 2)
        if size_mb < 150:
            print(f"[vit_data_utils] Partial archive detected ({size_mb:.1f} MB), removing.")
            os.remove(archive)

    # Download with wget if archive is missing
    if not os.path.isfile(archive):
        ok = _wget_download(_CIFAR100_URL, data_root, _CIFAR100_FILE)
        if not ok:
            print("[vit_data_utils] wget unavailable – falling back to torchvision downloader.")


# ---------------------------------------------------------------------------
# ViT-compatible transforms
# ---------------------------------------------------------------------------

def get_vit_transforms(img_size: int = 224, augment: bool = True):
    """
    Returns (train_transform, val_transform) suitable for ViT.

    * Train: random crop + horizontal flip + colour jitter + normalize
    * Val  : centre crop + normalize
    """
    mean = (0.485, 0.456, 0.406)   # ImageNet stats (ViT was pre-trained on ImageNet-21k)
    std  = (0.229, 0.224, 0.225)

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_tf = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Task split helper
# ---------------------------------------------------------------------------

def split_classes_into_tasks(
    n_classes: int,
    n_tasks: int,
    shuffle: bool = True,
    seed: int = 1234,
) -> List[List[int]]:
    """
    Partition ``n_classes`` class-IDs into ``n_tasks`` equal groups.

    Returns
    -------
    List[List[int]]
        ``class_masks[task_id]`` is the list of original class labels for
        that task.
    """
    rng = np.random.default_rng(seed)
    order = list(range(n_classes))
    if shuffle:
        rng.shuffle(order)

    classes_per_task = n_classes // n_tasks
    return [order[i * classes_per_task: (i + 1) * classes_per_task]
            for i in range(n_tasks)]


def make_task_subset(
    dataset: Dataset,
    class_ids: List[int],
    remap_labels: bool = True,
) -> Dataset:
    """
    Return a Subset of ``dataset`` containing only samples from ``class_ids``.

    If ``remap_labels=True`` the class labels are remapped to
    ``[0, len(class_ids))`` for use with a task-specific head.
    """
    targets = np.array(dataset.targets)
    indices = np.where(np.isin(targets, class_ids))[0].tolist()

    subset = Subset(dataset, indices)

    if remap_labels:
        label_map = {old: new for new, old in enumerate(sorted(class_ids))}
        subset = _RemappedSubset(dataset, indices, label_map)

    return subset


class _RemappedSubset(Dataset):
    """Subset with remapped integer labels."""

    def __init__(self, dataset: Dataset, indices: List[int], label_map: dict):
        self.dataset   = dataset
        self.indices   = indices
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, self.label_map[label]


# ---------------------------------------------------------------------------
# Split CIFAR-100
# ---------------------------------------------------------------------------

def build_split_cifar100(
    data_root: str = "./data/cifar100",
    n_tasks: int = 10,
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 4,
    augment: bool = True,
    shuffle_classes: bool = True,
    seed: int = 1234,
    val_split: float = 0.1,
) -> Tuple[Dict, Dict, Dict, List[List[int]]]:
    """
    Build Split CIFAR-100 dataloaders.

    Parameters
    ----------
    data_root : str
        Where to cache / find CIFAR-100 files.
    n_tasks : int
        Number of tasks to split 100 classes into (must divide 100 evenly).
    batch_size : int
        Per-task batch size.
    img_size : int
        Resize target; 224 for standard ViT-B/16.
    num_workers : int
        DataLoader worker processes.
    augment : bool
        Enable training augmentation.
    shuffle_classes : bool
        Randomly permute class-to-task assignment.
    seed : int
        Random seed for reproducibility.
    val_split : float
        Fraction of training data used for validation.

    Returns
    -------
    train_tasks : Dict[str, DataLoader]
        Keys: task names ``"task_0" … "task_{n_tasks-1}"``.
    val_tasks   : Dict[str, DataLoader]
    test_tasks  : Dict[str, DataLoader]
    class_masks : List[List[int]]
        ``class_masks[i]`` – original CIFAR-100 class IDs for task ``i``.
    """
    assert 100 % n_tasks == 0, f"n_tasks={n_tasks} must divide 100"

    train_tf, val_tf = get_vit_transforms(img_size, augment=augment)

    # Ensure the archive is fully downloaded before torchvision touches it
    _ensure_cifar100(data_root)

    # Full datasets (no remapping yet)
    full_train = datasets.CIFAR100(data_root, train=True,  download=True,
                                   transform=train_tf)
    full_val   = datasets.CIFAR100(data_root, train=True,  download=False,
                                   transform=val_tf)
    full_test  = datasets.CIFAR100(data_root, train=False, download=True,
                                   transform=val_tf)

    class_masks = split_classes_into_tasks(100, n_tasks,
                                           shuffle=shuffle_classes, seed=seed)

    train_tasks: Dict[str, DataLoader] = {}
    val_tasks:   Dict[str, DataLoader] = {}
    test_tasks:  Dict[str, DataLoader] = {}

    for task_id, class_ids in enumerate(class_masks):
        task_name = f"task_{task_id}"

        # Build per-task subsets with remapped labels [0, classes_per_task)
        train_sub = make_task_subset(full_train, class_ids)
        val_sub   = make_task_subset(full_val,   class_ids)
        test_sub  = make_task_subset(full_test,  class_ids)

        # Carve out validation from training split
        n_train = len(train_sub)
        n_val   = max(1, int(n_train * val_split))
        n_tr    = n_train - n_val
        rng     = torch.Generator().manual_seed(seed + task_id)
        tr_subset, vl_subset = torch.utils.data.random_split(
            train_sub, [n_tr, n_val], generator=rng)

        train_tasks[task_name] = DataLoader(
            tr_subset,  batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True)
        val_tasks[task_name]   = DataLoader(
            vl_subset,  batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        test_tasks[task_name]  = DataLoader(
            test_sub,   batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    return train_tasks, val_tasks, test_tasks, class_masks


# ---------------------------------------------------------------------------
# Split CUB-200-2011
# ---------------------------------------------------------------------------

class CUB200Dataset(Dataset):
    """
    CUB-200-2011 dataset loader.

    Expected directory layout after extracting the official archive::

        cub_root/
            images/
                001.Black_footed_Albatross/
                    Black_Footed_Albatross_0001_796111.jpg
                    ...
                002.Laysan_Albatross/
                    ...
            train_test_split.txt   # 1 = train, 0 = test
            images.txt
            image_class_labels.txt

    Download: https://www.vision.caltech.edu/datasets/cub_200_2011/
    """

    def __init__(self, root: str, train: bool = True,
                 transform=None):
        self.root      = root
        self.train     = train
        self.transform = transform
        self.loader    = default_loader

        self.data: List[str]  = []
        self.targets: List[int] = []

        self._load_metadata()

    def _load_metadata(self):
        images_file  = os.path.join(self.root, "images.txt")
        labels_file  = os.path.join(self.root, "image_class_labels.txt")
        split_file   = os.path.join(self.root, "train_test_split.txt")

        for f in (images_file, labels_file, split_file):
            if not os.path.isfile(f):
                raise FileNotFoundError(
                    f"CUB-200 metadata file not found: {f}\n"
                    "Please download CUB-200-2011 from "
                    "https://www.vision.caltech.edu/datasets/cub_200_2011/ "
                    f"and extract it to {self.root}"
                )

        id2path   = {}
        id2label  = {}
        id2split  = {}

        with open(images_file)  as fh:
            for line in fh:
                img_id, path = line.strip().split()
                id2path[img_id] = path

        with open(labels_file) as fh:
            for line in fh:
                img_id, label = line.strip().split()
                id2label[img_id] = int(label) - 1   # 0-indexed

        with open(split_file) as fh:
            for line in fh:
                img_id, is_train = line.strip().split()
                id2split[img_id] = int(is_train)

        for img_id in sorted(id2path.keys()):
            is_train = id2split[img_id] == 1
            if is_train == self.train:
                self.data.append(os.path.join(self.root, "images", id2path[img_id]))
                self.targets.append(id2label[img_id])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img    = self.loader(self.data[idx])
        label  = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def build_split_cub200(
    data_root: str = "./data/CUB_200_2011",
    n_tasks: int = 10,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    augment: bool = True,
    shuffle_classes: bool = True,
    seed: int = 1234,
    val_split: float = 0.1,
) -> Tuple[Dict, Dict, Dict, List[List[int]]]:
    """
    Build Split CUB-200-2011 dataloaders.

    Parameters are identical to ``build_split_cifar100`` except ``n_tasks``
    must divide 200 evenly.  Default: 10 tasks × 20 classes.

    Returns
    -------
    Same 4-tuple as ``build_split_cifar100``.
    """
    assert 200 % n_tasks == 0, f"n_tasks={n_tasks} must divide 200"

    train_tf, val_tf = get_vit_transforms(img_size, augment=augment)

    full_train = CUB200Dataset(data_root, train=True,  transform=train_tf)
    full_val   = CUB200Dataset(data_root, train=True,  transform=val_tf)
    full_test  = CUB200Dataset(data_root, train=False, transform=val_tf)

    class_masks = split_classes_into_tasks(200, n_tasks,
                                           shuffle=shuffle_classes, seed=seed)

    train_tasks: Dict[str, DataLoader] = {}
    val_tasks:   Dict[str, DataLoader] = {}
    test_tasks:  Dict[str, DataLoader] = {}

    for task_id, class_ids in enumerate(class_masks):
        task_name = f"task_{task_id}"

        train_sub = make_task_subset(full_train, class_ids)
        val_sub   = make_task_subset(full_val,   class_ids)
        test_sub  = make_task_subset(full_test,  class_ids)

        n_train  = len(train_sub)
        n_val    = max(1, int(n_train * val_split))
        n_tr     = n_train - n_val
        rng      = torch.Generator().manual_seed(seed + task_id)
        tr_subset, vl_subset = torch.utils.data.random_split(
            train_sub, [n_tr, n_val], generator=rng)

        train_tasks[task_name] = DataLoader(
            tr_subset,  batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True)
        val_tasks[task_name]   = DataLoader(
            vl_subset,  batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        test_tasks[task_name]  = DataLoader(
            test_sub,   batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    return train_tasks, val_tasks, test_tasks, class_masks

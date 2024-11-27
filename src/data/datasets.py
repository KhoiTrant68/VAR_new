import os

from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS, DatasetFolder
from torchvision.transforms import InterpolationMode, transforms


def normalize_01_into_pm1(x):
    """Normalize x from [0, 1] to [-1, 1] by (x * 2) - 1."""
    return x.mul(2).add_(-1)


def pil_loader(path: str) -> Image.Image:
    """Load an image and convert it to RGB."""
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def build_dataset(
    data_path: str, final_reso: int, hflip: bool = False, mid_reso_scale: float = 1.125
):
    """
    Build training and validation datasets with specified transformations.

    Args:
        data_path (str): Root directory containing `train` and `val` subdirectories.
        final_reso (int): Final resolution of cropped images.
        hflip (bool): Whether to apply horizontal flipping augmentation.
        mid_reso_scale (float): Scale factor for intermediate resolution.

    Returns:
        tuple: Number of classes, training dataset, and validation dataset.
    """
    mid_reso = round(mid_reso_scale * final_reso)

    # Common transformations
    base_transforms = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]

    # Training transformations
    train_transforms = [transforms.RandomCrop((final_reso, final_reso))]
    if hflip:
        train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms += base_transforms

    # Validation transformations
    val_transforms = [transforms.CenterCrop((final_reso, final_reso))] + base_transforms

    # Compose transformations
    train_aug = transforms.Compose(train_transforms)
    val_aug = transforms.Compose(val_transforms)

    # Create datasets
    train_set = DatasetFolder(
        root=os.path.join(data_path, "train"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=train_aug,
    )
    val_set = DatasetFolder(
        root=os.path.join(data_path, "val"),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=val_aug,
    )

    num_classes = len(train_set.classes)

    # Log dataset info and transformations
    print(
        f"[Dataset] train size: {len(train_set)}, val size: {len(val_set)}, num_classes: {num_classes}"
    )
    log_transforms(train_aug, "Training")
    log_transforms(val_aug, "Validation")

    return num_classes, train_set, val_set


def log_transforms(transform: transforms.Compose, label: str):
    """Log the transformations applied to a dataset."""
    print(f"\n[{label} Transformations]")
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            print(f" - {t}")
    else:
        print(f" - {transform}")

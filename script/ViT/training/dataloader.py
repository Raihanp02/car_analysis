import random
from collections import defaultdict
from pathlib import Path
 
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --------------------------------------------------------------------------
# 1. Dataset: ImageFolder-style, one subfolder per fine-grained class
# --------------------------------------------------------------------------
class VehicleDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []       # list[(path, label_idx)]
        self.classes = sorted([str(d.relative_to(self.root).parent) for d in self.root.rglob("*.[jJpP][pPnN]*[gG]")])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
 
        self.class_to_indices = defaultdict(list)  # for PK sampler
        for cls in self.classes:
            for img_path in (self.root / cls).rglob("*.[jJpP][pPnN]*[gG]"):
                idx = len(self.samples)
                self.samples.append((img_path, self.class_to_idx[cls]))
                self.class_to_indices[self.class_to_idx[cls]].append(idx)
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
 
 
# --------------------------------------------------------------------------
# 2. PK Sampler: P classes x K images per batch -> guarantees real positives
#    and lets you engineer hard negatives by forcing twin-pair classes into
#    the same batch (see `twin_groups` below).
# --------------------------------------------------------------------------
class PKSampler(Sampler):
    def __init__(self, group_to_indices, P=16, K=2, iterations=300):
        self.group_to_indices = {g: idx for g, idx in group_to_indices.items() if len(idx) >= K}
        self.groups = list(self.group_to_indices.keys())
        self.P, self.K = P, K
        self.iterations = iterations
 
    def __iter__(self):
        for _ in range(self.iterations):
            chosen_groups = random.sample(self.groups, min(self.P, len(self.groups)))
            batch = []
            for g in chosen_groups:
                pool = self.group_to_indices[g]
                batch.extend(random.sample(pool, self.K))
            yield batch
 
    def __len__(self):
        return self.iterations
 
 
# --------------------------------------------------------------------------
# 3. Augmentation
#    Keep it label-preserving: badges/grilles are the discriminative signal,
#    so avoid aggressive crops that could remove them. Color jitter is
#    useful since paint color must NOT be a shortcut feature.
# --------------------------------------------------------------------------
IMG_SIZE = 224  # match DINOv2 registers checkpoint patch grid (14 * 16)
 
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.35, hue=0.02),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transform_alb = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.7),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ImageCompression(quality_lower=40, quality_upper=90, p=0.3),  # simulate CCTV codec artifacts
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
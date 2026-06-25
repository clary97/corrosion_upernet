import glob
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


# BGR color → class index mapping (기존 레포와 동일)
BGR_TO_CLASS = {
    (0,   0,   0): 0,   # Good
    (0,   0, 128): 1,   # Fair
    (0, 128,   0): 2,   # Poor
    (0, 128, 128): 3,   # Severe
}


def mask_to_index(mask_bgr: np.ndarray) -> np.ndarray:
    h, w = mask_bgr.shape[:2]
    index = np.zeros((h, w), dtype=np.int64)
    for bgr, cls in BGR_TO_CLASS.items():
        match = np.all(mask_bgr == np.array(bgr, dtype=np.uint8), axis=2)
        index[match] = cls
    return index


def get_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            rotate=(-10, 10),
            translate_percent=(-0.025, 0.025),
            shear=(-5, 5),
            scale=(0.975, 1.025),
            p=1.0,
        ),
    ])


class CorrosionDataset(Dataset):
    def __init__(self, root_dir, image_processor, augment=False, num_classes=4):
        """
        root_dir: Train/ 또는 Test/ 경로 (Images/, Masks/ 하위 폴더 포함)
        image_processor: SegformerImageProcessor 인스턴스
        num_classes: 4 → Good/Fair/Poor/Severe, 2 → Good/Corrosion(Fair·Poor·Severe 통합)
        """
        self.image_processor = image_processor
        self.augment = augment
        self.aug = get_augmentation() if augment else None
        self.num_classes = num_classes

        img_dir  = os.path.join(root_dir, 'Images')
        msk_dir  = os.path.join(root_dir, 'Masks')
        all_imgs = sorted(glob.glob(os.path.join(img_dir, '*')))

        # stem 기반 매핑 (확장자 불일치 대응)
        mask_by_stem = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(msk_dir, '*'))
        }
        self.pairs = [
            (img, mask_by_stem[os.path.splitext(os.path.basename(img))[0]])
            for img in all_imgs
            if os.path.splitext(os.path.basename(img))[0] in mask_by_stem
        ]
        print(f"  {root_dir}: {len(self.pairs)} pairs loaded")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]

        image = cv2.imread(img_path)               # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # → RGB
        mask  = cv2.imread(msk_path)               # BGR

        label = mask_to_index(mask)                # (H, W) int64, 0~3

        # 이진 모드: Fair(1)/Poor(2)/Severe(3) → Corrosion(1), Good(0)은 그대로
        if self.num_classes == 2:
            label = (label > 0).astype(np.int64)

        if self.augment and self.aug:
            result = self.aug(image=image, mask=label.astype(np.int32))
            image  = result['image']
            label  = result['mask'].astype(np.int64)

        # SegformerImageProcessor: 정규화 + pixel_values 반환
        encoded = self.image_processor(
            images=image,
            return_tensors='pt',
        )
        pixel_values = encoded['pixel_values'].squeeze(0)  # (3, H, W)
        labels = torch.tensor(label, dtype=torch.long)     # (H, W)

        return {'pixel_values': pixel_values, 'labels': labels}


def get_dataloaders(data_dir, image_processor, batch_size=4, num_classes=4):
    train_ds = CorrosionDataset(
        os.path.join(data_dir, 'Train'), image_processor,
        augment=True, num_classes=num_classes)
    test_ds  = CorrosionDataset(
        os.path.join(data_dir, 'Test'),  image_processor,
        augment=False, num_classes=num_classes)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True)
    test_loader  = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False,
        num_workers=4, drop_last=False)

    return {'Train': train_loader, 'Test': test_loader}

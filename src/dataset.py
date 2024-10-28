import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split

from config import config as cfg

# batch size
BATCH_SIZE = cfg.batch_size

# the training transforms
train_transform = transforms.Compose([
    transforms.Resize(cfg.img_size),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(90, 90)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# the validation transforms
valid_transform = transforms.Compose([
    transforms.Resize(cfg.img_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

dataset = datasets.ImageFolder(
    root=cfg.data_dir
)

# Разделяем на тренировочную и валидационную выборки
train_size = int((1 - cfg.val_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Применяем трансформации отдельно для каждой выборки
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = valid_transform

# training data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=4, pin_memory=True)

# validation data loaders
valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=1, pin_memory=True)

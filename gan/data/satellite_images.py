# download data by
# `kaggle datasets download mahmoudreda55/satellite-image-classification`
# and unzip it
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


def load_dataset(path: str, test_size=0.2, seed=1) -> tuple[Subset, Subset]:
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    data = datasets.ImageFolder(path, transform)

    train_data, test_data = random_split(
        data,
        [(1 - test_size), test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    return train_data, test_data


def get_dataloaders(
    train_data: Dataset, test_data: Dataset, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_data, _ = load_dataset("gan/data/satellite_images")

    img, label = train_data[10]

    print(len(train_data))
    print(train_data[0])
    print(img.permute(1, 2, 0).shape)

    plt.imshow(img.permute(1, 2, 0))
    plt.title(label)
    plt.show()

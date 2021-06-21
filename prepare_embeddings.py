from torch.utils.data import Dataset
from torch import nn
import torch
import os
import tqdm
from PIL import Image
import time
import requests
import io
import models
import torchvision as tv
import numpy as np

CROP = (178, 218)
MODEL = "BiT-M-R50x1"

transform = tv.transforms.Compose([
        tv.transforms.Resize(CROP),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

class CelebA(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.images = os.listdir(root)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.images[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return os.path.basename(image_path), image


def prepare_weights(variant):
    print("Downloading Weights...")
    response = requests.get(f'https://storage.googleapis.com/bit_models/{variant}.npz')
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))


class Model(nn.Module):
    def __init__(self, load_weights=True):
        super().__init__()

        self.model = models.KNOWN_MODELS[MODEL]()
        if load_weights:
            weights = prepare_weights(MODEL)
            self.model.load_from(weights)
            print("Done...")

    def forward(self, x):
        x = self.model.root(x)
        x = self.model.body(x)
        x = self.model.head[0](x)
        x = self.model.head[1](x)
        x = self.model.head[2](x)

        return x


def prepare_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # TODO: If folder is not there, download
    dataset = CelebA("img_align_celeba", transform=transform)

    model = Model(load_weights=True)
    model = model.to(device)

    start = time.time()
    model.eval()

    embeddings = dict()

    for name, img in tqdm.tqdm(dataset):
        with torch.no_grad():
            img = img.to(device)
            img = img.unsqueeze(0)
            
            logits = model(img)
            logits = logits.squeeze()

            embeddings[name] = logits.cpu()

    torch.save(embeddings, "embeddings.pt")
        

if __name__ == "__main__":
    prepare_embeddings()

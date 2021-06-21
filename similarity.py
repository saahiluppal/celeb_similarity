import torch
import os
import torchvision as tv
from PIL import Image
from prepare_embeddings import Model
import tqdm
from scipy import spatial
import matplotlib.pyplot as plt
import shutil
import requests
import zipfile
import argparse

CROP = (178, 218)

transform = tv.transforms.Compose([
        tv.transforms.Resize(CROP),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def load_embeddings():
    print("=> Loading Embeddings...")

    if not os.path.exists("embeddings.pt"):
        print("=> Downloading Embeddings")
        url = "https://github.com/saahiluppal/similar/releases/download/v1.0/embeddings.pt"
        response = requests.get(url, stream=True)
        with open("embeddings.pt", "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

    embeddings = torch.load("embeddings.pt")
    return embeddings

def process_image(image_path):
    print("=> Pre-Processing Image...")
    image = Image.open(image_path)
    image = transform(image)
    return image


def similarity(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if not os.path.exists("img_align_celeba"):
        print("=> Downloading Dataset...")
        url = "https://github.com/saahiluppal/similar/releases/download/v1.0/img_align_celeba.zip"
        response = requests.get(url, stream=True)
        with open("img_align_celeba.zip", "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

        print("=> Extracting Dataset...")
        with zipfile.ZipFile("img_align_celeba.zip", "r") as zip_ref:
            zip_ref.extractall()


    embeddings = load_embeddings()
    image = process_image(image_path)
    model = Model(load_weights=True)
    model = model.to(device)

    score = dict()

    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        logits = model(image)
        logits = logits.cpu().squeeze()

    print("=> Comparing...")
    for name, emb in tqdm.tqdm(embeddings.items()):
        cos = 1 - spatial.distance.cosine(logits, emb)
        score[name] = cos

    sorted_score = dict(sorted(score.items(), key=lambda item: item[1], reverse=True))
    score_iterator = sorted_score.__iter__()

    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 3
    for i in range(1, columns * rows + 1):
        image_file = next(score_iterator)
        image_path = os.path.join("img_align_celeba", image_file)
        img = Image.open(image_path)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

    
parser = argparse.ArgumentParser(description="Find Similar Images")
parser.add_argument('--image', required=True, help="[/path/to/image]")
args = parser.parse_args()
similarity(args.image)

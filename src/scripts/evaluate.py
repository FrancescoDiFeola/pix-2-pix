import torch
from src.utils.utils_checkpoint import save_checkpoint, load_checkpoint
import torch.nn as nn
import torch.optim as optim
import src.utils.config as config
from src.utils.util_data import MapDataset, save_some_examples
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image


def test_fn(gen, loader, folder):

    gen.eval()
    with torch.no_grad():
        loop = tqdm(loader, leave=True)
        for idx, (x, y) in enumerate(loop):
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5  # remove normalization#
            save_image(y_fake, folder + f"/y_gen_{idx}.png")
            save_image(x * 0.5 + 0.5, folder + f"/input_{idx}.png")
            save_image(y * 0.5 + 0.5, folder + f"/label_{idx}.png")
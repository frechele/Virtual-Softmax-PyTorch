import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import pytorch_lightning as pl

from virtsm.models.metrics import build_loss


class CIFAR10Module(pl.LightningModule):
    def __init__(self, model_type, **kargs):
        super(CIFAR10Module, self).__init__()

        self.model = model_type(3, [32, 64, 128], 10)
        self.args = kargs

        self.criteria = build_loss(kargs['loss'])()

    @property
    def train_dataset(self) -> CIFAR10:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        return CIFAR10('cifar10', train=True, transform=transform, download=True)

    @property
    def val_dataset(self) -> CIFAR10:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        return CIFAR10('cifar10', train=False, transform=transform, download=True)

    def training_step(self, batch, batch_idx):
        inp, target = batch

        pred = self.model(inp)

        loss = self.criteria(pred, target)

        acc = torch.mean((torch.argmax(pred, dim=1) == target).float())

        self.log('train_acc', acc.item())
        self.log('train_loss', loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        inp, target = batch

        pred = self.model(inp)

        loss = self.criteria(pred, target)

        acc = torch.mean((torch.argmax(pred, dim=1) == target).float())

        self.log('val_acc', acc.item(), prog_bar=True)
        self.log('val_loss', loss.item(), prog_bar=True)

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=2e-5)
        sch = CosineAnnealingWarmRestarts(
            opt, T_0=5, eta_min=1e-10, last_epoch=-1)
        return [opt], [sch]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

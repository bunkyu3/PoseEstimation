import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd

from src.data.dataset import BinarySegmentationDataset
from src.models.unet import UNet
from src.utils.log import MetricLogger, ConfigLogger, BestModelLogger, ValImageLogger

def train_one_epoch(manager, epoch, model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for _, _, xx, yy in dataloader:
        optimizer.zero_grad()
        yy_hat = model(xx)
        loss = criterion(yy_hat, yy)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(dataloader)
    manager.run(MetricLogger("train_loss", train_loss, epoch))


def evaluate(cfg, manager, epoch, model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xx_paths, yy_paths, xx, yy in dataloader:
            yy_hat = model(xx)
            loss = criterion(yy_hat, yy)
            total_loss += loss.item()
            yy_hat_paths = [path.replace("_y.png", "_y_hat.png") for path in yy_paths]
            for x_path, y_path, y_hat_path, x, y, y_hat in zip(xx_paths, yy_paths, yy_hat_paths, xx, yy, yy_hat):
                manager.run(ValImageLogger(cfg, epoch, x_path, x))
                manager.run(ValImageLogger(cfg, epoch, y_path, y))
                manager.run(ValImageLogger(cfg, epoch, y_hat_path, y_hat))
        val_loss = total_loss / len(dataloader)
        manager.run(MetricLogger("val_loss", val_loss, epoch))


def train_model(cfg, manager):
    # データの取得
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    df = pd.read_csv(cfg.local.read_loc.csv)
    train_df = df[df["mode"] == "train"]
    val_df = df[df["mode"] == "val"]
    train_dataset = BinarySegmentationDataset(cfg.local.read_loc.rawdata_dir, train_df, transform=transform)
    val_dataset = BinarySegmentationDataset(cfg.local.read_loc.rawdata_dir, val_df, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.hparam.train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.hparam.val_batch_size, shuffle=False)

    # ネットワークと学習設定
    model = UNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.hparam.learning_rate)
    # 学習
    for epoch in range(cfg.hparam.num_epochs):
        print(epoch)
        train_one_epoch(manager, epoch, model, train_loader, criterion, optimizer)
        evaluate(cfg, manager, epoch, model, val_loader, criterion)
    # ログ
    manager.run(ConfigLogger(cfg))
    manager.run(BestModelLogger(cfg, model))
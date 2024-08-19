from omegaconf import OmegaConf
import os
import torch.optim as optim
from log import *
from utils import *
from Segmentation.model.unet import *
from data.data_loader import *


def evaluate(model, dataloader):
    model.eval()
    total_corrects = 0
    num_samples = len(dataloader.dataset)
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_corrects += (predicted == labels).sum().item()
    accuracy = total_corrects / num_samples
    print(accuracy)


def test(cfg):
    # データの取得
    test_loader = create_test_dataloader(cfg)
    # ネットワークと学習設定
    model = SimpleFCNN(input_size=28*28, hidden_size=128, output_size=10)
    model.load_state_dict(torch.load(cfg.save_dir.local.best_model))
    evaluate(model, test_loader)
    return model


if __name__ == '__main__':
    # ワーキングディレクトリの設定
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    #乱数設定、ログ設定  
    set_seed(42)
    # config読み込み
    cfg = OmegaConf.load("./config/config.yaml")
    # Mlflow無効化
    set_enable_mlflow(False)
    test(cfg)

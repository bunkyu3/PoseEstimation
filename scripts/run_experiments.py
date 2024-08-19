import os
import sys

import hydra
import mlflow
from omegaconf import DictConfig

sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.log import LoggerManager, DeviceLogger, HparamLogger
from src.utils.utils import set_seed
from src.training.train_model import train_model

@hydra.main(config_name="config", version_base=None, config_path="../config")
def rum_experiments(cfg: DictConfig) -> None:
    # ワーキングディレクトリの設定
    project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_directory)

    # MLFLOW_TRACKING_URIを設定
    tracking_dir = os.path.join(project_directory, cfg.mlflow.tracking_dir)
    os.makedirs(tracking_dir, exist_ok=True)
    tracking_uri = f"file:///{tracking_dir.replace(os.sep, '/')}"
    mlflow.set_tracking_uri(tracking_uri)
    
    # MLFLOW開始設定
    if cfg.mlflow.enable:
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        mlflow.end_run()
        mlflow.start_run()
    
    # Log設定・実行、乱数固定、学習
    manager = LoggerManager(enable_mlflow=cfg.mlflow.enable)
    manager.run(DeviceLogger())
    manager.run(HparamLogger(cfg))
    set_seed(42)
    train_model(cfg, manager)

    # MLFLOW終了設定
    if cfg.mlflow.enable:
        mlflow.end_run()
        

if __name__ == '__main__':
    rum_experiments()

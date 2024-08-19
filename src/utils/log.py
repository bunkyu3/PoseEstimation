import os
import mlflow
import torch
from PIL import Image
from abc import ABC, abstractmethod
from omegaconf import OmegaConf
from torchvision.io import write_jpeg, write_png

class BaseLogger(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def log_on_local(self):
        pass

    @abstractmethod
    def log_on_mlflow(self):
        pass


class MetricLogger(BaseLogger):
    def __init__(self, key, value, step):
        self.key = key
        self.value = value
        self.step = step

    def log_on_local(self):
        pass

    def log_on_mlflow(self):
        mlflow.log_metric(self.key, self.value, step=self.step)


class DeviceLogger(BaseLogger):
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.cuda.get_device_name()
        else:
            self.device = "cpu"

    def log_on_local(self):
        pass
    
    def log_on_mlflow(self):
        mlflow.log_param("device", self.device)
        print(f"Saved Using Device on mlflow")


class HparamLogger(BaseLogger):
    def __init__(self, cfg):
        super().__init__(cfg)

    def log_on_local(self):
        pass
    
    def log_on_mlflow(self):
        mlflow.log_params(self.cfg.local.read_loc)
        mlflow.log_params(self.cfg.hparam)
        print(f"Saved hparameter on mlflow")


class ConfigLogger(BaseLogger):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.local_file_path = self.cfg.local.write_loc.config
        self.mlflow_file_path = self.cfg.mlflow.write_loc.config

    def log_on_local(self):
        os.makedirs(os.path.dirname(self.local_file_path), exist_ok=True)
        with open(self.local_file_path, "w") as file:
            OmegaConf.save(config=self.cfg, f=file.name)
        print(f"Saved Config to {self.local_file_path}")

    def log_on_mlflow(self):
        mlflow.log_artifact(self.local_file_path, self.mlflow_file_path)
        print(f"Saved Config on mlflow")
    

class BestModelLogger(BaseLogger):
    def __init__(self, cfg, model):
        super().__init__(cfg)
        self.local_file_path = self.cfg.local.write_loc.best_model
        self.mlflow_file_path = self.cfg.mlflow.write_loc.best_model
        self.model = model

    def log_on_local(self):
        os.makedirs(os.path.dirname(self.local_file_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.local_file_path)
        print(f"Saved Best Model to {self.local_file_path}")

    def log_on_mlflow(self):
        mlflow.log_artifact(self.local_file_path, self.mlflow_file_path)
        print(f"Saved Best Model on mlflow")


class ValImageLogger(BaseLogger):
    def __init__(self, cfg, epoch, file_path, image):
        super().__init__(cfg)
        self.image = image
        self.local_file_path = self.cfg.local.write_loc.valimage_dir + f"/epoch{epoch}/" + file_path
        self.mlflow_file_path = self.cfg.mlflow.write_loc.valimage_dir + f"/epoch{epoch}"

    def log_on_local(self):
        os.makedirs(os.path.dirname(self.local_file_path), exist_ok=True)
        image = (self.image * 255).to(torch.uint8)
        write_png(image, self.local_file_path)

    def log_on_mlflow(self):
        mlflow.log_artifact(self.local_file_path, self.mlflow_file_path)


class LoggerManager:
    def __init__(self, enable_mlflow):
        self.enable_mlflow = enable_mlflow

    def run(self, Logger: BaseLogger):
        Logger.log_on_local()
        if self.enable_mlflow:
            Logger.log_on_mlflow()
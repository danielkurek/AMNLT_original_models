import fire
import json
import torch
from data_amnlt import AMNLTDataset
from smt_trainer import SMT_Trainer
import os

from AMNLT.configs.smt_dan_config.ExperimentConfig import experiment_config_from_dict
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

def main(config_path, checkpoint_path):

    # Check if checkpoint path exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")
    
    with open(config_path, "r") as f:
        config = experiment_config_from_dict(json.load(f))
    
    datamodule = AMNLTDataset(config=config.data)
    dataset_name = config.data.dataset_name.replace('/','-')


    model = SMT_Trainer.load_from_checkpoint(checkpoint_path, weights_only=False)
    model.freeze()
    
    wandb_logger = WandbLogger(project='SMT_AMNLT-Test', group=dataset_name, name=f"SMT_{dataset_name}", log_model=False)

    trainer = Trainer(logger=wandb_logger,
                      precision="16-mixed")

    trainer.test(model, datamodule=datamodule)

def launch(config_path, checkpoint_path):
    main(config_path, checkpoint_path)

if __name__ == "__main__":
    fire.Fire(launch)
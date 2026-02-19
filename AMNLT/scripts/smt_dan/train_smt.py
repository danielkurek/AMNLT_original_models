import fire
import json
import torch
from data_amnlt import AMNLTDataset
from smt_trainer import SMT_Trainer

from AMNLT.configs.smt_dan_config.ExperimentConfig import experiment_config_from_dict
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

def main(config_path, patience):
    
    with open(config_path, "r") as f:
        config = experiment_config_from_dict(json.load(f))
    
    datamodule = AMNLTDataset(config=config.data)
    dataset_name = config.data.dataset_name.replace('/','-')

    max_height, max_width = datamodule.train_set.get_max_hw()
    max_len = datamodule.train_set.get_max_seqlen()

    model_wrapper = SMT_Trainer(maxh=max_height, maxw=max_width, maxlen=max_len, 
                                out_categories=len(datamodule.train_set.w2i), padding_token=datamodule.train_set.w2i["<pad>"], 
                                in_channels=1, w2i=datamodule.train_set.w2i, i2w=datamodule.train_set.i2w, 
                                d_model=256, dim_ff=256, num_dec_layers=8)
    
    wandb_logger = WandbLogger(project='SMT_AMNLT', group=dataset_name, name=f"SMT_{dataset_name}", log_model=False)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/{dataset_name}/", filename=f"{dataset_name}_SMT", 
                                   monitor="val_CER", mode='min',
                                   save_top_k=1, verbose=True)
    early_stopper = EarlyStopping(
            monitor="val_CER",
            min_delta=0.1,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            divergence_threshold=100.00,
            check_on_train_epoch_end=False,
        )

    trainer = Trainer(max_epochs=10000, 
                      check_val_every_n_epoch=5, 
                      logger=wandb_logger, callbacks=[checkpointer, early_stopper],
                      precision="16-mixed")
    
    trainer.fit(model_wrapper,datamodule=datamodule)

    model = SMT_Trainer.load_from_checkpoint(checkpointer.best_model_path, weights_only=False)

    trainer.test(model, datamodule=datamodule)

def launch(config_path, patience = 5):
    main(config_path, patience)

if __name__ == "__main__":
    fire.Fire(launch)
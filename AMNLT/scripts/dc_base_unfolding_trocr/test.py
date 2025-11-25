import gc
import os
import random

import fire
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader

from AMNLT.utils.dc_base_unfolding_utils.dataset import CTCDataset
from AMNLT.models.dc_base_unfolding_trocr_models.model import CTCTrainedCRNN, LightningE2EModelUnfolding

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Deterministic behavior
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def test(
    ds_name,
    checkpoint_path,
    model_name,
    encoding_type="char",
    project="AMNLT",
    group="Baseline-CharacterLevel",
    entity="el_iseo",
    ctc="greedy",
    num_workers=2,
):
    gc.collect()
    torch.cuda.empty_cache()

    # Check if checkpoint path exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")


    # Experiment info
    print(f"Running experiment: {project} - {group}")
    print(f"\tTest dataset: {ds_name}")
    print(f"\tCheckpoint path ({ds_name}): {checkpoint_path}")
    print(f"\tEncoding type: {encoding_type}")

    # Dataset
    test_ds = CTCDataset(
        ds_name=ds_name,
        split="test",
        model_name=model_name,
        train=False,
        encoding_type=encoding_type,
    )
    
    if model_name == "cnnt2d":
        mh, mw = test_ds.get_mx_hw()
    else:
        mh, mw = 0, 0
        
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=num_workers
    )  # prefetch_factor=2

    # Model
    if(model_name == "crnn"):
        model = CTCTrainedCRNN.load_from_checkpoint(
            checkpoint_path, ctc=ctc, ytest_i2w=test_ds.i2w, ds_name=ds_name
        )
    elif(model_name == "fcn" or model_name == "crnnunfolding" or model_name == "cnnt2d" or model_name == "van"):
        model = LightningE2EModelUnfolding.load_from_checkpoint(
            checkpoint_path, ytest_i2w=test_ds.i2w, model=model_name, mh=mh, mw=mw
        )
    model.freeze()

    # Test: automatically auto-loads the best weights from the previous run
    run_name = f"{encoding_type.upper()}-Test-{ds_name}"
    trainer = Trainer(
        logger=WandbLogger(
            project=project,
            group=group,
            name=run_name,
            log_model=False,
            entity=entity,
        ),
        precision="16-mixed",
    )
    trainer.test(model, dataloaders=test_loader)    


if __name__ == "__main__":
    fire.Fire(test)
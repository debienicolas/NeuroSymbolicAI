import sys
import os

# Add the directory containing main_script.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
## need to fix the above

from sympy import Sum, im
from nesy.model import NeSyModel, MNISTEncoder
from example.dataset import AdditionTask
from nesy.logic import ForwardChaining
from nesy.semantics import Semantics, SumProductSemiring,ProductTNorm, LukasieviczTNorm, GodelTNorm

from pytorch_lightning.loggers import WandbLogger

import torch
import pytorch_lightning as pl
import wandb


task_train = AdditionTask(n_classes=2)
task_test = AdditionTask(n_classes=2, train=False)

neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

SEMANTICS = [SumProductSemiring(), ProductTNorm(), LukasieviczTNorm(), GodelTNorm()]
for sem in SEMANTICS:
    model = NeSyModel(program=task_train.program,
                      logic_engine=ForwardChaining(),
                      neural_predicates=neural_predicates,
                      label_semantics=sem)
    wandb_logger = WandbLogger(project='nesy')
    batch_size = 256
    max_epochs = 2
    wandb_logger.experiment.config.update({"epochs": max_epochs, "batch_size": batch_size, "semantics": sem.__class__.__name__})
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu", logger=wandb_logger)
    trainer.fit(model=model,
                train_dataloaders=task_train.dataloader(batch_size=batch_size),
                val_dataloaders=task_test.dataloader(batch_size=batch_size))
    wandb_logger.finalize("success")
    wandb.finish()

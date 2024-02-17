import sys
import os

# Add the directory containing main_script.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
## need to fix the above

from nesy.model import NeSyModel, MNISTEncoder
from example.dataset import AdditionTask
from nesy.logic import ForwardChaining
from nesy.semantics import SumProductSemiring,ProductTNorm

from pytorch_lightning.loggers import WandbLogger

import torch
import pytorch_lightning as pl
import wandb

for i in range(2,6):
    task_train = AdditionTask(n_classes=i)
    task_test = AdditionTask(n_classes=i, train=False)

    neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

    model = NeSyModel(program=task_train.program,
                    logic_engine=ForwardChaining(),
                    neural_predicates=neural_predicates,
                    label_semantics=SumProductSemiring())

    wandb_logger = WandbLogger(project='nesy',name=f"nesy_classes_strat{i}")

    max_epochs = 10
    batch_size = 256
    wandb_logger.experiment.config["epochs"] = max_epochs
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["semantics"] = SumProductSemiring().__class__.__name__
    wandb_logger.experiment.config["train_examples"] = task_train.nr_examples
    wandb_logger.experiment.config["test_examples"] = task_test.nr_examples
    wandb_logger.experiment.config["n_classes"] = i

    # check the balance of the classes
    for j in range(i):
        print(f"Class {j} has {task_train.original_images[task_train.original_targets==j].shape[0]} examples")

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu", logger=wandb_logger)
    #trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu")
    trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=batch_size),
            val_dataloaders=task_test.dataloader(batch_size=batch_size))
    wandb_logger.finalize("success")
    wandb.finish()
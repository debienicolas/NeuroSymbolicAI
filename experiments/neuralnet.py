import sys
import os

# Add the directory containing main_script.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
## need to fix the above

from sympy import Sum, im
from nesy.model import NeSyModel, MNISTEncoder,MNISTEncoderConv, MNISTEncoderLarge
from example.dataset import AdditionTask
from nesy.logic import ForwardChaining
from nesy.semantics import Semantics, SumProductSemiring,ProductTNorm, LukasieviczTNorm, GodelTNorm

from pytorch_lightning.loggers import WandbLogger

import torch
import pytorch_lightning as pl
import wandb

for i in range(3,6):
    i = 5
    n = 2
    n_classes = i
    task_train = AdditionTask(n = n,n_classes=n_classes)
    task_test = AdditionTask(n=n,n_classes=n_classes, train=False)

    ## MLP Large network
    network = MNISTEncoderLarge(task_train.n_classes)
    max_epochs = 10
    print("Convolutional network")
    wandb_logger = WandbLogger(project='nesy',name=f"MLP_Large_{i}")

    
    neural_predicates = torch.nn.ModuleDict({"digit": network})

    model = NeSyModel(program=task_train.program,
                    logic_engine=ForwardChaining(),
                    neural_predicates=neural_predicates,
                    label_semantics=SumProductSemiring())
    batch_size = 128
    wandb_logger.experiment.config["epochs"] = max_epochs
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["semantics"] = SumProductSemiring().__class__.__name__
    wandb_logger.experiment.config["train_examples"] = task_train.nr_examples
    wandb_logger.experiment.config["test_examples"] = task_test.nr_examples
    wandb_logger.experiment.config["n_classes"] = n_classes
    wandb_logger.experiment.config["network"] = "MLP" if isinstance(network,MNISTEncoder) else "Convolutional"
    wandb_logger.experiment.config["n_arguments"] = n

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu", logger=wandb_logger)
    #trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu")
    trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=batch_size),
            val_dataloaders=task_test.dataloader(batch_size=batch_size))
    wandb_logger.finalize("success")
    wandb.finish()

    ## CNN network
    network = MNISTEncoderConv(task_train.n_classes)
    max_epochs = 10
    print("Convolutional network")
    wandb_logger = WandbLogger(project='nesy',name=f"conv_{i}")

    
    neural_predicates = torch.nn.ModuleDict({"digit": network})

    model = NeSyModel(program=task_train.program,
                    logic_engine=ForwardChaining(),
                    neural_predicates=neural_predicates,
                    label_semantics=SumProductSemiring())
    batch_size = 128
    wandb_logger.experiment.config["epochs"] = max_epochs
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["semantics"] = SumProductSemiring().__class__.__name__
    wandb_logger.experiment.config["train_examples"] = task_train.nr_examples
    wandb_logger.experiment.config["test_examples"] = task_test.nr_examples
    wandb_logger.experiment.config["n_classes"] = n_classes
    wandb_logger.experiment.config["network"] = "MLP" if isinstance(network,MNISTEncoder) else "Convolutional"
    wandb_logger.experiment.config["n_arguments"] = n

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu", logger=wandb_logger)
    #trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu")
    trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=batch_size),
            val_dataloaders=task_test.dataloader(batch_size=batch_size))
    wandb_logger.finalize("success")
    wandb.finish()

    ## MLP network
    network = MNISTEncoder(task_train.n_classes)
    max_epochs = 10
    print("MLP network")
    wandb_logger = WandbLogger(project='nesy',name=f"mlp_{i}")
        
    neural_predicates = torch.nn.ModuleDict({"digit": network})

    model = NeSyModel(program=task_train.program,
                    logic_engine=ForwardChaining(),
                    neural_predicates=neural_predicates,
                    label_semantics=SumProductSemiring())

    batch_size = 128
    wandb_logger.experiment.config["epochs"] = max_epochs
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["semantics"] = SumProductSemiring().__class__.__name__
    wandb_logger.experiment.config["train_examples"] = task_train.nr_examples
    wandb_logger.experiment.config["test_examples"] = task_test.nr_examples
    wandb_logger.experiment.config["n_classes"] = n_classes
    wandb_logger.experiment.config["network"] = "MLP" if isinstance(network,MNISTEncoder) else "Convolutional"
    wandb_logger.experiment.config["n_arguments"] = n

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu", logger=wandb_logger)
    #trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu")
    trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=batch_size),
            val_dataloaders=task_test.dataloader(batch_size=batch_size))
    wandb_logger.finalize("success")
    wandb.finish()


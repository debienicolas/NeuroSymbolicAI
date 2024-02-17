from nesy.model import NeSyModel, MNISTEncoder
from dataset import AdditionTask
from nesy.logic import ForwardChaining
from nesy.semantics import SumProductSemiring,ProductTNorm,LukasieviczTNorm, GodelTNorm

from pytorch_lightning.loggers import WandbLogger

import torch
import pytorch_lightning as pl

import semantics

task_train = AdditionTask(n_classes=2)
task_test = AdditionTask(n_classes=2, train=False)

print("Number of training examples: ",task_train.nr_examples)
print("Number of testing examples: ",task_test.nr_examples)

print("Shape of training images: ",task_train.original_images.shape)
print("Shape of testing images: ",task_test.original_images.shape)


neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

semantic = LukasieviczTNorm()

model = NeSyModel(program=task_train.program,
                  logic_engine=ForwardChaining(),
                  neural_predicates=neural_predicates,
                  label_semantics=semantic)

print("Init the training")
wandb_logger = WandbLogger(project='nesy')

max_epochs = 10
batch_size = 256

if True:    
    wandb_logger.experiment.config["epochs"] = max_epochs
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["semantics"] = semantic.__class__.__name__
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu", logger=wandb_logger)
    
else:
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu")
trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=batch_size),
            val_dataloaders=task_test.dataloader(batch_size=batch_size))




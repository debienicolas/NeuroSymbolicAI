from nesy.model import NeSyModel, MNISTEncoder
from dataset import AdditionTask
from nesy.logic import ForwardChaining
from nesy.semantics import SumProductSemiring,ProductTNorm

import torch
import pytorch_lightning as pl

task_train = AdditionTask(n_classes=2)
task_test = AdditionTask(n_classes=2, train=False)

print("Number of training examples: ",task_train.nr_examples)
print("Number of testing examples: ",task_test.nr_examples)

print("Shape of training images: ",task_train.original_images.shape)
print("Shape of testing images: ",task_test.original_images.shape)


neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

model = NeSyModel(program=task_train.program,
                  logic_engine=ForwardChaining(),
                  neural_predicates=neural_predicates,
                  label_semantics=ProductTNorm())

print("Init the training")
trainer = pl.Trainer(max_epochs=4, accelerator="cpu")
trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=1),
            val_dataloaders=task_test.dataloader(batch_size=1))




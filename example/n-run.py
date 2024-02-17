from nesy.model import NeSyModel, MNISTEncoder
from dataset import AdditionTask
from nesy.logic import ForwardChaining
from nesy.semantics import SumProductSemiring,ProductTNorm

import torch
import pytorch_lightning as pl

task_train = AdditionTask(n_classes=3)
task_test = AdditionTask(n_classes=3, train=False)

program = task_train.program
for i in program:
    print(i)


neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

# model = NeSyModel(program=task_train.program,
#                   logic_engine=ForwardChaining(),
#                   neural_predicates=neural_predicates,
#                   label_semantics=SumProductSemiring())

# print("Init the training")
# trainer = pl.Trainer(max_epochs=4, accelerator="cpu")
# trainer.fit(model=model,
#             train_dataloaders=task_train.dataloader(batch_size=64),
#             val_dataloaders=task_test.dataloader(batch_size=64))




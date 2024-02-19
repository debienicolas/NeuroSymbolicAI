from nesy.model import NeSyModel, MNISTEncoder
from dataset import AdditionTask
from nesy.logic import ForwardChaining
from nesy.semantics import SumProductSemiring,ProductTNorm

import torch
import pytorch_lightning as pl

n = 5
n_classes = 2
task_train = AdditionTask(n=n,n_classes=2)
task_test = AdditionTask(n=n,n_classes=2, train=False)

neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

_,queries,_ = task_train[0]
#print(queries)

_,queries,_ = task_test[1]
#print(queries)

model = NeSyModel(program=task_train.program,
                  logic_engine=ForwardChaining(),
                  neural_predicates=neural_predicates,
                  label_semantics=SumProductSemiring())

print("Init the training")
trainer = pl.Trainer(max_epochs=4, accelerator="cpu")
trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=128),
            val_dataloaders=task_test.dataloader(batch_size=128))


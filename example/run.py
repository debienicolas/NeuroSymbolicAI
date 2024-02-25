from nesy.model import NeSyModel, MNISTEncoder
from dataset import AdditionTask
from nesy.logic import ForwardChaining
from nesy.semantics import SumProductSemiring,ProductTNorm,LukasieviczTNorm, GodelTNorm
import torch
import pytorch_lightning as pl


task_train = AdditionTask(n=2,n_classes=2)
task_test = AdditionTask(n=2,n_classes=2, train=False)


neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

semantic = SumProductSemiring()

model = NeSyModel(program=task_train.program,
                  logic_engine=ForwardChaining(),
                  neural_predicates=neural_predicates,
                  label_semantics=semantic)


max_epochs = 10
batch_size = 256

trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu")
trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=batch_size),
            val_dataloaders=task_test.dataloader(batch_size=batch_size))




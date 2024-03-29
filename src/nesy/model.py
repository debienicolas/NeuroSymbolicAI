import re
import time
from tracemalloc import start
from typing import List, Dict
import torch
import pytorch_lightning as pl

import nesy.parser
from nesy.semantics import Semantics
from nesy.term import Clause, Term,Fact
from nesy.logic import LogicEngine
from torch import Tensor, device, nn
from sklearn.metrics import accuracy_score
from nesy.evaluator import Evaluator

class MNISTEncoder(nn.Module):
    def __init__(self, n):
        self.n = n
        super(MNISTEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 30),
            nn.ReLU(),
            nn.Linear(30, n),
            nn.Softmax(-1))

    def forward(self, x):
        #We flatten the tensor
        original_shape = x.shape
        # original_shape = [1,1,28,28]
        n_dims = len(original_shape)
        # n_dims = 4
        x = x.view(-1, 784)
        o =  self.net(x)
        # o.shape = [1,2]
        #We restore the original shape
        o = o.view(*original_shape[0:n_dims-3], self.n)
        # o.shape = [2]
        # output has probability for each class
        return o

class MNISTEncoderLarge(nn.Module):
    def __init__(self, n):
        self.n = n
        super(MNISTEncoderLarge, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 85),
            nn.ReLU(),
            nn.Linear(85, 30),
            nn.ReLU(),
            nn.Linear(30, n),
            nn.Softmax(-1))

    def forward(self, x):
        #We flatten the tensor
        original_shape = x.shape
        # original_shape = [1,1,28,28]
        n_dims = len(original_shape)
        # n_dims = 4
        x = x.view(-1, 784)
        o =  self.net(x)
        # o.shape = [1,2]
        #We restore the original shape
        o = o.view(*original_shape[0:n_dims-3], self.n)
        # o.shape = [2]
        # output has probability for each class
        return o

    
    
class MNISTEncoderConv(nn.Module):
    def __init__(self, n):
        super(MNISTEncoderConv, self).__init__()
        self.n = n
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,kernel_size=9, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(4*12*12, n)


    def forward(self, x):
        #print("Input shape: ", x.shape)
        x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.conv3(x)
        x = x.view(-1)
        output = self.out(x)
        #print("Output shape: ", output.shape)
        # Turn the output into a probability distribution
        output = nn.Softmax(-1)(output)
        return output


class NeSyModel(pl.LightningModule):

    def __init__(self, program : List[Clause],
                 neural_predicates: torch.nn.ModuleDict,
                 logic_engine: LogicEngine,
                 label_semantics: Semantics,
                 learning_rate = 0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_predicates = neural_predicates
        self.logic_engine = logic_engine
        self.label_semantics = label_semantics
        self.program = tuple(program)
        self.learning_rate = learning_rate
        self.bce = torch.nn.BCELoss()
        self.evaluator = Evaluator(neural_predicates=neural_predicates, label_semantics=label_semantics)


    def forward(self, tensor_sources: Dict[str, torch.Tensor],  queries: List[Term] | List[List[Term]]):
        # TODO: Note that you need to handle both the cases of single queries (List[Term]), like during training
        #  or of grouped queries (List[List[Term]]), like during testing.
        #  Check how the dataset provides such queries.
        
        # Test case
        if isinstance(queries[0], list):
            results = []
            for i,query in enumerate(queries):
                and_or_tree = self.logic_engine.reason(self.program, query)
                
                result = self.evaluator.evaluate(tensor_sources, and_or_tree, query, i)
                results.append(result)

            results = torch.stack(results)
            #results = result
        
        # Training case
        else:
            start = time.time()
            and_or_tree = self.logic_engine.reason(self.program, queries)
            self.log("reasoning_time", time.time() - start,on_step=True, on_epoch=True, prog_bar=True)
            start = time.time()
            results = self.evaluator.evaluate(tensor_sources, and_or_tree, queries, index=0, train=True)
            self.log("evaluation_time", time.time() - start,on_step=True,on_epoch=True, prog_bar=True)

        # and_or_tree = self.logic_engine.reason(self.program, queries)
        # results = self.evaluator.evaluate(tensor_sources, and_or_tree, queries)
        return results

    def training_step(self, I, batch_idx):
        tensor_sources, queries, y_true = I
        y_preds = self.forward(tensor_sources, queries)
        loss = self.bce(y_preds.squeeze(), y_true.float().squeeze())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, I, batch_idx):
        tensor_sources, queries, y_true = I
        y_preds = self.forward(tensor_sources, queries)
        accuracy = accuracy_score(y_true, y_preds.argmax(dim=-1))
        self.log("test_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        # log the amount of non neural facts
        self.log("non_neural_facts", len([c for c in self.program if isinstance(c, Fact) and c.weight is None]))
        self.log("neural_facts", len([c for c in self.program if isinstance(c, Fact) and c.weight is not None]))

        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

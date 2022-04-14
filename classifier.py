import argparse
import logging
from math import floor, ceil
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import SubsetRandomSampler

import numpy as np

from torch import nn
from torch.nn import functional as F
from typing import Callable
from torch import optim
from torch.optim.optimizer import Optimizer
#import torchvision.datasets

from datasets import load_from_disk

from tqdm.auto import tqdm


# Argparser --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run classifier")

    parser.add_argument(
        "--noexp-embeddings-filepath", 
        type=str, 
        default="embeddings/noexp_embeddings.pt", 
        help="Location of Apache Arrow NoExp dataset."
    )

    parser.add_argument(
        "--exp-embeddings-filepath", 
        type=str, 
        default="./embeddings/exp_embeddings.pt", 
        help="Location of Apache Arrow Exp dataset."
    )
    
    parser.add_argument(
        "--noexp-dataset-filepath", 
        type=str, 
        default="./dataset/crisis_dataset/noexp/", 
        help="Location of Apache Arrow No Exp dataset."
    )  

    parser.add_argument(
        "--exp-dataset-filepath", 
        type=str, 
        default="./dataset/crisis_dataset/exp/", 
        help="Location of Apache Arrow Exp dataset."
    )

    parser.add_argument(
        '--exp-flag', 
        action='store_true', 
        help="Run ExpBERT"
    )

    parser.add_argument(
        "--percent-dataset", 
        type=float, 
        default=1.0, 
        help="Percentage of the training data to use."
    )

    parser.add_argument(
        "--train-test-split", 
        type=float, 
        default=0.7, 
        help="Percentage of splitting train and test sets."
    )

    parser.add_argument(
        "--num-hidden-layers", 
        default=1, 
        type=int, 
        help="Number of hidden layers to have in the classifer"
    )

    parser.add_argument(
        "--hidden-layer-size", 
        default=100, 
        type=int, 
        help="Number of neurons in a hidden layer"
    )

    parser.add_argument(
        "--learning-rate", 
        default=1e-2, 
        type=float, 
        help="Learning rate"
    )

    parser.add_argument(
        "--sgd-momentum", 
        default=0.9, 
        type=float, 
        help="SGD Momentum"
    )

    parser.add_argument(
        "--num-epochs", 
        type=int, 
        default=10, 
        help="Total number of epochs to perform during training."
    )
    
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="Number of images within each mini-batch",
    )

    parser.add_argument(
        "-j",
        "--worker-count",
        default=cpu_count(),
        type=int,
        help="Number of worker processes used to load data.",
    )

    parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
    )

    parser.add_argument(
        "--print-frequency",
        default=10,
        type=int,
        help="How frequently to print progress to the command line in number of steps",
    )

    args = parser.parse_args()
    return args




# Accuracy -----------------------------------------------
def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)



# Classifier --------------------------------------------
class MLP_1h(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_layer_size: int,
                 output_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, output_size)
        self.activation_fn = activation_fn
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = self.activation_fn(x)
        x = self.l2(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

class MLP_2h(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_layer_size: int,
                 output_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.l3 = nn.Linear(hidden_layer_size, output_size)
        self.activation_fn = activation_fn
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = self.activation_fn(x)
        x = self.l2(x)
        x = self.activation_fn(x)
        x = self.l3(x)
        x = torch.nn.functional.softmax(x)
        return x


class MLP_3h(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_layer_size: int,
                 output_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.l3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.l4 = nn.Linear(hidden_layer_size, output_size)
        self.activation_fn = activation_fn
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = self.activation_fn(x)
        x = self.l2(x)
        x = self.activation_fn(x)
        x = self.l3(x)
        x = self.activation_fn(x)
        x = self.l4(x)
        x = torch.nn.functional.softmax(x)
        return x


# Trainer -----------------------------------------------
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        #features,
        #labels,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        #self.features = features
        #self.labels = labels
        self.criterion = criterion
        self.optimizer = optimizer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        start_epoch: int = 0
    ):
        self.model.train()

        progress_bar = tqdm(range(epochs))

        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()
                
                logits = self.model.forward(batch)
                print(logits.shape)


                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                progress_bar.update(1)

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            if ((epoch + 1) % val_frequency) == 0:
                self.validate() #Run validation set
                self.model.train() #Need to put model back into train mode


    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )
   
    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")



# Custom Dataset -----------------------------------------

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, embeddings_IDs, labels):
        self.embeddings_IDs = embeddings_IDs
        self.labels = labels

  def __len__(self):
        return len(self.embeddings_IDs)

  def __getitem__(self, index):
        #Return subset of embeddings and labels if provided with list
        #of indices
        if type(index) == list:
            embeddings = []
            labels = []
            for val in index:
                embeddings.append(self.embeddings_IDs[val])
                labels.append(self.labels[val])
            return embeddings, labels
        else:
            return self.embeddings_IDs[index], self.labels[index]

       

#Data loader -------------------------------------------
def get_datasets(args):
    with torch.no_grad():
        if args.exp_flag:
            embeddings = torch.load(args.exp_embeddings_filepath)
            raw_datasets = load_from_disk(args.exp_dataset_filepath)
        else:
            embeddings = torch.load(args.noexp_embeddings_filepath)
            raw_datasets = load_from_disk(args.noexp_dataset_filepath)

        labels = raw_datasets['train']['labels']

        dataset = Dataset(embeddings, labels)
        #TRAIN TEST SPLIT

        if args.percent_dataset != 1.0:
            dataset_size = len(dataset)
            dataset_indices = list(range(dataset_size))

            np.random.shuffle(dataset_indices)
            val_split_index = int(np.floor(args.percent_dataset * dataset_size))
            dataset_idx  = dataset_indices[:val_split_index]

            emb_subset, labels_subset = dataset[dataset_idx]
            dataset = Dataset(emb_subset, labels_subset)

        
        #If the split results in equal values e.g. 70 and 30
        if (args.train_test_split * len(dataset)) % 1 == 0:
            train_size = int(args.train_test_split * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size,
                test_size])
        else: #unequal split e.g. 25.2 and 10.79
            train_dataset, test_dataset = random_split(dataset,
            [int(floor(args.train_test_split *
                len(dataset))), int(ceil((1-args.train_test_split)*len(dataset)))])


    return train_dataset, test_dataset



# Main --------------------------------------------------

def main():
    args = parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    train_dataset, test_dataset = get_datasets(args)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=8,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=8,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #Feature count is the number of neurons from BERT
    #For NoExp we have 768 neurons from BERT
    #For ExpBERT we have 768*(num explanations + num textual descriptions)
    feature_count = train_dataset[0][0].shape[0]
    class_count = 9

    # Define the model to optimze
    #model = MLP_1h(feature_count, hidden_layer_size, class_count)
    
    if args.num_hidden_layers == 1:
        model = MLP_1h(feature_count, args.hidden_layer_size, class_count)
    elif args.num_hidden_layers == 2:
        model = MLP_2h(feature_count, args.hidden_layer_size, class_count)
    else:
        model = MLP_3h(feature_count, args.hidden_layer_size, class_count)

    model = model.to(device)


    # The optimizer we'll use to update the model parameters
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    # Now we define the loss function.
    criterion = nn.CrossEntropyLoss() 
    

    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, device
    )

    trainer.train(
        args.num_epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
    )








if __name__ == "__main__":
    main()

import argparse
import logging
from math import floor, ceil
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
import pickle
from pathlib import Path
import os

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

from sklearn.metrics import precision_score, recall_score, f1_score

from datasets import load_from_disk

from tqdm.auto import tqdm


# Argparser --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run classifier")

    # Filepaths ------------------------------------------
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
        "--output-logs", 
        type=str, 
        default="logs", 
        help="Where to store the logs of the model."
    )
    
    parser.add_argument(
        "--output-metrics", 
        type=str, 
        default="metrics", 
        help="Where to store the metrics of the model during training and testing."
    )
    
    #Flags and model variables -----------------------------
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


    #Model hyperparameters -----------------------------------
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
        "--print-frequency",
        default=10,
        type=int,
        help="How frequently to print progress to the command line in number of steps",
    )

    # Other -----------------------------------------------------
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="bert-base-cased", 
        help="Specify the checkpoint of your model e.g. bert-base-cased."
    )

    # ----------------------------------------------------------

    args = parser.parse_args()
    return args




# Explanation helper functions ----------------------------------------------
def get_explanation_type(exp_dataset_filepath):
    if exp_dataset_filepath == "./dataset/crisis_dataset/noexp/" or ("size" in
        exp_dataset_filepath):
        explanation_type = "normal"
    else:
        #e.g. ./dataset/crisis_dataset_few/exp/
        filename = exp_dataset_filepath.split("/")
        idx_explanation = [idx for idx, s in enumerate(filename) if 'crisis_dataset' in s][0]
        explanation_type = filename[idx_explanation].split("_")[-1]

    return explanation_type

def get_filepath_numbered(log_dir, exp_flag, checkpoint, num_epochs,
    percent_dataset, explanation_type):
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.
    """ 
    checkpoint = checkpoint.replace("-","_") 
    if exp_flag:
        tb_log_dir_prefix = (
            f"Exp_{checkpoint}_" 
            f"pd={percent_dataset}_" 
            f"epochs={num_epochs}_" 
            f"explanations={explanation_type}_"
            f"run_"
            )
    else:
        tb_log_dir_prefix = (
            f"NoExp_{checkpoint}_"
            f"pd={percent_dataset}_" 
            f"epochs={num_epochs}_" 
            f"run_"
            )

    i = 0
    while i < 1000:
        #Creates the PosixPath with run iteration appended
        tb_log_dir = log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


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
        x = torch.nn.functional.softmax(x, dim=1)
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
        x = torch.nn.functional.softmax(x, dim=1)
        return x


# Trainer -----------------------------------------------
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.step = 0

    def train(
        self,
        epochs: int,
        print_frequency: int = 20,
        start_epoch: int = 0
    ):
        self.model.train()

        progress_bar = tqdm(range(epochs))
        
        train_metrics_list = []
        val_metrics_list = []


        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()

            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                data_load_end_time = time.time()
                
                logits = self.model.forward(batch)
                #TODO: Add logits for each epoch
                #to a list, so that I can do train metrics
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
            

            #TODO: get train metrics here




            #TODO: get validation metrics and append to val_metrics list 
            val_results_epoch = self.validate() #Run validation set
            val_metrics_list.append(val_results_epoch)
            self.model.train() #Need to put model back into train mode

        #TODO: Save the arrays here
        print(len(val_metrics_list))


        with open(metrics_filepath+'/train.p', 'wb') as fp:
            pickle.dump(train_metrics_list, fp, protocol=pickle.HIGHEST_PROTOCOL)

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
        

        precision = precision_score(results["labels"], results["preds"],
        labels=list(range(9)), average=None, zero_division=0)

        recall = recall_score(results["labels"], results["preds"],
                labels=list(range(9)), average=None, zero_division=0)

        f1 = f1_score(results["labels"], results["preds"],
                labels=list(range(9)), average=None, zero_division=0)

        precision_weighted = precision_score(results["labels"], results["preds"],
                labels=list(range(9)), average="weighted", zero_division=0)

        recall_weighted = recall_score(results["labels"], results["preds"],
                labels=list(range(9)), average="weighted", zero_division=0)

        f1_weighted = f1_score(results["labels"], results["preds"],
                labels=list(range(9)), average="weighted", zero_division=0)

        results = {"accuracy": accuracy, 
                "f1_weighted": f1_weighted,
                "precision_weighted": precision_weighted,
                "recall_weighted": recall_weighted,
                "f1": f1,
                "precision": precision,
                "recall": recall
        }

        average_loss = total_loss / len(self.val_loader)

        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

        return results




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

    #Find explanation type (normal, bad, few, many)
    explanation_type = get_explanation_type(args.exp_dataset_filepath)
    
    if args.checkpoint == "cardiffnlp/twitter-roberta-base":
        checkpoint = args.checkpoint.split("/")[-1]
        logs_filepath = get_filepath_numbered(Path(args.output_logs), args.exp_flag,
            checkpoint, args.num_epochs, args.percent_dataset,
            explanation_type)
    else:
        logs_filepath = get_filepath_numbered(Path(args.output_logs), args.exp_flag,
            args.checkpoint, args.num_epochs, args.percent_dataset,
            explanation_type)

    if not os.path.exists('metrics'):
        os.makedirs('metrics')


    print(logs_filepath)
    exit(0)

    
    #current run is the name used for all visualizations for a specific run
    current_run = logs_filepath.split("/")[-1]
    current_run_number = int(current_run.split("_")[-1])

    metrics_filepath = "./metrics/" + current_run + "/"
    plots_filepath = "./plots/" + current_run + "/"

    if not os.path.exists(metrics_filepath):
        os.makedirs(metrics_filepath)

    if not os.path.exists(plots_filepath):
        os.makedirs(plots_filepath)


    
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
        print_frequency=args.print_frequency,
    )
    



if __name__ == "__main__":
    main()

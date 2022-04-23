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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets import load_from_disk

from tqdm.auto import tqdm

from visualizations import *


# Argparser --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run classifier")

    # Filepaths ------------------------------------------
    parser.add_argument(
        "--noexp-embeddings-filepath", 
        type=str, 
        default="embeddings/noexp_bert-base-cased_embeddings.pt", 
        help="Location of Apache Arrow NoExp dataset."
    )

    parser.add_argument(
        "--exp-embeddings-filepath", 
        type=str, 
        default="./embeddings/exp_normal_bert-base-cased_embeddings.pt", 
        help="Location of Exp embeddings torch file."
    )
    
    parser.add_argument(
        "--noexp-dataset-filepath", 
        type=str, 
        default="./dataset/crisis_dataset/noexp/", 
        help="Location of NoExp embeddings torch file."
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


def get_filepath_numbered(log_dir, exp_flag, checkpoint, num_epochs,
    percent_dataset, explanation_type, num_hidden_layers):
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter."""
    checkpoint = checkpoint.replace("-","_") 

    
    if exp_flag:
        tb_log_dir_prefix = (
            f"Exp_{checkpoint}_" 
            f"pd={percent_dataset}_" 
            f"epochs={num_epochs}_" 
            f"explanations={explanation_type}_"
            f"hidden={num_hidden_layers}_"
            f"run_"
            )
    else:
        tb_log_dir_prefix = (
            f"NoExp_{checkpoint}_"
            f"pd={percent_dataset}_" 
            f"epochs={num_epochs}_" 
            f"hidden={num_hidden_layers}_"
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

def get_metrics(y_trues, preds):
    accuracy = accuracy_score(y_trues, preds)

    precision = precision_score(y_trues, preds,
        labels=list(range(9)), average=None, zero_division=0)

    recall = recall_score(y_trues, preds,
        labels=list(range(9)), average=None, zero_division=0)

    f1 = f1_score(y_trues, preds,
        labels=list(range(9)), average=None, zero_division=0)

    precision_weighted = precision_score(y_trues, preds,
        labels=list(range(9)), average="weighted", zero_division=0)

    recall_weighted = recall_score(y_trues, preds,
        labels=list(range(9)), average="weighted", zero_division=0)

    f1_weighted = f1_score(y_trues, preds,
        labels=list(range(9)), average="weighted", zero_division=0)

    results = {"accuracy": accuracy, 
            "f1_weighted": f1_weighted,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1": f1,
            "precision": precision,
            "recall": recall
    }
    
    return results

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
        metrics_filepath: str,
        current_run: str,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics_filepath = metrics_filepath
        self.current_run = current_run
        self.step = 0

    def train(
        self,
        epochs: int,
        print_frequency: int = 20,
        start_epoch: int = 0
    ):
        self.model.train()

        progress_bar = tqdm(range(epochs))
        
        train_metrics_total = []
        test_metrics_total = []

        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            train_logits = []
            train_preds = []
            train_labels = []
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                data_load_end_time = time.time()
                
                logits = self.model.forward(batch)

                train_logits.append(logits.detach().cpu().numpy())
                train_labels.append(labels.cpu().numpy())

                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                progress_bar.update(1)

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    train_preds.append(preds.cpu().numpy())
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()
            
            #Arrange them to have correct shape 
            train_preds = np.concatenate(train_preds).ravel()
            train_labels = np.concatenate(train_labels).ravel()
            train_results_epoch = get_metrics(train_labels, train_preds)
            train_metrics_total.append(train_results_epoch)
            print(train_results_epoch)

            #TODO: get validation metrics and append to val_metrics list 
            test_results_epoch = self.validate() #Run validation set
            test_metrics_total.append(test_results_epoch)
            self.model.train() #Need to put model back into train mode
        
        #Save metrics 
        with open(self.metrics_filepath+'/train.p', 'wb') as fp:
            pickle.dump(train_metrics_total, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.metrics_filepath+'/test.p', 'wb') as fp:
            pickle.dump(test_metrics_total, fp, protocol=pickle.HIGHEST_PROTOCOL)

        self.visualizations_model()


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

        results = get_metrics(results["labels"], results["preds"])

        average_loss = total_loss / len(self.val_loader)

        print(f"validation loss: {average_loss:.5f}")
        print(results)

        return results
    
    #Visualizations of model
    def visualizations_model(self):
        train_results = {"preds": [], "labels": [], "preds_all": [],
            "labels_all": []}
        test_results = {"preds": [], "labels": [], "preds_all": [],
            "labels_all": []}

        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                train_results["preds_all"].append(logits.cpu().numpy())
                train_results["labels_all"].append(labels.cpu().numpy())
                preds = logits.argmax(dim=-1).cpu().numpy()
                train_results["preds"].extend(list(preds))
                train_results["labels"].extend(list(labels.cpu().numpy()))

        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                test_results["preds_all"].append(logits.cpu().numpy())
                test_results["labels_all"].append(labels.cpu().numpy())
                preds = logits.argmax(dim=-1).cpu().numpy()
                test_results["preds"].extend(list(preds))
                test_results["labels"].extend(list(labels.cpu().numpy()))

        #Convert predictions to correct format
        predictions_all = np.vstack(train_results["preds_all"]).transpose()
        
        #Convert labels into correct format
        labels_all = np.reshape(np.hstack(train_results["labels_all"]).astype(int), (1,-1))
        labels_all = np.repeat(labels_all, 9, axis=0)
        #predictions_a = np.hstack(predictions)

        l = list(range(0,9))
        # Binarize labels to have 9 arrays, 1 for each label
        # where for class i, the values where i is we have as 1, all other classes
        # -1
        for idx, val in enumerate(l):
            #labels = np.where(labels == val, -1, labels)
            labels_all[idx] = np.where(labels_all[idx] == val, -1, 0)
            labels_all[idx] = np.where(labels_all[idx] == -1, 1, labels_all[idx])


        with open(self.metrics_filepath+'/test_preds.p', 'wb') as fp:
            pickle.dump(test_results["preds"], fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(self.metrics_filepath+'/test_labels.p', 'wb') as fp:
            pickle.dump(test_results["labels"], fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.metrics_filepath+'/test_preds_all.p', 'wb') as fp:
            pickle.dump(predictions_all, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(self.metrics_filepath+'/test_labels_all.p', 'wb') as fp:
            pickle.dump(labels_all, fp, protocol=pickle.HIGHEST_PROTOCOL)


        vis(test_results["preds"], test_results["labels"],
            predictions_all, labels_all, self.current_run)





# Custom Dataset -----------------------------------------

class Dataset(torch.utils.data.Dataset):
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
        else:
            embeddings = torch.load(args.noexp_embeddings_filepath)
        
        #You want labels from NoExp as we have concatenated the 
        #embeddings to be of size (num datapoints, num_exp_td * 768)
        raw_datasets = load_from_disk(args.noexp_dataset_filepath)
        
        #Train includes all datapoints at this point
        labels = raw_datasets['train']['labels']

        dataset = Dataset(embeddings, labels)
        """
        if args.percent_dataset != 1.0:
            dataset_size = len(dataset)
            dataset_indices = list(range(dataset_size))

            np.random.shuffle(dataset_indices)
            val_split_index = int(np.floor(args.percent_dataset * dataset_size))
            dataset_idx  = dataset_indices[:val_split_index]

            emb_subset, labels_subset = dataset[dataset_idx]
            dataset = Dataset(emb_subset, labels_subset)
        """

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

    #Alter embeddings filepath for different checkpoints
    if args.checkpoint == "roberta-base":
        if args.exp_flag:
            args.exp_embeddings_filepath = \
            "embeddings/exp_normal_roberta-base_embeddings.pt"
        else:
            args.noexp_embeddings_filepath = "embeddings/noexp_roberta-base_embeddings.pt"
    elif args.checkpoint == "cardiffnlp/twitter-roberta-base":
        if args.exp_flag:
            args.exp_embeddings_filepath = \
                "embeddings/exp_normal_twitter-roberta-base_embeddings.pt"
        else:
            args.noexp_embeddings_filepath = \
                "embeddings/noexp_twitter-roberta-base_embeddings.pt"
        

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    #Find explanation type (normal, bad, few, many)
    if args.exp_flag:
        explanation_type = args.exp_embeddings_filepath.split("_")[1]
    else:
        explanation_type = "none"


    if args.checkpoint == "cardiffnlp/twitter-roberta-base":
        checkpoint = args.checkpoint.split("/")[-1]
        logs_filepath = get_filepath_numbered(Path(args.output_logs), args.exp_flag,
            checkpoint, args.num_epochs, args.percent_dataset,
            explanation_type, args.num_hidden_layers)
    else:
        logs_filepath = get_filepath_numbered(Path(args.output_logs), args.exp_flag,
            args.checkpoint, args.num_epochs, args.percent_dataset,
            explanation_type, args.num_hidden_layers)

    if not os.path.exists('metrics'):
        os.makedirs('metrics')


    #current run is the name used for all visualizations for a specific run
    current_run = logs_filepath.split("/")[-1]
    current_run_number = int(current_run.split("_")[-1])

    logs_filepath = args.output_logs + "/" + current_run + "/"
    metrics_filepath = "./metrics/" + current_run + "/"
    plots_filepath = "./plots/" + current_run + "/"
   
    if not os.path.exists(logs_filepath):
        os.makedirs(logs_filepath)

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
        model, train_loader, test_loader, criterion, 
            optimizer, device, metrics_filepath, current_run
    )

    trainer.train(
        args.num_epochs,
        print_frequency=args.print_frequency,
    )
    
    

if __name__ == "__main__":
    main()

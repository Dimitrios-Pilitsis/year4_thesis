import argparse
import logging

import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn

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
        "--num-epochs", 
        type=int, 
        default=10, 
        help="Total number of epochs to perform during training."
    )

    parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
    )

    parser.add_argument(
        "--log-frequency",
        default=10,
        type=int,
        help="How frequently to save logs to tensorboard in number of steps",
    )

    parser.add_argument(
        "--print-frequency",
        default=10,
        type=int,
        help="How frequently to print progress to the command line in number of steps",
    )

    # Specific to ADL ------------------------------------------


    args = parser.parse_args()
    return args


# Shuffle and sample ----------------------------------------
#Shuffle data
def shuffle_data(embeddings, labels, percent_dataset):
    vals = torch.arange(0, embeddings.shape[0], dtype=float) #Tensor of
    #indices in order to shuffle and get randomized points from embeddings
    idx = torch.multinomial(vals, num_samples=vals.shape[0], replacement=False)
    embeddings = embeddings[idx]

    #Shuffle labels
    labels = np.array(labels).reshape((-1, 1))
    idx_list = idx.tolist()
    labels = labels[idx_list]

    if percent_dataset != 1.0:
        sample = int(embeddings.shape[0] * percent_dataset)
        embeddings = embeddings[:sample,]
    
    return embeddings, labels


# Arrange tensor into train and test -------------------------------

def train_test_split(embeddings, labels):
    with torch.no_grad():
        train_split = int(0.7*embeddings.shape[0])
        print("Train split")
        print(train_split)
        labels = labels.flatten()

        train_embeddings = embeddings[:train_split].type(torch.float32)
        test_embeddings = embeddings[train_split:].type(torch.float32)

        train_labels = torch.tensor(labels[:train_split], dtype=torch.long)
        test_labels = torch.tensor(labels[train_split:], dtype=torch.long)


        features = {
            'train': train_embeddings,
            'test' : test_embeddings,
        }
        labels = {
            'train': train_labels,
            'test': test_labels,
        }
        """ 
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.worker_count,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.worker_count,
            pin_memory=True,
        )
        """

    return features, labels


# Preprocess overall -------------------------------------
def preprocess(exp_flag, exp_embeddings_filepath, noexp_embeddings_filepath,
    exp_dataset_filepath, noexp_dataset_filepath, percent_dataset):

    with torch.no_grad():
        if exp_flag:
            embeddings = torch.load(exp_embeddings_filepath)
            raw_datasets = load_from_disk(exp_dataset_filepath)
        else:
            embeddings = torch.load(noexp_embeddings_filepath)
            raw_datasets = load_from_disk(noexp_dataset_filepath)

        labels = raw_datasets['train']['labels']

        embeddings, labels = shuffle_data(embeddings, labels, percent_dataset)
        features, labels = train_test_split(embeddings, labels)

    return features, labels


# Classifier helper functions --------------------------------
def accuracy(probs: torch.FloatTensor, labels: torch.LongTensor) -> float:
    """
    Args:
        probs: A float32 tensor of shape ``(batch_size, class_count)`` where each value 
            at index ``i`` in a row represents the score of class ``i``.
        targets: A long tensor of shape ``(batch_size,)`` containing the batch examples'
            labels.
    """
    with torch.no_grad():
        predicted = probs.argmax(dim=1)
        assert len(labels) == len(predicted)
        return float((labels == predicted).sum()) / len(labels)
        
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
class MLP(nn.Module):
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
        return x

# Trainer -----------------------------------------------


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        #train_loader: DataLoader,
        #val_loader: DataLoader,
        features,
        labels,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        #self.train_loader = train_loader
        #self.val_loader = val_loader
        self.features = features
        self.labels = labels
        self.criterion = criterion
        self.optimizer = optimizer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()

        progress_bar = tqdm(range(epochs))
        for epoch in range(0, epochs):
            #logger.info(f"***** Running train set Epoch {epoch + 1}*****")
            logits = self.model.forward(self.features['train']) #Forward pass of network
            loss = self.criterion(logits, self.labels['train']) 
            
            print("epoch: {} train accuracy: {:2.2f}, loss: {:5.5f}".format(
                epoch,
                accuracy(logits, self.labels['train']) * 100,
                loss.item()
            ))
            
            loss.backward() #Compute backward pass, populates .grad attributes
            self.optimizer.step() #Update model parameters using gradients
            self.optimizer.zero_grad() #Zero out the .grad buffers
            progress_bar.update(1)

        """
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                ## TASK 1: Compute the forward pass of the model, print the output shape
                ##         and quit the program
                #outputs = sself.model.forward(batch)
                logits = self.model.forward(batch)
                print(logits.shape)
                #import sys; sys.exit(1)


                ## TASK 9: Compute the loss using self.criterion and
                ##         store it in a variable called `loss`
                loss = self.criterion(logits, labels)

                ## TASK 10: Compute the backward pass
                loss.backward()
                ## TASK 12: Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()
        """

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
        logger.info(f"***** Running test set *****")
        logits = self.model.forward(self.features['test'])    
        test_accuracy = accuracy(logits, self.labels['test']) * 100
        print("test accuracy: {:2.2f}".format(test_accuracy))
 
    """
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
    """
#Data loader -------------------------------------------

def get_data_loaders(embeddings, labels):
    return 1


# Main --------------------------------------------------

def main():
    args = parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    features, labels = preprocess(args.exp_flag,
        args.exp_embeddings_filepath, args.noexp_embeddings_filepath,
        args.exp_dataset_filepath, args.noexp_dataset_filepath,
        args.percent_dataset)
    

    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    """


    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    #Feature count is the number of neurons from BERT
    #768*(num explanations + num textual descriptions)
    feature_count = features['train'][0].shape[0] 
    print(feature_count)
    hidden_layer_size = 100
    class_count = 9

    # Define the model to optimze
    model = MLP(feature_count, hidden_layer_size, class_count)
    model = model.to(device)


    # The optimizer we'll use to update the model parameters
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    # Now we define the loss function.
    criterion = nn.CrossEntropyLoss() 
    

    trainer = Trainer(
        model, features, labels, criterion, optimizer, device
        #model, train_loader, test_loader, criterion, optimizer, device
    )

    trainer.train(
        args.num_epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )








if __name__ == "__main__":
    main()

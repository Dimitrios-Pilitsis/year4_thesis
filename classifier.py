import argparse
import logging

import torch

import numpy as np

from torch import nn
from torch.nn import functional as F
from typing import Callable
from torch import optim

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

    logger.info("***** Running training process *****")
    # Now we iterate over the dataset a number of times. Each iteration of the entire dataset 
    # is called an epoch.
    progress_bar = tqdm(range(args.num_epochs))
    for epoch in range(0, args.num_epochs):
        logger.info(f"***** Running train set Epoch {epoch + 1}*****")
        logits = model.forward(features['train']) #Forward pass of network
        loss = criterion(logits,  labels['train']) 
        
        print("epoch: {} train accuracy: {:2.2f}, loss: {:5.5f}".format(
            epoch,
            accuracy(logits, labels['train']) * 100,
            loss.item()
        ))
        
        print("Before")
        loss.backward() #Compute backward pass, populates .grad attributes
        print("After")
        optimizer.step() #Update model parameters using gradients
        optimizer.zero_grad() #Zero out the .grad buffers
        progress_bar.update(1)


        #logger.info(f"***** Running test set Epoch {epoch + 1}*****")
    
    logger.info(f"***** Running test set *****")
    logits = model.forward(features['test'])    
    test_accuracy = accuracy(logits, labels['test']) * 100
    print("test accuracy: {:2.2f}".format(test_accuracy))
 











if __name__ == "__main__":
    main()

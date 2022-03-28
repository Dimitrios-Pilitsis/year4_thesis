import transformers
from accelerate import Accelerator
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets import load_from_disk, load_metric, DatasetDict
import numpy as np


# Metrics -----------------------------------------------------
def compute_metrics(accelerator, model, dataloader):

    metric1 = load_metric("accuracy")
    metric2 = load_metric("f1")
    metric3 = load_metric("precision")
    metric4 = load_metric("recall")
    metric2_weighted = load_metric("f1")
    metric3_weighted = load_metric("precision")
    metric4_weighted  = load_metric("recall")

    metrics = [metric1, metric2, metric3, metric4, metric2_weighted,
        metric3_weighted, metric4_weighted]

    model.eval()

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        for metric in metrics:
            metric.add_batch(predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch['labels']))

    #Per class metrics
    labels = list(range(0,9))
    f1 = metric2.compute(average=None, labels=labels)["f1"]
    precision = metric3.compute(average=None, labels=labels)["precision"]
    recall = metric4.compute(average=None, labels=labels)["recall"]


    #Weighted metrics
    accuracy = metric1.compute()["accuracy"]
    f1_weighted = metric2_weighted.compute(average="weighted", labels=labels)["f1"]
    precision_weighted = metric3_weighted.compute(average="weighted", labels=labels)["precision"]
    recall_weighted = metric4_weighted.compute(average="weighted", labels=labels)["recall"]

    results = {"accuracy": accuracy, 
                "f1_weighted": f1_weighted,
                "precision_weighted": precision_weighted,
                "recall_weighted": recall_weighted,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }

    return results




# Summary writers ------------------------------------------------------------------

def summary_writer_train(summary_writer, train_metrics, epoch):
    summary_writer.add_scalars("Train averaged statistics",
                                    {"accuracy/train": train_metrics["accuracy"],
                                     "f1_weighted/train": train_metrics["f1_weighted"],
                                     "precision_weighted/train": train_metrics["precision_weighted"],
                                     "recall_weighted/train":
                                     train_metrics["recall_weighted"]},
                                     epoch+1)
     

    f1_train = {}
    precision_train = {}
    recall_train = {}

    for idx, val in enumerate(train_metrics["f1"]):
        f1_train[f'f1_class_{idx}/train'] = val
    
    for idx, val in enumerate(train_metrics["precision"]):
        precision_train[f'precision_class_{idx}/train'] = val
    
    for idx, val in enumerate(train_metrics["recall"]):
        recall_train[f'recall_class_{idx}/train'] = val

    summary_writer.add_scalars("Train f1 per class", f1_train, epoch+1)

    summary_writer.add_scalars("Train precision per class", precision_train,
        epoch+1)

    summary_writer.add_scalars("Train recall per class", recall_train, epoch+1)
                               



def summary_writer_test(summary_writer, test_metrics, epoch):

    summary_writer.add_scalars("Test averaged statistics",
                                 {"accuracy/test": test_metrics["accuracy"],
                                  "f1_weighted/test": test_metrics["f1_weighted"],
                                  "precision_weighted/test": test_metrics["precision_weighted"],
                                  "recall_weighted/test": test_metrics["recall_weighted"]},
                                  epoch+1)

    f1_test = {}
    precision_test = {}
    recall_test = {}

    for idx, val in enumerate(test_metrics["f1"]):
        f1_test[f'f1_class_{idx}/test'] = val
    
    for idx, val in enumerate(test_metrics["precision"]):
        precision_test[f'precision_class_{idx}/test'] = val
    
    for idx, val in enumerate(test_metrics["recall"]):
        recall_test[f'recall_class_{idx}/test'] = val

    summary_writer.add_scalars("Test f1 per class", f1_test, epoch+1)

    summary_writer.add_scalars("Test precision per class", precision_test,
        epoch+1)

    summary_writer.add_scalars("Test recall per class", recall_test, epoch+1)


# Visualizations ---------------------------------------------------------
def summary_writer_pr_curves(accelerator, model, dataloader, current_run):
    labels = []
    predictions = []
    model.eval()

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        softmax = torch.nn.Softmax(dim=0)
        prediction = softmax(logits)
        prediction = accelerator.gather(prediction).detach().cpu().numpy()
        label = accelerator.gather(batch['labels']).detach().cpu().numpy()

        labels.append(label)
        predictions.append(prediction)
       
    #Flattens array even when subarrays aren't of equal dimension (due to batch
    # size)
    labels = np.hstack(np.array(labels, dtype=object))


    l = list(range(0,9))
    #Binarize predictions
    #labels = np.hstack(np.array(labels, dtype=object))
    predictions = np.array(predictions)
    print(predictions)
    predictions = np.reshape(predictions, (-1,
        predictions.shape[2])).transpose()
    predictions = np.split(predictions, 9, axis=0)
    print(len(predictions))
    print(predictions[0].shape)




    # Binarize labels to have 9 arrays, 1 for each label
    final_labels = []
    for val in l:
        x = np.where(labels != val, -1, labels)
        x = np.where(x == val, 1, x)
        x = np.where(x == -1, 0, x)
        final_labels.append(x)


    #Flatten list of numpy arrays
    predictions = np.concatenate(predictions).ravel()

    #Predictions are ready, they are probabilities for y_predo

    #Need to convert labels into correct form for pr_curve()
    print(final_labels)
    #We get 9 arrrays of (80,) where 80 is number of datapoints

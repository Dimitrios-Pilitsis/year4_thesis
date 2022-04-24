from accelerate import Accelerator
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets import load_metric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#NoExp and ExpBERT



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




#NoExp fine tune ------------------------------------------------
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

def summary_writer_metrics(summary_writer, metrics, epoch, train_flag):
    if train_flag:
        current_set = "train"
    else:
        current_set = "test"

    summary_writer.add_scalars(f"{current_set} averaged statistics",
                                 {f"accuracy/{current_set}": metrics["accuracy"],
                                 f"f1_weighted/{current_set}": metrics["f1_weighted"],
                                 f"precision_weighted/{current_set}": metrics["precision_weighted"],
                                 f"recall_weighted/{current_set}": metrics["recall_weighted"]},
                                 epoch+1)

    f1 = {}
    precision = {}
    recall = {}

    for idx, val in enumerate(metrics["f1"]):
        f1[f'f1_class_{idx}/{current_set}'] = val
    
    for idx, val in enumerate(metrics["precision"]):
        precision[f'precision_class_{idx}/{current_set}'] = val
    
    for idx, val in enumerate(metrics["recall"]):
        recall[f'recall_class_{idx}/{current_set}'] = val

    summary_writer.add_scalars(f"{current_set} f1 per class", f1, epoch+1)

    summary_writer.add_scalars(f"{current_set} precision per class", precision,
        epoch+1)

    summary_writer.add_scalars(f"{current_set} recall per class", recall, epoch+1)

import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from itertools import cycle

from accelerate import Accelerator

import torch

from torch.utils.tensorboard import SummaryWriter

# Dataset visualizations -------------------------------------------------
def label_count_plot(filepath, exp_flag):
    data = pd.read_csv(filepath, header=0)
    count_plot = sns.countplot(data['labels'])
    fig = count_plot.get_figure()
    if exp_flag:
        filepath = "./plots/Exp_label_count.png"
        count_plot.set_title('Label count of explanation dataset')
    else:
        filepath = "./plots/NoExp_label_count.png"
        count_plot.set_title('Label count of no explanation dataset')

    fig.savefig(filepath, bbox_inches='tight')


def label_distribution_plot(filepath):
    df = pd.read_csv(filepath, header=0)
    counts = df.groupby('labels').count()/len(df.index) #percentage of each label
    data = (counts['text']*100)
    bar_plot = sns.barplot(list(range(len(data))), data)
    bar_plot.set_title('Label distribution of dataset')
    bar_plot.set_xlabel("Label")
    bar_plot.set_ylabel("Percentage (%)")
    fig = bar_plot.get_figure()
    fig.savefig("./plots/label_distribution.png", bbox_inches='tight')

def visualizations_dataset(noexp_fp, exp_fp):
    label_distribution_plot(noexp_fp)
    label_count_plot(exp_fp, True)
    label_count_plot(noexp_fp, False)





# Model metric visualizations ---------------------------------------------
def get_preds_and_labels(accelerator, model, dataloader):
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

    
    #Flatten them by stacking them vertically, this converts it from 3D to 2D
    #by converting it from (number of batches, batch size, number of classes)
    #to (number of datapoints, number of classes)
    #We then transpose the matrix to get classes on the rows
    predictions = np.vstack(predictions).transpose()
    
    #We use hstack to flatten the array, reshape to convert from vector to
    #matrix, then repeat the matrix 9 times, once per class
    labels = np.reshape(np.hstack(labels).astype(int), (1,-1))
    labels = np.repeat(labels, 9, axis=0)
    #predictions_a = np.hstack(predictions)

    l = list(range(0,9))
    # Binarize labels to have 9 arrays, 1 for each label
    # where for class i, the values where i is we have as 1, all other classes
    # -1
    for idx, val in enumerate(l):
        #labels = np.where(labels == val, -1, labels)
        labels[idx] = np.where(labels[idx] == val, -1, 0)
        labels[idx] = np.where(labels[idx] == -1, 1, labels[idx])

    return predictions, labels






def summary_writer_pr_curves(summary_writer, accelerator, model, dataloader, epoch):
    
    predictions, labels = get_preds_and_labels(accelerator, model, dataloader)

    l = list(range(0,9))
    # Create precision recall curves ----------------------------------
    for val in l:
        summary_writer.add_pr_curve(f"PR curve for class {val}", labels[val],
            predictions[val], epoch+1) 





def create_roc_curves(accelerator, model, dataloader, current_run):
    predictions, labels = get_preds_and_labels(accelerator, model, dataloader)

    n_classes = 9

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(0,n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[i], predictions[i])
    
    #Replace nan with 0 and calculate area under curve (AUC)
    for i in range(0,n_classes):
        fpr[i] = np.nan_to_num(fpr[i], nan=0)
        tpr[i] = np.nan_to_num(tpr[i], nan=0)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Interpolate all ROC curves
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # average values and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="turquoise",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(['black', 'gray', 'r', 'g', 'b', 'orange', 'purple', 'pink',
        'm'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=3,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic (ROC) multiclass plots")

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    filepath = "./plots/" + current_run + "_roc_curve.png"

    #bbox_inches=tight makes sure everything fits in the saved image (including
    #legend)
    plt.savefig(filepath, bbox_inches='tight')





def create_confusion_matrix(accelerator, model, dataloader, current_run, epoch):
    
    true_values = []
    predictions = []
    model.eval()

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)
        prediction = accelerator.gather(prediction).detach().cpu().numpy()
        label = accelerator.gather(batch['labels']).detach().cpu().numpy()
        true_values.append(label)
        predictions.append(prediction)
       
    #Flattens array even when subarrays aren't of equal dimension (due to batch
    # size)
    true_values = np.hstack(true_values).astype(int)
    predictions = np.hstack(predictions).astype(int)

    ConfusionMatrixDisplay.from_predictions(predictions, true_values,
        labels=list(range(0, 9)))
      
    plt.title("Confusion matrix")

    filepath = "./plots/" + current_run + "_confusion_matrix.png"
    plt.savefig(filepath, bbox_inches='tight')
    #plt.show()







def visualizations(summary_writer, accelerator, model, dataloader, current_run,
    epoch):
    
    summary_writer_pr_curves(summary_writer, accelerator, model, dataloader, epoch)
    
    # Individual visualizations
    create_confusion_matrix(accelerator, model, dataloader, current_run, epoch)

    create_roc_curves(accelerator, model, dataloader, current_run)

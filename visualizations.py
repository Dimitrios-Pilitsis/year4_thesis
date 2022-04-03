import torch

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

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





def create_roc_curves(accelerator, model, dataloader, plots_filepath):
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

    filepath = plots_filepath + "roc_curve.png"
    #bbox_inches=tight makes sure everything fits in the saved image (including
    #legend)
    plt.savefig(filepath, bbox_inches='tight')


def create_pr_curves(accelerator, model, dataloader, plots_filepath):
    predictions, labels = get_preds_and_labels(accelerator, model, dataloader)

    n_classes = 9

    # Calculate precision, recall, average precision for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels[i],
            predictions[i], pos_label=1)
   
    #Deal with edge case where first value of recall is nan
    for i in range(n_classes):
        recall[i] = np.nan_to_num(recall[i], nan=1)

        average_precision[i] = average_precision_score(labels[i],
            predictions[i], pos_label=1)
        
        average_precision[i] = np.nan_to_num(average_precision[i], nan=0)


    colors = cycle(['black', 'gray', 'r', 'g', 'b', 'orange', 'purple', 'pink',
        'm'])

    _, ax = plt.subplots()
    
    #iso-f1 curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))


    #PR curves
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            lw=3,
            label="Precision Recall curve of class {0} (area = {1:0.2f})".format(i,
                average_precision[i]),
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision recall plots")


    handles, labels = ax.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])

    plt.legend(handles=handles, labels=labels, loc="center left", bbox_to_anchor=(1, 0.5))

    filepath = plots_filepath + "pr_curve.png"
    #bbox_inches=tight makes sure everything fits in the saved image (including
    #legend)
    plt.savefig(filepath, bbox_inches='tight')



def create_confusion_matrix(accelerator, model, dataloader, plots_filepath):
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

    filepath = plots_filepath + "confusion_matrix.png"
    plt.savefig(filepath, bbox_inches='tight')



def metrics_plots(metrics_filepath, plots_filepath, current_run):
    with open(metrics_filepath + 'train.p', 'rb') as fp:
        metrics_train = pickle.load(fp)
    
    with open(metrics_filepath + 'test.p', 'rb') as fp:
        metrics_test = pickle.load(fp)

    #Rearrange train and test metrics to be based on metric rather than epoch
    accuracy_train = []
    f1_weighted_train = []
    precision_weighted_train = []
    recall_weighted_train = []
    f1_train = []
    precision_train = []
    recall_train = []

    for val in metrics_train:
        accuracy_train.append(val["accuracy"])
        f1_weighted_train.append(val["f1_weighted"])
        precision_weighted_train.append(val["precision_weighted"])
        recall_weighted_train.append(val["recall_weighted"])
        f1_train.append(val["f1"].reshape(9,-1))
        precision_train.append(val["precision"].reshape(9,-1))
        recall_train.append(val["recall"].reshape(9,-1))
   

    f1_train = np.squeeze(np.array(f1_train)).transpose()
    precision_train = np.squeeze(np.array(precision_train)).transpose()
    recall_train = np.squeeze(np.array(recall_train)).transpose()
    
    accuracy_test = []
    f1_weighted_test = []
    precision_weighted_test = []
    recall_weighted_test = []
    f1_test = []
    precision_test = []
    recall_test = []


    for val in metrics_test:
        accuracy_test.append(val["accuracy"])
        f1_weighted_test.append(val["f1_weighted"])
        precision_weighted_test.append(val["precision_weighted"])
        recall_weighted_test.append(val["recall_weighted"])
        f1_test.append(val["f1"].reshape(9,-1))
        precision_test.append(val["precision"].reshape(9,-1))
        recall_test.append(val["recall"].reshape(9,-1))
   
    #Have matrix versions in the form num_classes x num_epochs
    f1_test = np.squeeze(np.array(f1_test)).transpose()
    precision_test = np.squeeze(np.array(precision_test)).transpose()
    recall_test = np.squeeze(np.array(recall_test)).transpose()


    #Variables for plots
    num_epochs = len(metrics_train)
    n_classes = 9
    epochs_graph = list(range(1, num_epochs+1))
    colors = cycle(['black', 'gray', 'r', 'g', 'b', 'orange', 'purple', 'pink',
        'm'])

    #Accuracy plots ------------------------------------------------------
    plt.figure()
    
    plt.plot(
        epochs_graph,
        accuracy_train,
        color="b",
        lw=3,
        label="Train accuracy",
    )
    
    plt.plot(
        epochs_graph,
        accuracy_test,
        color="r",
        lw=3,
        label="Test accuracy",
    )
    
    #plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Epoch")
    plt.ylabel("Percentage") 
    plt.title("Accuracy")
    plt.xticks(np.arange(1, num_epochs+1, 1))
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    filepath = plots_filepath + "accuracy_curves.png"

    #bbox_inches=tight makes sure everything fits in the saved image (including
    #legend)
    plt.savefig(filepath, bbox_inches='tight')

    
    #F1 per class train ------------------------------------------------------
    plt.figure()
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            epochs_graph,
            f1_train[i],
            color=color,
            lw=3,
            label="F1 curve of class {0}".format(i),
        )
    
    plt.plot(
        epochs_graph,
        f1_weighted_train,
        color="gold",
        lw=3,
        label="F1 weighted macro",
    )

    plt.ylim([0.0, 1.05])
    plt.xlabel("Epoch")
    plt.ylabel("F1 value") 
    plt.title("F1 train")
    plt.xticks(np.arange(1, num_epochs+1, 1))
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    filepath = plots_filepath + "f1_train_curves.png"

    plt.savefig(filepath, bbox_inches='tight')

    #F1 per class test ------------------------------------------------------
    plt.figure()
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            epochs_graph,
            f1_test[i],
            color=color,
            lw=3,
            label="F1 curve of class {0}".format(i),
        )
    
    plt.plot(
        epochs_graph,
        f1_weighted_test,
        color="gold",
        lw=3,
        label="F1 weighted macro",
    )

    plt.ylim([0.0, 1.05])
    plt.xlabel("Epoch")
    plt.ylabel("F1 value") 
    plt.title("F1 test")
    plt.xticks(np.arange(1, num_epochs+1, 1))
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    filepath = plots_filepath + "f1_test_curves.png"

    plt.savefig(filepath, bbox_inches='tight')


    #Precision per class train -------------------------------
    plt.figure()

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            epochs_graph,
            precision_train[i],
            color=color,
            lw=3,
            label="Precision curve of class {0}".format(i),
        )
    
    plt.plot(
        epochs_graph,
        precision_weighted_train,
        color="gold",
        lw=3,
        label="Precision weighted macro",
    )

    plt.ylim([0.0, 1.05])
    plt.xlabel("Epoch")
    plt.ylabel("Precision value") 
    plt.title("Precision per class train")
    plt.xticks(np.arange(1, num_epochs+1, 1))
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    filepath = plots_filepath + "precision_train_curves.png"

    plt.savefig(filepath, bbox_inches='tight')
    
    #Precision per class test -------------------------------
    plt.figure()

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            epochs_graph,
            precision_test[i],
            color=color,
            lw=3,
            label="Precision curve of class {0}".format(i),
        )
    
    plt.plot(
        epochs_graph,
        precision_weighted_test,
        color="gold",
        lw=3,
        label="Precision weighted macro",
    )

    plt.ylim([0.0, 1.05])
    plt.xlabel("Epoch")
    plt.ylabel("Precision value") 
    plt.title("Precision per class test")
    plt.xticks(np.arange(1, num_epochs+1, 1))
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    filepath = plots_filepath + "precision_test_curves.png"

    plt.savefig(filepath, bbox_inches='tight')

    #Recall per class train -------------------------------------------
    plt.figure()
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            epochs_graph,
            recall_train[i],
            color=color,
            lw=3,
            label="Recall curve of class {0}".format(i),
        )
    
    plt.plot(
        epochs_graph,
        recall_weighted_train,
        color="gold",
        lw=3,
        label="Recall weighted macro",
    )

    plt.ylim([0.0, 1.05])
    plt.xlabel("Epoch")
    plt.ylabel("Recall value") 
    plt.title("Recall per class train")
    plt.xticks(np.arange(1, num_epochs+1, 1))
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    filepath = plots_filepath + "recall_train_curves.png"

    plt.savefig(filepath, bbox_inches='tight')

    #Recall per class test -------------------------------------------
    plt.figure()
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            epochs_graph,
            recall_test[i],
            color=color,
            lw=3,
            label="Recall curve of class {0}".format(i),
        )
    
    plt.plot(
        epochs_graph,
        recall_weighted_test,
        color="gold",
        lw=3,
        label="Recall weighted macro",
    )

    plt.ylim([0.0, 1.05])
    plt.xlabel("Epoch")
    plt.ylabel("Recall value") 
    plt.title("Recall per class test")
    plt.xticks(np.arange(1, num_epochs+1, 1))
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    filepath = plots_filepath + "recall_test_curves.png"

    plt.savefig(filepath, bbox_inches='tight')




def visualizations(summary_writer, accelerator, model, dataloader,
    current_run, epoch):
    
    metrics_filepath = "./metrics/" + current_run + "/"
    plots_filepath = "./plots/" + current_run + "/"

    summary_writer_pr_curves(summary_writer, accelerator, model, dataloader, epoch)
    
    metrics_plots(metrics_filepath, plots_filepath, plots_filepath)
    # Individual visualizations
    create_confusion_matrix(accelerator, model, dataloader, plots_filepath)

    create_roc_curves(accelerator, model, dataloader, plots_filepath)

    create_pr_curves(accelerator, model, dataloader, plots_filepath)




"""
#To test visualizations without running run.py
metrics_filepath = "./metrics/NoExp_bert_base_cased_pd=1.0_epochs=3_run_0"
plots_filepath = "./plots/NoExp_bert_base_cased_pd=1.0_epochs=3_run_0"
current_run = "NoExp_bert_base_cased_pd=1.0_epochs=3_run_0"
metrics_plots(metrics_filepath, plots_filepath, current_run)
"""

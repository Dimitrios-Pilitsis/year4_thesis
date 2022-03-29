import torch

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

from accelerate import Accelerator

import torch

from torch.utils.tensorboard import SummaryWriter


# Visualizations ---------------------------------------------

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

    print(true_values)
    print(predictions)

    ConfusionMatrixDisplay.from_predictions(predictions, true_values,
        labels=list(range(0, 9)))
    
    filepath = "./plots/" + current_run + "_confusion_matrix.png"
    plt.savefig(filepath, bbox_inches='tight')
    #plt.show()



def summary_writer_pr_curves(summary_writer, accelerator, model, dataloader, epoch):
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
    
    

    # Create precision recall curves ----------------------------------
    for val in l:
        summary_writer.add_pr_curve(f"PR curve for class {val}", labels[val],
            predictions[val], epoch+1) 



def visualizations(summary_writer, accelerator, model, dataloader, current_run,
    epoch):
    
    summary_writer_pr_curves(summary_writer, accelerator, model, dataloader, epoch)
    
    # Individual visualizations
    create_confusion_matrix(accelerator, model, dataloader, current_run, epoch)



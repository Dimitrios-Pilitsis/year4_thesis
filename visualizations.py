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
       
    #Difference with normal way is that we need to deal with the last array
    #seperately as it has a different size to the rest of the arrays
    if predictions[0].shape != predictions[-1].shape:
        pred_remainder = predictions[-1]
        predictions = predictions[:-1]
        
        predictions = np.array(predictions)
        predictions = np.reshape(predictions, (-1,
            predictions.shape[2])).transpose()
        
        pred_remainder = np.array(pred_remainder).transpose()
        
        #Combine the 2 arrays to get (9, number of points in all batches)
        predictions = np.hstack((predictions, pred_remainder))



        label_remainder = labels[-1]
        labels = labels[:-1]
        
        labels = np.hstack(np.array(labels, dtype=int))

        labels = labels.reshape(labels.shape[0], 1) 

        labels = labels.transpose()

        label_remainder = label_remainder.reshape(label_remainder.shape[0], 1) 
        label_remainder = label_remainder.transpose()

        labels =np.hstack((labels, label_remainder))

        labels = np.repeat(labels, 9, axis=0)

    else:
        predictions = np.array(predictions)
        predictions = np.reshape(predictions, (-1,
            predictions.shape[2])).transpose()
        
        labels = np.hstack(np.array(labels, dtype=int))
        #labels = np.hstack(np.array(labels), dtype=int)
        #Transform labels from vector to matrix i.e. (80,) => (80,1)
        labels = labels.reshape(labels.shape[0], 1) 
        #Transpose labels to become (1,80)
        labels = labels.transpose()
  

        #Replicate labels 9 times (once for each class)
        #labels becomes (1,80) => (9,80)
        labels = np.repeat(labels, 9, axis=0)

    """
    # Prediction ---------------------------------------------------------
    #Binarize predictions
    predictions = np.array(predictions)
    #predictions = np.array(predictions, dtype=np.float32)
    #^ should be a 3D array (number of batches, batch size, number of classes)
    #So for training it is (10, 8, 9)
    #Convert then to (10*8,9), then transpose to (9, 10*8) and we done
    
    #ADD DIMENSION HERE IF IT IS 2D OR SKIP BELOW STEP
    
    print(predictions)
    print(predictions.shape)
    

    #DO BELOW FOR CASES WHEN WE HAVE EQUAL NUMBER OF PREDICTION ARRAYS E.G.
    #8,8,8

    #Reshape from 3D to 2D with 9 columns (number of classes)
    #i.e. remove number of batches so that it becomes each individual datapoint
    predictions = np.reshape(predictions, (-1,
        predictions.shape[2])).transpose()

    print(predictions)
    print(predictions.shape)
    #predictions = np.reshape(predictions, (predictions.shape[0] *
    #    predictions.shape[1], predictions.shape[2])).transpose()
    #we also transposed ^ so that we can have it in the form 
    #(number of classes, number of datapoints) so (9,80)
    


    # Label -----------------------------------------------------------
    #Flattens array even when subarrays aren't of equal dimension (due to batch
    # size), becomes numpy array of shape (N,1)
    labels = np.hstack(np.array(labels, dtype=int))
    #labels = np.hstack(np.array(labels), dtype=int)
    #Transform labels from vector to matrix i.e. (80,) => (80,1)
    labels = labels.reshape(labels.shape[0], 1) 
    #Transpose labels to become (1,80)
    labels = labels.transpose()
  

    #Replicate labels 9 times (once for each class)
    #labels becomes (1,80) => (9,80)
    labels = np.repeat(labels, 9, axis=0)
    """ 
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



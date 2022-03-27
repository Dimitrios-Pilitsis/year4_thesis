#from accelerate import Accelerator
import torch

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay 
# Visualizations ---------------------------------------------

def create_confusion_matrix(labels, predictions, current_run):
    ConfusionMatrixDisplay.from_predictions(predictions, labels,
        labels=list(range(0, 9)))
    
    filepath = "./plots/" + current_run + "_confusion_matrix.png"
    plt.savefig(filepath, bbox_inches='tight')
    #plt.show()


def visualizations(accelerator, model, dataloader, current_run):
    labels = []
    predictions = []
    model.eval()

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)
        prediction = accelerator.gather(prediction).detach().cpu().numpy()
        label = accelerator.gather(batch['labels']).detach().cpu().numpy()
        labels.append(label)
        predictions.append(prediction)
       
    #Flattens array even when subarrays aren't of equal dimension (due to batch
    # size)
    labels = np.hstack(np.array(labels, dtype=object))
    predictions = np.hstack(np.array(predictions, dtype=object))

    print(labels)
    print(predictions)
    
    # Individual visualizations
    create_confusion_matrix(labels, predictions, current_run)




import numpy as np


def main():
    preds = np.load('error_analysis/preds.npy')
    labels = np.load('error_analysis/labels.npy')
    idx = np.load('error_analysis/idx.npy')

    print(preds)
    print(labels)
    print(idx)

    datapoint_of_interest = 0 #The index of the datapoint,
    #that we are interested in inspecting, before shuffling
    #in reality this point is picked randomly, unless you
    #go to the original dataset and find a datapoint of interest


    idx_datapoint = np.where(idx==datapoint_of_interest)
    print(idx_datapoint)

    #Issue is that we don't know if its in train or test set, need to note it
    #print(train_dataset[idx_datapoint])



if __name__ == "__main__":
    main()

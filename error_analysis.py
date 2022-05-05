import numpy as np
import argparse





def parse_args():
    parser = argparse.ArgumentParser(description="Run classifier")
    # Filepaths ------------------------------------------
    parser.add_argument(
        "--error-analysis-noexp-filepath", 
        type=str, 
        default="error_analysis/gcp/NoExp_WCEL/", 
        help="Location of NoExp error analysis numpy arrays."
    )
    
    parser.add_argument(
        "--error-analysis-exp-filepath", 
        type=str, 
        default="error_analysis/gcp/ExpBERT_WCEL/", 
        help="Location of ExpBERT error analysis numpy arrays."
    )
    
    args = parser.parse_args()
    return args


def classifications(preds, labels, idx):
    correct = 0
    incorrect = 0

    #Find how many datapoints were correctly and 
    #incorrectly classified by NoExp
    for i in range(0, len(idx)):
        idx_shuffled = np.where(idx==i)[0][0]
        pred = preds[idx_shuffled]
        label = labels[idx_shuffled]

        if pred == label:
            correct += 1
        else:
            incorrect += 1

    return correct, incorrect



def main():
    args = parse_args()
    preds_noexp = np.load(f'{args.error_analysis_noexp_filepath}/preds.npy')
    labels_noexp = np.load(f'{args.error_analysis_noexp_filepath}/labels.npy')
    idx_noexp = np.load(f'{args.error_analysis_noexp_filepath}/idx.npy')

    preds_exp = np.load(f'{args.error_analysis_exp_filepath}/preds.npy')
    labels_exp = np.load(f'{args.error_analysis_exp_filepath}/labels.npy')
    idx_exp = np.load(f'{args.error_analysis_exp_filepath}/idx.npy')


    #This accuracy metric should not be used for analysis as it combines
    #the train and test set together
    print("NoExp accuracy")
    print(np.sum(preds_noexp == labels_noexp)/17117)
    print(np.sum(preds_noexp == labels_noexp))
    
    print("ExpBERT accuracy")
    print(np.sum(preds_exp == labels_exp)/17117)
    print(np.sum(preds_exp == labels_exp))


    #The index of the datapoint that we are interested in 
    #inspecting, before shuffling, in reality this point
    #is picked randomly, unless you


    cc = 0 #points were correct and remained correct
    ic = 0 #points were incorrect and became correct
    ci = 0 #points were correct and became incorrect
    ii = 0

    #Find how many datapoints were correctly and 
    #incorrectly classified by NoExp
    for i in range(0, len(idx_noexp)):
        idx_noexp_shuffled = np.where(idx_noexp==i)[0][0]
        idx_exp_shuffled = np.where(idx_exp==i)[0][0]

        pred_noexp = preds_noexp[idx_noexp_shuffled]
        label_noexp = labels_noexp[idx_noexp_shuffled]

        pred_exp = preds_exp[idx_exp_shuffled]
        label_exp = labels_exp[idx_exp_shuffled]
        
        #If was correct and remained correct
        if (pred_noexp == label_noexp) and \
            (pred_exp == label_exp): 
            cc += 1
        elif (pred_noexp != label_noexp) and \
            (pred_exp == label_exp):
            ic += 1
        elif (pred_noexp == label_noexp) and \
            (pred_exp != label_exp):
            ci += 1
        else:
            ii += 1

    print(cc)
    print(ic)
    print(ci)
    print(ii)
    print(cc+ic+ci+ii)




if __name__ == "__main__":
    main()

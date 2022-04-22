cd ..

#Bad explanations
python3 create_dataset.py --explanations-filepath explanations/explanations_bad.txt --exp-csv-filepath ./dataset/dataset_exp_bad.csv --output-exp-directory ./dataset/crisis_dataset_bad/exp/
python3 create_embeddings.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_bad/exp/ --split-value 150
python3 classifier.py --exp-flag --exp-embeddings-filepath ./embeddings/exp_bad_bert-base-cased_embeddings.pt --exp-dataset-filepath ./dataset/crisis_dataset_bad/exp/

#ONLY DO THE BELOW EXPERIMENTS IF BAD EXPLANATIONS DOES SURPRISINGLY WELL
#Random explanations
python3 create_dataset.py --explanations-filepath explanations/explanations_random.txt --exp-csv-filepath ./dataset/dataset_exp_random.csv --output-exp-directory ./dataset/crisis_dataset_random/exp/
python3 create_embeddings.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_random/exp/ --split-value 150
python3 classifier.py --exp-flag --exp-embeddings-filepath ./embeddings/exp_random_bert-base-caseed_embeddings.pt --exp-dataset-filepath ./dataset/crisis_dataset_random/exp/

#Horrible explanations
python3 create_dataset.py --explanations-filepath explanations/explanations_horrible.txt --exp-csv-filepath ./dataset/dataset_exp_horrible.csv --output-exp-directory ./dataset/crisis_dataset_horrible/exp/
python3 create_embeddings.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_horrible/exp/ --split-value 150
python3 classifier.py --exp-flag --exp-embeddings-filepath ./embeddings/exp_horrible_bert-base-cased_embeddings.pt --exp-dataset-filepath ./dataset/crisis_dataset_horrible/exp/




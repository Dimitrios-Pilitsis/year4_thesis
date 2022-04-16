cd ..


python3 create_dataset.py --explanations-filepath explanations/explanations_few.txt --exp-csv-filepath ./dataset/dataset_exp_few.csv --output-exp-directory ./dataset/crisis_dataset_few/exp/
python3 create_embeddings.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_few/exp/
python3 classifier.py --exp-flag --exp-embeddings-filepath ./embeddings/exp_few_embeddings.pt --exp-dataset-filepath ./dataset/crisis_dataset_few/exp/


python3 create_dataset.py --explanations-filepath explanations/explanations_many.txt --exp-csv-filepath ./dataset/dataset_exp_many.csv --output-exp-directory ./dataset/crisis_dataset_many/exp/
python3 create_embeddings.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_many/exp/
python3 classifier.py --exp-flag --exp-embeddings-filepath ./embeddings/exp_many_embeddings.pt --exp-dataset-filepath ./dataset/crisis_dataset_many/exp/






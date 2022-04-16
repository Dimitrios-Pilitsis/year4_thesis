cd ..
python3 create_dataset.py --explanations-filepath explanations/explanations_horrible.txt --exp-csv-filepath ./dataset/dataset_exp_horrible.csv --output-exp-directory ./dataset/crisis_dataset_horrible/exp/
python3 run.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_horrible/exp/ 


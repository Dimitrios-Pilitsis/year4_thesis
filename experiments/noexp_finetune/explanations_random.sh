cd ..
python3 create_dataset.py --explanations-filepath explanations/explanations_random.txt --exp-csv-filepath ./dataset/dataset_exp_random.csv --output-exp-directory ./dataset/crisis_dataset_random/exp/
python3 run.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_random/exp/ 


cd ..
python3 create_dataset.py --explanations-filepath explanations/explanations_few.txt --exp-csv-filepath ./dataset/dataset_exp_few.csv --output-exp-directory ./dataset/crisis_dataset_few/exp/
python3 run.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_few/exp/ 

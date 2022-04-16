cd ..
python3 create_dataset.py --explanations-filepath explanations/explanations_bad.txt --exp-csv-filepath ./dataset/dataset_exp_bad.csv --output-exp-directory ./dataset/crisis_dataset_bad/exp/
python3 run.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_bad/exp/ 


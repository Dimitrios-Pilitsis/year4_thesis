cd ..

python3 create_dataset.py --dataset-percent 0.01 --explanations-filepath explanations/explanations_few.txt --exp-csv-filepath ./dataset/dataset_1_exp_few.csv --output-exp-directory ./dataset/crisis_dataset_1_exp_few/exp/
python3 run.py --exp-flag --dataset-percent 0.01 --exp-dataset-filepath ./dataset/crisis_dataset_1_exp_few/exp/ 


python3 create_dataset.py --dataset-percent 0.1 --explanations-filepath explanations/explanations_few.txt --exp-csv-filepath ./dataset/dataset_10_exp_few.csv --output-exp-directory ./dataset/crisis_dataset_10_exp_few/exp/
python3 run.py --exp-flag --dataset-percent 0.1 --exp-dataset-filepath ./dataset/crisis_dataset_10_exp_few/exp/ 


python3 create_dataset.py --dataset-percent 0.5 --explanations-filepath explanations/explanations_few.txt --exp-csv-filepath ./dataset/dataset_50_exp_few.csv --output-exp-directory ./dataset/crisis_dataset_50_exp_few/exp/
python3 run.py --exp-flag --dataset-percent 0.5 --exp-dataset-filepath ./dataset/crisis_dataset_50_exp_few/exp/ 

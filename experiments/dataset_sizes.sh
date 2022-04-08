cd ..

python3 create_dataset.py --dataset-percent 0.01 --output-noexp-directory ./dataset/crisis_dataset_size_1/noexp/ --output-exp-directory ./dataset/crisis_dataset_size_1/exp/ --noexp-csv-filepath ./dataset/dataset_noexp_size_1.csv --exp-csv-filepath ./dataset/dataset_exp_1.csv
python3 run.py --noexp-dataset-filepath ./dataset/crisis_dataset_size_1/noexp/ 
python3 run.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_size_1/exp/ 


python3 create_dataset.py --dataset-percent 0.1 --output-noexp-directory ./dataset/crisis_dataset_size_10/noexp/ --output-exp-directory ./dataset/crisis_dataset_size_10/exp/ --noexp-csv-filepath ./dataset/dataset_noexp_size_10.csv --exp-csv-filepath ./dataset/dataset_exp_10.csv
python3 run.py --noexp-dataset-filepath ./dataset/crisis_dataset_size_10/noexp/ 
python3 run.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_size_10/exp/ 


python3 create_dataset.py --dataset-percent 0.5 --output-noexp-directory ./dataset/crisis_dataset_size_50/noexp/ --output-exp-directory ./dataset/crisis_dataset_size_50/exp/ --noexp-csv-filepath ./dataset/dataset_noexp_size_50.csv --exp-csv-filepath ./dataset/dataset_exp_50.csv
python3 run.py --noexp-dataset-filepath ./dataset/crisis_dataset_size_50/noexp/
python3 run.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_size_50/exp/ 


python3 create_dataset.py --dataset-percent 0.75 --output-noexp-directory ./dataset/crisis_dataset_size_75/noexp/ --output-exp-directory ./dataset/crisis_dataset_size_75/exp/ --noexp-csv-filepath ./dataset/dataset_noexp_size_75.csv --exp-csv-filepath ./dataset/dataset_exp_75.csv
python3 run.py --noexp-dataset-filepath ./dataset/crisis_dataset_size_75/noexp/
python3 run.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_size_75/exp/ 

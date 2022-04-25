cd ../..

python3 create_dataset.py --percent-dataset 0.01 --output-noexp-directory ./dataset/crisis_dataset_size_1/noexp/ --output-exp-directory ./dataset/crisis_dataset_size_1/exp/ --noexp-csv-filepath ./dataset/dataset_noexp_size_1.csv --exp-csv-filepath ./dataset/dataset_exp_1.csv
python3 noexp_finetune_run.py --noexp-dataset-filepath ./dataset/crisis_dataset_size_1/noexp/ --percent-dataset 0.01


python3 create_dataset.py --percent-dataset 0.1 --output-noexp-directory ./dataset/crisis_dataset_size_10/noexp/ --output-exp-directory ./dataset/crisis_dataset_size_10/exp/ --noexp-csv-filepath ./dataset/dataset_noexp_size_10.csv --exp-csv-filepath ./dataset/dataset_exp_10.csv
python3 noexp_finetune_run.py --noexp-dataset-filepath ./dataset/crisis_dataset_size_10/noexp/  --percent-dataset 0.1

python3 create_dataset.py --percent-dataset 0.25 --output-noexp-directory ./dataset/crisis_dataset_size_25/noexp/ --output-exp-directory ./dataset/crisis_dataset_size_25/exp/ --noexp-csv-filepath ./dataset/dataset_noexp_size_25.csv --exp-csv-filepath ./dataset/dataset_exp_25.csv
python3 noexp_finetune_run.py --noexp-dataset-filepath ./dataset/crisis_dataset_size_25/noexp/  --percent-dataset 0.25


python3 create_dataset.py --percent-dataset 0.5 --output-noexp-directory ./dataset/crisis_dataset_size_50/noexp/ --output-exp-directory ./dataset/crisis_dataset_size_50/exp/ --noexp-csv-filepath ./dataset/dataset_noexp_size_50.csv --exp-csv-filepath ./dataset/dataset_exp_50.csv
python3 noexp_finetune_run.py --noexp-dataset-filepath ./dataset/crisis_dataset_size_50/noexp/ --percent-dataset 0.5


python3 create_dataset.py --percent-dataset 0.75 --output-noexp-directory ./dataset/crisis_dataset_size_75/noexp/ --output-exp-directory ./dataset/crisis_dataset_size_75/exp/ --noexp-csv-filepath ./dataset/dataset_noexp_size_75.csv --exp-csv-filepath ./dataset/dataset_exp_75.csv
python3 noexp_finetune_run.py --noexp-dataset-filepath ./dataset/crisis_dataset_size_75/noexp/ --percent-dataset 0.75

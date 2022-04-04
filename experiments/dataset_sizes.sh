cd ..
python3 create_dataset.py --dataset-percent 0.1 --output-noexp-directory ./dataset/crisis_dataset_size_0.1/noexp/ --output-exp-directory ./dataset/crisis_dataset_size_0.1/exp/ 
python3 run.py --noexp-dataset-filepath ./dataset/crisis_dataset_size_0.1/noexp/ 
python3 run.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_size_0.1/exp/ 


python3 create_dataset.py --dataset-percent 0.5 --output-noexp-directory ./dataset/crisis_dataset_size_0.5/noexp/ --output-exp-directory ./dataset/crisis_dataset_size_0.5/exp/ 
python3 run.py --noexp-dataset-filepath ./dataset/crisis_dataset_size_0.5/noexp/
python3 run.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_size_0.5/exp/ 


python3 create_dataset.py --dataset-percent 0.75 --output-noexp-directory ./dataset/crisis_dataset_size_0.75/noexp/ --output-exp-directory ./dataset/crisis_dataset_size_0.75/exp/ 
python3 run.py --noexp-dataset-filepath ./dataset/crisis_dataset_size_0.75/noexp/
python3 run.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_size_0.75/exp/ 

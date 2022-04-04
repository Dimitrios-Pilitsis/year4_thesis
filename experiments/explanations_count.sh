cd ..
python3 create_dataset.py --explanations-filepath explanations/explanations_few.txt --output-exp-directory ./dataset/crisis_dataset_few/exp/
python3 run.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_few/exp/ 

python3 create_dataset.py --explanations-filepath explanations/explanations_many.txt --output-exp-directory ./dataset/crisis_dataset_many/exp/
python3 run.py --exp-flag --exp-dataset-filepath ./dataset/crisis_dataset_many/exp/ 


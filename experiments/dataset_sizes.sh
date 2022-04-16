cd ..
python3 create_dataset.py 

python3 create_embeddings.py 
python3 create_embeddings.py --exp-flag 

python3 classifier.py --percent-dataset 0.1 
python3 classifier.py --exp-flag --percent-dataset 0.1 

python3 classifier.py --percent-dataset 0.25
python3 classifier.py --exp-flag --percent-dataset 0.25 

python3 classifier.py --percent-dataset 0.5 
python3 classifier.py --exp-flag --percent-dataset 0.5 

python3 classifier.py --percent-dataset 0.75
python3 classifier.py --exp-flag --percent-dataset 0.75 

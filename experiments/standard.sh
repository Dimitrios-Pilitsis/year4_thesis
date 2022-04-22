cd ..
python3 create_dataset.py 
python3 create_embeddings.py 
python3 create_embeddings.py --exp-flag --split-value 150

python3 classifier.py
python3 classifier.py --exp-flag

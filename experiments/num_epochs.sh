cd ..

python3 create_dataset.py 
python3 create_embeddings.py
python3 create_embeddings.py --exp-flag --split-value 150

#Do a couple of tests to see how it performs, then run num_epochs tests

python3 classifier.py --num-epochs 100
python3 classifier.py --exp-flag --num-epochs 100


python3 classifier.py --num-epochs 1000
python3 classifier.py --exp-flag --num-epochs 1000

cd ..

python3 create_dataset.py 
python3 create_embeddings.py 
python3 create_embeddings.py --exp-flag

python3 classifier.py --num-hidden-layers 2
python3 classifier.py --exp-flag --num-hidden-layers 2

python3 classifier.py --num-hidden-layers 3
python3 classifier.py --exp-flag --num-hidden-layers 3


python3 classifier.py --exp-flag --num-hidden-layers 3 --num-epochs 1000

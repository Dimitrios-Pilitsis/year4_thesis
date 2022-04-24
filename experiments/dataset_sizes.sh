cd ..
python3 create_dataset.py --percent-dataset 0.1

python3 create_embeddings.py 
python3 classifier.py --percent-dataset 0.1 --num-epochs 100

python3 create_embeddings.py --exp-flag --percent-dataset 0.1 --split-value 150
python3 classifier.py --exp-flag --percent-dataset 0.1





python3 create_dataset.py  --percent-dataset 0.25

python3 create_embeddings.py 
python3 classifier.py  --percent-dataset 0.25 --num-epochs 100


python3 create_embeddings.py --exp-flag --percent-dataset 0.25 --split-value 150
python3 classifier.py --exp-flag --percent-dataset 0.25





python3 create_dataset.py  --percent-dataset 0.5

python3 create_embeddings.py 
python3 classifier.py  --percent-dataset 0.5 --num-epochs 100


python3 create_embeddings.py --exp-flag --percent-dataset 0.5 --split-value 150
python3 classifier.py --exp-flag --percent-dataset 0.5





python3 create_dataset.py  --percent-dataset 0.75

python3 create_embeddings.py 
python3 classifier.py  --percent-dataset 0.75 --num-epochs 100


python3 create_embeddings.py --exp-flag --percent-dataset 0.75 --split-value 150
python3 classifier.py --exp-flag --percent-dataset 0.75


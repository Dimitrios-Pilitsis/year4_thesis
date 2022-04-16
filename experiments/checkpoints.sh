cd ..
python3 create_dataset.py 

python3 create_embeddings.py --checkpoint roberta-base
python3 create_embeddings.py --exp-flag --checkpoint roberta-base

python3 create_embeddings.py --checkpoint cardiffnlp/twitter-roberta-base
python3 create_embeddings.py --exp-flag --checkpoint cardiffnlp/twitter-roberta-base


python3 classifier.py --checkpoint roberta-base
python3 classifier.py --exp-flag --checkpoint roberta-base

python3 classifier.py --checkpoint cardiffnlp/twitter-roberta-base
python3 classifier.py --exp-flag --checkpoint cardiffnlp/twitter-roberta-base

cd ..
python3 create_dataset.py 

python3 create_embeddings.py --checkpoint roberta-base
python3 create_embeddings.py --exp-flag --checkpoint roberta-base

python3 create_embeddings.py --checkpoint cardiffnlp/twitter-roberta-base
python3 create_embeddings.py --exp-flag --checkpoint cardiffnlp/twitter-roberta-base




python3 classifier.py --checkpoint roberta-base --noexp-embeddings-filepath embeddings/noexp_roberta-base_embeddings.pt --num-epochs 100
python3 classifier.py --exp-flag --checkpoint roberta-base --exp-embeddings-filepath embeddings/exp_normal_roberta-base_embeddings.pt --num-epochs 400




python3 classifier.py --checkpoint cardiffnlp/twitter-roberta-base --noexp-embeddings-filepath embeddings/noexp_twitter-roberta-base_embeddings.pt --num-epochs 100
python3 classifier.py --exp-flag --checkpoint cardiffnlp/twitter-roberta-base --exp-embeddings-filepath embeddings/exp_normal_twitter-roberta-base_embeddings.pt --num-epochs 400

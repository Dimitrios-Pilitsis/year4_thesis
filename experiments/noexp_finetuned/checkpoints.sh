cd ../..
python3 create_dataset.py
python3 noexp_finetune_run.py --checkpoint roberta-base 
python3 noexp_finetune_run.py --checkpoint cardiffnlp/twitter-roberta-base

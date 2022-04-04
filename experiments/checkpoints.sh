cd ..
python3 run.py --checkpoint roberta-base 
python3 run.py --checkpoint cardiffnlp/twitter-roberta-base 

python3 run.py --exp-flag --checkpoint roberta-base 
python3 run.py --exp-flag --checkpoint cardiffnlp/twitter-roberta-base 

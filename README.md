# Master's Thesis

This repository contains all the code that was written for my Master's Thesis that is a component of my Mathematics and Computer Science (MEng) degree at the University of Bristol.

It contains the original dataset, the experiments that were run in the form of bash scripts, the explanations for ExpBERT, all the plots from running my experiments. 

This repository includes requirement text files so that anyone can set up their enviornment to run the codebase. The requirements text files can be run using the following command `pip3 install -r requirements_gpu.txt` which uses pip to install the packages.

Multiple language models were created to perform document classification on a dataset containing tweets of people affected by natural disasters. The model is based on the state-of-the-art proof-of-concept "ExpBERT" model (derived from the seminal BERT model) that was created by researchers at Stanford University. The model’s classifier uses a list of explanations created by humans to correct its mistakes, similar to how humans learn. 

Two other models were created. "NoExp" is structually very similar to ExpBERT but doesn't use explanations, while "NoExp-finetuned" is a model that instead is finetuned on the dataset. This model was created so that two different model types could be compared.

The original ExpBERT model was disseminated through the following paper:
> Shikhar Murty, Pang Wei Koh, Percy Liang
>
> [ExpBERT: Representation Engineering with Natural Language Explanations]

The dataset is from:
> Muhammad Imran, Prasenjit Mitra, Carlos Castillo: Twitter as a Lifeline: Human-annotated Twitter Corpora for NLP of Crisis-related Messages. In Proceedings of the 10th Language Resources and Evaluation Conference (LREC), pp. 1638-1643. May 2016, Portorož, Slovenia.


To see how to run the pipeline, look at the experiments in the `experiments` directory. To run the pipeline using default parameters, see `standard.sh` for NoExp and ExpBERT, and `direct_comparison.sh` within the directory `noexp-finetuned` for NoExp-finetuned.

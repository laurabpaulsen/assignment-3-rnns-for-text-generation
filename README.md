# Language modelling and text generation using RNNs
Assignment 3 for language analytics (S2023). This repository contains code for training a text generation model on comments from articles from *The New York Times* using recurrent neural networks. Furthermore, a script is provided for generating text from a user-suggested prompt.

## Description of the data
The model is trained on comments from articles from *The New York Times*. The data is available [here](https://www.kaggle.com/datasets/aashita/nyt-comments).

**Note:** The model in this repo was trained on 50,000 randomly chosen comments to avoid memory issues.

## Usage and reproducibility
To reproduce the results follow the directions below. All terminal commands should be executed from the root of the repository.
1. Clone the repository
2. Download the data from Kaggle and place it in the `data` directory (see repository structure section below if in doubt)
3. Create a virtual environment and install the requirements 
```
bash setup.sh
```
4. Train the model by running the following commands from the command-line
```
source ./env/bin/activate
python src/train_model.py --n_comments 50000
```
5. Generate text
```
python src/generate_text.py --prompt "Donald Trump"
```

It is possible to specify the model you want to use for text generation, if you have several in the repository.

## Repository structure
```
├── data                        <- data folder (not included in the github repo)
├── mdl
│   ├── model_seq_291.h5
│   └── tokenizer_seq_291.pickle 
├── src
│   ├── generate_text.py
│   └── train_model.py
├── assignment_description.md
├── README.md
├── requirements.txt
└── setup.sh
```

## Results
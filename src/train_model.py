"""
Pipeline for training a recurrent neural network on comments from New York Times articles. The following steps are taken
    - loads data and chooses n random comments to train on
    - cleaning the data 
    - tokenization of the data
    - creating sequences and padding them
    - training a recurrent neural network on the padded sequences
    - saving the model
    - plotting the accuracy and loss of the model

Authour: Laura Bock Paulsen (202005791@post.au.dk)
"""

# data processing tools
import string
from pathlib import Path
import pandas as pd
import numpy as np
np.random.seed(42)

import random
import matplotlib.pyplot as plt
import pickle

# tensorflow imports
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

import logging
import argparse as ap

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-n", "--n_comments", type = int, default = None)

    return parser.parse_args()

def custom_logger(name):
    logger = logging.getLogger(__name__)
    
    # console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    # format log message
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def clean_txt(txt:str): 
    """
    Cleans the text by removing newlines, punctuation and non-ascii characters.

    Parameters
    ----------
    txt : str
        The text to be cleaned.

    Returns
    -------
    txt : str
        The cleaned text.
    """

    # remove newlines
    txt = txt.replace("<br/>", " ")

    # remove punctuation and lower case
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    
    # encode and decode to remove non-ascii characters
    txt = txt.encode("utf8").decode("ascii",'ignore')

    return txt


def load_data(path:str, query:str): 
    """
    Loads the data from the given path and returns a list of strings containing the data given the query.

    Parameters
    ----------
    path : str
        The path to the directory containing the data.
    query : str
        The query to be used to load the data. Can be either "Comments" or "Articles".

    Returns
    -------
    texts : list
        A list of strings containing the data given the query. If query is "Comments", the list will contain the comments. If query is "Articles", the list will contain headlines.
    """

    # check if query is valid
    if query not in ["Comments", "Articles"]:
        raise ValueError("Query must be either 'Comments' or 'Articles'")

    # list files in directory
    files = path.glob("*.csv")

    # loop through files and load data
    texts = []
    for f in files:
        if query in f:
            if query == "Comments":
                df = pd.read_csv(path / f, usecols=["commentBody"])
            else:
                df = pd.read_csv(path / f, usecols=["headlines"])
            
            # append to list
            texts.extend(df.values.flatten().tolist())

    return texts


def get_sequences(texts:list, tokenizer:Tokenizer):
    """
    Tokenizes a list of texts, converts the texts to sequences and pads the sequences.

    Parameters
    ----------
    texts : list
        A list of strings containing the texts to be tokenized.

    Returns
    -------
    sequences : np.array
        A list of padded sequences.

    total_words : int
        The total number of unique words in the texts.
    
    max seq: int
        The number of sequences after padding
    """

    ## tokenization
    tokenizer.fit_on_texts(texts)
    
    n_words = len(tokenizer.word_index) + 1

    # convert data to sequences
    sequences = tokenizer.texts_to_sequences(texts)

    # find max sequence length
    seq_len = max([len(x) for x in sequences])

    # pad sequences
    sequences = np.array(pad_sequences(sequences, maxlen=seq_len, padding='pre'))

    return sequences, n_words, seq_len


def create_model(sequence_len, total_words):
    input_len = sequence_len - 1
    model = Sequential()
    
    # add input embedding layer
    model.add(Embedding(total_words, 
                        10, 
                        input_length=input_len))
    
    # add hidden layer (LSTM) 
    model.add(LSTM(30))
    model.add(Dropout(0.1))
    
    # add output layer
    model.add(Dense(total_words, 
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model


def train_model(sequences:list, n_words:int):
    """
    Trains a RNN model on the given sequences.

    Parameters
    ----------
    sequences : list
        A list of padded sequences.

    n_words : int
        The total number of unique words in the texts.
    
    Returns
    -------
    model : keras model
        The trained model.

    history : keras history
        The history of the training.
    """

    # initialise model
    model = create_model(sequences.shape[1], n_words)

    # preparing predictors and label
    predictors, label = sequences[:,:-1], sequences[:,-1]

    label = to_categorical(label, num_classes=n_words)

    history = model.fit(predictors, 
                    label, 
                    epochs=20,
                    batch_size=128, 
                    verbose=1)

    return model, history


def plot_history(history, save_path:str = None):
    """
    Plots the loss curve 
    
    Parameters
    ---------
    history : tf.keras.callbacks.History
        History returned from training model

    save_path : str
        Path for saving plot. 

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi = 300)

    # plot loss
    ax.plot(history.history["loss"], label = "loss")

    # set labels
    ax.set_xlabel("Epoch")

    # add legend
    ax.legend()

    if save_path:
        plt.savefig(save_path)
    
    plt.close()


def main():
    path = Path(__file__)

    data_path = path.parents[1] / "data"

    args = parse_args()

    # initialise logger
    logger = custom_logger("train-model")
    
    # load data
    logger.info("LOADING DATA")
    comments = load_data(data_path, "Comments")
    
    if args.n_comments: # if it is not None, a subset of the comments are chosen
        comments = random.choices(comments, k = args.n_comments)
    
    print(len(comments))

    # clean data
    logger.info("CLEANING DATA")
    comments = [clean_txt(txt) for txt in comments if txt is not np.nan]

    # tokenize and sequence data
    tokenizer = Tokenizer()

    logger.info("GETTING SEQUENCES")
    sequences, n_words, seq_len = get_sequences(comments, tokenizer)

    # save tokenizer (needed for text generation)
    tokenizer_path = path.parents[1] /  "mdl" / f"tokenizer_seq_{seq_len}.pickle"
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # train model
    logger.info("TRAINING MODEL")
    model, history = train_model(sequences, n_words)

    # save model
    model_path = path.parents[1] / "mdl" / f"model_seq_{seq_len}.h5"
    model.save(model_path)

    # plot history
    fig_path = path.parents[1] / "figs" / f"history_seq_{seq_len}.png"
    plot_history(history, save_path = fig_path)


if __name__ == "__main__": 
    main()
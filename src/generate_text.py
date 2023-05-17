"""
Generates text from the model trained using train_model.py

Authour: Laura Bock Paulsen (202005791@post.au.dk)
"""
import argparse
import pickle
from pathlib import Path
import numpy as np
#import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type = str, required = True)
    parser.add_argument("--n_words", type = int, default = 5)
    parser.add_argument("--model", type = str, default = "model_seq_297.h5")
    
    return parser.parse_args()

def search_dict(dict:dict, value):
    """
    Searches for a value in a dictionary, and returns the key of the first match

    Parameters
    ----------
    dict : dict
        A dictionary
    
    value : str, int, float
        The value to search for in the dictionary

    Returns
    -------
    key : str
        The key with the value
    """

    key_list = list(dict.keys())
    value_list = list(dict.values())

    key = key_list[value_list.index(value)]

    return key


def generate_text(prompt:str, n_words:int, model, tokenizer, seq_len:int):
    """
    Generates text

    Parameters
    ----------
    prompt : str
        The user-given input prompt from which to continue text generation
    
    n_words : int
        The number of words to generate following the prompt
    
    model : 
        The trained model
    
    tokenizer :
        The tokenizer used on the training data

    seq_len: int
        The length of the padded sequences used for training the model

    Returns
    -------

    """
    prompt = prompt.lower()
    
    output = ""
    word_dict = tokenizer.word_index
    for _ in range(n_words): # loop over the number of words to generate
        # tokenizing the prompt
        token_list = tokenizer.texts_to_sequences([prompt + output])[0]
        
        # padding the tokenized prompt
        token_list = pad_sequences([token_list], 
                                    maxlen=seq_len - 1, 
                                    padding='pre')
        
        # predicting the next word
        predicted = np.argmax(model.predict(token_list),
                                            axis=1)
        
        # determining the word
        output_word = search_dict(word_dict, predicted)

        output += " " + output_word
    
    return output

def main():
    # parsing args from command-line
    args = parse_args()

    # extract sequence length from filename of model
    seq_len = int(args.model.split(".h5")[0].split('_')[-1])
    
    path = Path(__file__)
    model_path = path.parents[1] / "mdl" / args.model
    tokenizer_path = path.parents[1] /  "mdl" / f"tokenizer_seq_{seq_len}.pickle"

    # loading the trained model
    model = load_model(model_path)

    print(type(model))

    # loading tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # generate text from the user-specified prompt
    generated_txt = generate_text(args.prompt, args.n_words, model, tokenizer, seq_len = seq_len)
    
    print(args.prompt.upper() + generated_txt)


if __name__ == "__main__": 
    main()

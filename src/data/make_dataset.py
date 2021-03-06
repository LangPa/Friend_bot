# -*- coding: utf-8 -*-
import os
from pathlib import Path
import json
import numpy as np
import io

def extract(input_filepath, output_filepath = 'data'):
    """Concatinates and saves message data taken from single chat

    Args:
        input_filepath (str): filepath to directory containing messages
        output_filepath (str): Save location

    """
    input_filepath = Path(input_filepath)
    message_files = input_filepath.glob('message_*.JSON')
    messages = []
    for message_file in message_files:
        json_file = open(message_file, "r", encoding='utf-8')
        # jdict = json.load(json_file)  ## TEST TO SEE IF MODIFYING WORKS
        # messages += jdict['messages']
        messages += json.load(json_file)['messages']                
        json_file.close()

    # Create dictionary of participants continuous chat 
    message_content_dict = {}

    for message in messages:
        if 'content' in message:
            message_content_dict[message['sender_name']] = message_content_dict.get(message['sender_name'], '') + message['content'].encode('latin1').decode('utf8') + '\n'

    # write data to file, separated by participants
    path = Path(output_filepath) / input_filepath.stem
    if not os.path.exists(path):
        path.mkdir()

    for participant in message_content_dict:
        with io.open(str(path / participant.replace(' ','_')) + '_messages.txt', 'w', encoding = 'utf-8') as out_file:
            out_file.write(message_content_dict[participant])



class data():
    """Returns batch data for a single facebook user in a messenger chat.

    Attributes:
        char (tuple): all characters appearing in text
        int_to_char (dict): integer to character encoding dictionary
        char_to_int (dict): character to integer encoding dicitonary
        data_loc (str): processed data location
        encoded (array): encoded data
    
    Arguments:
        input_filepath (str): input filepath to raw messenger data.
        output_filepath (str): Destination of processed data
    """
    def __init__(self):

        self.chars = None
        self.int_to_char = None
        self.char_to_int = None
        self.data_loc = None
        self.encoded = None
                
    def encode(self, input_filepath):
        """ encodes data

        Args:
            input_filepath (str): message data chosen. Must be continuous text

        Returns:
            encoded (str): encoded text
        """
        
        f = io.open(input_filepath, 'r', encoding='utf-8')
        text = f.read()
        f.close()

        self.data_loc = input_filepath
        self.chars = tuple(set(text))
        self.int_to_char = dict(enumerate(self.chars))
        self.char_to_int = {ch: ii for ii, ch in self.int_to_char.items()}

        self.encoded = np.array([self.char_to_int[ch] for ch in text])
        return self.encoded


    def get_batches(self, batch_size, seq_length, train_val = None, val_frac = 0.3):
        """ Generates batch data

        Args:
            batch_size (int): Batch size, the number of sequences per batch
            seq_length (int): Number of encoded chars in a sequence


        Yields:
            x (array): batch_size x seq_length output features
            y (array): batch_size x seq_length output targets (features shifted one step forward)

        """
        batch_size_total = batch_size * seq_length
        try:
            n_batches = len(self.encoded)//batch_size_total
        except TypeError:
            print('No data!\nData must be chosen and encoded using encode()')
            return None

        # Keep only enough characters to make full batches
        arr = self.encoded[:n_batches * batch_size_total]
        # Reshape into batch_size rows
        arr = arr.reshape((batch_size, -1))

        # selecting train or validate data
        if train_val == 'train':
            start, finish = 0, seq_length * int(arr.shape[1]/seq_length * (1 - val_frac))
            # print('training', start, finish)
        elif train_val == 'validate':
            start, finish = seq_length * int(arr.shape[1]/seq_length * (1 - val_frac)), arr.shape[1]
            # print('validating', start, finish)
        else:
            start, finish = 0, arr.shape[1]

        # iterate through the array, one sequence at a time
        for n in range(start, finish, seq_length):
            # The features
            x = arr[:, n:n+seq_length]
            # The targets, shifted by one
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            
            yield x, y

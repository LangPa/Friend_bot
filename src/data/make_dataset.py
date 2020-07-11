# -*- coding: utf-8 -*-
import os
from pathlib import Path
import json

class data():
    """Returns batch data for a single facebook user in a messenger chat.

    Attributes:
        friend (str): sender_name of persons messsenger data.
        char (tuple): all characters appearing in text
        int_to_char (dict): integer to character encoding dictionary
        char_to_int (dict): character to integer encoding dicitonary
        data_loc (str): processed data location
        encoded (array): encoded data
    
    Arguments:
        input_filepath (str): input filepath to raw messenger data.
        output_filepath (str): Destination of processed data
    """
    def __init__(self, input_filepath = None, output_filepath = 'data/processed'):
        
        self.friend = None
        self.chars = None
        self.int_to_char = None
        self.char_to_int = None
        self.data_loc = None
        self.encoded = None

        if input_filepath:
            message_files = Path(input_filepath).glob('message_*.JSON')
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
            path = Path(out) / input_filepath.split('/')[-1]
            path.mkdir()

            for participant in message_content_dict:
                with open(path / participant + '_messages.txt', 'w') as out_file:
                    out_file.write(message_content_dict[participant])

            self.data_loc = path
        
        
    def encode(self, input_filepath):
         """ encodes data

        Args:
            input_filepath (str): message data chosen. Must be continuous text

        Returns:
            encoded (str): encoded text

        """
        try:
            f = open(path, 'r')
            text = f.read()
            f.close()
        except:
            if self.data_loc:
                print(f'no / incorrect file found at {input_filepath}\nTry input_filepath =\n')
                for path in Path(self.data_loc):
                    print(path)
            else:
                print(f'no / incorrect file found at {input_filepath}'')
            return None

        self.chars = tuple(set(text))
        self.int_to_char = dict(enumerate(chars))
        self.char_to_int = {ch: ii for ii, ch in int_to_char.items()}

        self.encoded = np.array([char_to_int[ch] for ch in text])
        return self.encoded


    def one_hot_encode(self, arr):
        """One hot encodes array

        Args:
            arr (array): Array to be encoded, must be and integer array with values <= len(self.chars)

        Returns:
            one_hot (array): one hot encoded array
        """

        # Initialize the the encoded array
        one_hot = np.zeros((arr.size, len(self.chars)), dtype=np.float32)

        # Fill the appropriate elements with ones
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
        
        # Finally reshape it to get back to the original array
        one_hot = one_hot.reshape((*arr.shape, len(self.chars)))
        
        return one_hot


    def get_batches(self, batch_size, seq_length, one_hot = True):
        """ Generates batch data

        Args:
            batch_size (int): Batch size, the number of sequences per batch
            seq_length (int): Number of encoded chars in a sequence
            one_hot (bool): returns onehot encoded data


        Yields:
            x (array): batch_size x seq_length output features
            y (array): batch_size x seq_length output targets (features shifted one step forward)

        """

        if not self.encoded:
            print('No data!\nData must be chosen and encoded using encode()')
            return None
        
        batch_size_total = batch_size * seq_length
        n_batches = len(self.encoded)//batch_size_total

        # Keep only enough characters to make full batches
        arr = self.encoded[:n_batches * batch_size_total]
        # Reshape into batch_size rows
        arr = arr.reshape((batch_size, -1))

        # iterate through the array, one sequence at a time
        for n in range(0, arr.shape[1], seq_length):
            # The features
            x = arr[:, n:n+seq_length]
            # The targets, shifted by one
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]

            if one_hot:
                yield self.one_hot_encode(x), self.one_hot_encode(y) 
            else:
                yield x, y

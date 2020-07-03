# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
import json
from dotenv import find_dotenv, load_dotenv

#data\raw\Gurbir\message_1.json
#data\interim

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default = 'data/raw')
@click.option('--out', type=click.Path(), default = 'data/processed', help = 'output filepath')
@click.option('--interim', type=click.Path(), default='data/interim', help = 'interim filepath')
@click.option('--list', is_flag = True, help = 'list files in directory' )
def main(input_filepath, out, interim, list):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        Intermediate combined JSON files are put into ../interim
    """
    logger = logging.getLogger(__name__)
    logger.info('making data set from raw data')

    if list:
        for file in os.listdir(input_filepath):
            click.echo(file)
        return
    

    # Retrieve message files and combine into single dict, write to intermin file
    if len([path for path in Path(input_filepath).glob('message_*.JSON')]) == 0:
        click.echo(f'No message files found in {input_filepath}')
        return

    message_files = Path(input_filepath).glob('message_*.JSON')
    messages = []
    for message_file in message_files:
        json_file = open(message_file, "r", encoding='utf-8')
        jdict = json.load(json_file)
        messages += jdict['messages']
        json_file.close()

    message_dict = jdict.copy()
    message_dict['messages'] = messages

    path = Path(interim) / input_filepath.split('/')[-1]
    path.mkdir()

    with open(path / 'messages.json', "x") as interim_file:
        json.dump(message_dict, interim_file)

    logger.info('Json files combined in data/interim')


    # Get continuous chat from message content, save to 
    message_content_dict = {}

    for message in messages:
        if 'content' in message:
            message_content_dict[message['sender_name']] = message_content_dict.get(message['sender_name'], '') + message['content'].encode('latin1').decode('utf8') + '\n'


    path = Path(out) / input_filepath.split('/')[-1]
    path.mkdir()

    with open(path / 'message_contents.json', "x") as out_file:
        json.dump(message_content_dict, out_file)

    logger.info('message contents extracted')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

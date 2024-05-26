import os
import sys
import argparse
sys.path.append(os.path.join(".."))

import logging

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

logging.basicConfig(filename='logging.log',  # Log filename
                    filemode='a',  # Append mode, so logs are not overwritten
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp
                    level=logging.INFO,  # Logging level
                    datefmt='%Y-%m-%d %H:%M:%S')  # Timestamp format

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set logging level for console
logging.getLogger('').addHandler(console_handler)

# Redirect stdout and stderr to logging
sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

from utils import *
from train import *

def main():
    # Check if the script is running on the server by looking for the environment variable
    if os.getenv('RUNNING_ON_SERVER') == 'true':
        parser = argparse.ArgumentParser(description='Process data directory.')

        parser.add_argument('--train_data_dir', type=str, help='Path to the train data directory')
        parser.add_argument('--val_data_dir', type=str, help='Path to the validation data directory')
        parser.add_argument('--project_name', type=str, help='Name of the project')
        parser.add_argument('--train_continue', type=str, default='off', choices=['on', 'off'],
                            help='Flag to continue training: "on" or "off" (default: "off")')
        parser.add_argument('--extra_noise', type=bool, default=False,
                            help='Add extra noise to the data: True or False (default: False)')

        args = parser.parse_args()

        train_data_dir = args.train_data_dir
        val_data_dir = args.val_data_dir
        project_name = args.project_name
        train_continue = args.train_continue
        extra_noise = args.extra_noise
        project_dir = os.path.join('/g', 'prevedel', 'members', 'Rauscher', 'projects', 'one_adj_slice')
        
        print(f"Using train data directory: {train_data_dir}")
        print(f"Using val data directory: {val_data_dir}")
        print(f"Project name: {project_name}")
        print(f"Train continue: {train_continue}")
        print(f"Extra noise enabled: {extra_noise}")

    else:
        # Default settings for local testing
        train_data_dir = r"C:\Users\rausc\Documents\EMBL\data\Nematostella_B"
        val_data_dir = r"C:\Users\rausc\Documents\EMBL\data\Nematostella_B"
        project_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'projects', 'one_adj_slice')
        project_name = 'Nema_B-test_x'
        train_continue = 'off'
        extra_noise = False

    data_dict = {
        'train_data_dir': train_data_dir,
        'val_data_dir': val_data_dir,
        'project_dir': project_dir,
        'project_name': project_name,
        'num_epoch': 600,
        'batch_size': 8,
        'lr': 1e-5,
        'num_freq_disp': 5,
        'train_continue': train_continue,
        'extra_noise': extra_noise
    }

    trainer = Trainer(data_dict)
    trainer.train()

if __name__ == '__main__':
    main()




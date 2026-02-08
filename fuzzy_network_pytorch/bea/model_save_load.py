import pickle
import time
import os.path

def save_model(data, directory, filename, append_time=False):
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = f'{directory}/{filename}'
    
    if append_time or os.path.exists(path):
        path += '_' + time.strftime("%Y%m%d-%H%M%S") + '.model'

    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

    print(f'[+] Model successfully saved to: {path}')

def load_model(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)

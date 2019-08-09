import os
import json

def read_json(fpath):
    data = None
    try:
        with open(fpath, "r") as stream:
            data = json.load(stream)
    except FileNotFoundError:
        pass
    return data


def write_json(fpath, data):
    dpath = os.path.dirname(fpath)
    os.makedirs(name=dpath, exist_ok=True)
    with open(fpath, "w") as file:
        json.dump(data, file)





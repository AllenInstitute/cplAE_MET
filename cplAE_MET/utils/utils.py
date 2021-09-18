import csv
import numpy as np
import pickle


def write_list_to_csv(path, file):
    with open(path, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(file)
    print("Done writing!")


def read_list_from_csv(path):
    with open(path, 'r') as myfile:
        reader = csv.reader(myfile)
        data = list(reader)
    return data[0]


def savepkl(ob, fname):
    with open(fname, 'wb') as f:
        pickle.dump(ob, f)
    return


def loadpkl(fname):
    with open(fname, 'rb') as f:
        X = pickle.load(f)
        return X
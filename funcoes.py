import pandas as pd

def readDataset():
    return pd.read_csv(open('pathToDataset.txt', 'r').read().split("\n")[0])
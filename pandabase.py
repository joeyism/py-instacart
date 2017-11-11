import pandas as pd
from os import listdir
from os.path import isfile, join

class DataSet(pd.DataFrame):

    def rows(self, num):
        return self.loc[self.index <= num, :]

def readFiles(mypath):
    filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    result = []
    for filename in filenames:
        df = DataSet(pd.read_csv(mypath+filename))
        dict_name = filename.replace(".csv","")
        result.append((dict_name, df))
    return dict(result)


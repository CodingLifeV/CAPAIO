import arff
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# read csv data
def read_csv_file(file_path):

    f = arff.load(open(file_path, 'r'))
    data = f['data']

    data = [[value.decode('utf-8').strip("'") if isinstance(value, bytes) else value for value in row] for row in data]

    num_attr = len(data[0]) - 1
    att = f['attributes']
    relation = f['relation']
    meta = []

    # for every attribute check if it is numeric or categorical
    for i in range(len(att) - 1):
        attribute_name = att[i][0]  # get column name
        if (att[i][1] == "NUMERIC"):
            meta.append(0)
        else:
            meta.append(1)

    # split each sample into attributes and label
    X = [i[:num_attr] for i in data]
    y = [float(i[-1]) for i in data]

    # indentify the existing classes
    classes = np.unique(y)

    # create new class labels from 0 to n, where n is the number of classes
    y = [np.where(classes == float(i[-1]))[0][0] for i in data]

    return [X, y, meta, att, relation]
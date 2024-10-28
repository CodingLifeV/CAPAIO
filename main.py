from pathlib import Path

import numpy as np
import constants
from capaio.capaio import CAPAIO
from uitls import read_csv_file


import warnings
warnings.filterwarnings("ignore")

FOLDS_PATH = Path(__file__).parent / 'data' / 'folds'

def _get_data():

    [X, y, meta, attribute_names, relation] = read_csv_file(constants.FILE_PATH)
    constants.relation = relation
    constants.attribute_names = attribute_names

    return X, y, meta


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for name in constants.names:
        for fold in range(1, constants.K_FLOD+1):

            train_path = FOLDS_PATH / name / ('%s.%d.train.arff' % (name, fold))

            X_train, y_train, meta, attribute_names, relation = read_csv_file(train_path)
            constants.relation = relation
            constants.attribute_names = attribute_names

            # scaler = MinMaxScaler().fit(X_train)
            # X_train = np.c_[scaler.transform(X_train)]

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            capaio = CAPAIO()
            X_train_res, y_train_res = capaio.fit_resample(X_train, y_train, is_binary=True)










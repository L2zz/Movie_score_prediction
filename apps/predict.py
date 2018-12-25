import init_set
import preproc
import numpy as np
import tensorflow as tf

if __name__ == '__main__':

    sess = init_set.init()

    x_train, y_train = preproc.get_data(True)
    x_test, _ = preproc.get_data(False)

    



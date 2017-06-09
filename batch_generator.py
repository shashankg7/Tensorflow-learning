import numpy as np
import random
import pdb
import math

def batch_gen(X, batch_size):
    n_batches = X.shape[0]/float(batch_size)
    n_batches = int(math.ceil(n_batches))
    end = int(X.shape[0]/float(batch_size)) * batch_size
    n = 0
    i = 0
    #for i in xrange(0,n_batches):
    while(True):
        if i < n_batches - 1:
            print "in if"
            if len(X.shape) > 1:
                batch = X[i*batch_size:(i+1) * batch_size, :]
                yield batch
            else:
                batch = X[i*batch_size:(i+1) * batch_size]
                yield batch
            i += 1
        else:
            print "in else"
            if len(X.shape) > 1:
                batch = X[end: , :]
                n += X[end:, :].shape[0]
                yield batch
            else:
                batch = X[end:]
                n += X[end:].shape[0]
                yield batch
            i = 0
            np.random.shuffle(X)

if __name__ == "__main__":
    X = np.random.rand(5000, 300)
    for batch in batch_gen(X, 32):
        print batch.shape

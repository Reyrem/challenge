import numpy as np
import pandas as pd
import time
import pickle
from Constantes import *
from multiprocessing import Pool
import GaussNewton as MSE

if __name__ == '__main__':
    t0 = time.time()
   
    frames = np.load("frames.npy")
    for i in frames:
        if i> 3462719:
            i-=1
    pool = Pool(N)
    res = pool.map(MSE.localise, frames)
    
    pool.close()
    pool.terminate()
    
    MSE.exportData()
    print("TOTAL EXECUTION TIME : " + str(time.time() - t0))
    
import numpy as np
import pandas as pd
import random as rd
import math
import time
import pickle
import scipy.optimize as so
from Constantes import *
from Utilitaires import *
from multiprocessing import Pool
from multiprocessing import Lock




dataFile = 'round2_competition.csv'
sensFile = 'round2_sensors.csv'

df = pd.read_csv(dataFile, engine='python', sep=",", error_bad_lines=False)
print(df.iloc[0])
df_sens = pd.read_csv(sensFile, engine='python',  sep=",")
sensorsTable = np.load('sensorsTableCorrected.npy')
sensorsLoc = np.load('sensorsLoc.npy')
sensorsDrift = np.load('sensorsDrift.npy')
nonLinearSensors = np.load("NonLinearSensors.npy")
goodSensorsCorrectionFile = 'captorSynchroCOMPETITION_30_10.pkl'
flux = open(goodSensorsCorrectionFile, 'rb')
goodSensorsCorrection = pickle.load(flux)
flux.close()

df_sample = pd.read_csv('round2_sample_empty.csv')

# def readFiles(path, dataFile, sensFile):
#     df = pd.read_csv(dataFile)
#     df_sens = pd.read_csv(sensFile)
#     return df, df_sens

  
#modèle temporel
def tau(t, alpha, b):
  return (alpha*(t - b))
  
def correctedTime(t, serial):
  alpha = sensorsDrift[serial][0]
  b = sensorsDrift[serial][1]
  return tau(t, alpha, b)

#calibrated time for "good" sensors
def correctedTimeGood(t, serial):
  if serial in goodSensorsCorrection:
    return (t - goodSensorsCorrection.get(serial))
  else:
    return t 

#composante à minimiser
def delta(x, r, serials, l):
  f = []
  xt = tuple(x)
  fi0 = distance3D(xt, tuple(sensorsLoc[l]))

  for s in serials:
    fi = fi0 - distance3D(xt, tuple(sensorsLoc[s]))
    f.append(fi)
  
  res = r - np.array(f)
  return res


def exportData():
  df_sample.to_csv('RESULTS.csv', index=False)

#METHODE DE GAUSS NEWTON
def localise(indexTrame, verbose=True):
  measurements = readMeasurements(df.iloc[indexTrame]['measurements'])
  baroAltitude = df.iloc[indexTrame]['baroAltitude']
  k = 0
  strongerSignal = 0
  #we have to make sure the reference sensor is not broken
  while (sensorsTable[int(measurements[k][0])] < 0):
    k += 1

  l = int(measurements[k][0])
  timeStampL = seconds(float(measurements[k][1]))
  timeStampL = correctedTime(timeStampL, l)
  if sensorsTable[l]==1:
    timeStampL = correctedTimeGood(timeStampL, l)
  strongerSensor = l

  r = []
  serials = []

  if k==len(measurements):
    print("Localisation error")
    return None
  else:
    for i in range(k+1, len(measurements)):
      si = int(measurements[i][0])
      timeStampI = seconds(float(measurements[i][1]))
      timeStampI = correctedTime(timeStampI, si)
      if sensorsTable[si]==1:
        timeStampI = correctedTimeGood(timeStampI, si)
      if (int(measurements[i][2]) > strongerSignal):
        strongerSignal = int(measurements[i][2])
        strongerSensor = si


      serials.append(si)
      r.append((c / nAir) * (timeStampL - timeStampI))

  npR = np.array(r)
  #-----------------------------------------
  x = list(sensorsLoc[strongerSensor])
  x[2] = baroAltitude

  bnds = ((x[0] - 2.0, x[1] - 2.0, x[2] - 1000.0), (x[0] + 2.0, x[1] + 2.0, x[2] + 1000.0))

  arguments = (r, serials, l)

  result = so.least_squares(delta, x, jac='3-point', args=arguments, ftol = 10**(-8), xtol = 10**(-8), loss='soft_l1', x_scale=(0.01, 0.01, 10), bounds=bnds)
  if verbose:
    print(x)
    print(result.x)
    print(result.success)
    print(result.message)

  
  df_sample.latitude.iloc[indexTrame] = x[0]
  df_sample.longitude.iloc[indexTrame] = x[1]
  df_sample.longitude.iloc[indexTrame] = x[2]
  return result.x

  
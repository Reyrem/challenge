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

import csv

# from pandas.io import parser; try: your read_csv look for file f except (parser.CParserError) as detail: print f, detail

lock = Lock()
dataFile = 'round2_competition.csv'
sensFile = 'round2_sensors.csv'






# def my_data_generator(fp):
#   metadata = []
#   for line in fp:
#     data = line.strip().split(',')
#     print(data)
#     # if len(data) == 4:
#     #   metadata = data
#     # elif not metadata:
#     #   raise ValueError("csv file did not start with metadata")
#     # elif data:
#     #   yield metadata + data

# df = pd.DataFrame.from_records(my_data_generator(open('round2_competition.csv')))

# print(df)


#df = pd.read_csv(dataFile, engine='python', sep=",", error_bad_lines=False, quoting=csv.QUOTE_NONE)
# df = pd.read_csv(dataFile, engine='python', sep=",", quoting=csv.QUOTE_NONE, error_bad_lines=False, encoding='utf-8')


df = pd.read_csv(dataFile, engine='python', sep=",",  error_bad_lines=False, encoding='iso-8859-1')


# df = pd.read_csv(dataFile, engine='python', sep=",",error_bad_lines=False)
#df = pd.read_csv(dataFile, engine='python', sep=",", error_bad_lines=False, quoting=csv.QUOTE_NONE)
#df = pd.read_csv(dataFile, engine='python', sep=",", quoting=csv.QUOTE_NONE, error_bad_lines=False, encoding='utf-8')
#df = pd.read_csv(dataFile, engine='python', sep=",")
# try:
#   df = pd.read_csv(dataFile, sep=",",lineterminator='\n')
# except pd.errors.ParserError as detail : print(detail) 

#df = pd.read_csv(dataFile, engine='python', sep=",",warn_bad_lines=False)

for i in range (1000) : print(df['measurements'].count())
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
  df_sample.to_csv(r'./RESULTS_9-12-0830.csv', index=False)
  

#METHODE DE GAUSS NEWTON
def localise(indexTrame, verbose=True):


  try:
    measurements = readMeasurements(df.iloc[indexTrame]['measurements'])
    baroAltitude = df.iloc[indexTrame]['baroAltitude']
  except IndexError:
    print(" ++++++++++++++++++++++++++++++ INDEXTRAME +++++++++++++++++++++++++++++++++++ ", indexTrame)
    with open("frames_problematiques.txt","a") as prob:
      prob.write("frame non traitee le indexTrame est out of bounds askip     :   ")
      prob.write(str(indexTrame))
      prob.write("\n\n\n")
    return None


  k = 0
  strongerSignal = 0
  #we have to make sure the reference sensor is not broken
  while ((k < len(measurements)) and (sensorsTable[int(measurements[k][0])] < 0)):
    k += 1

  if k==len(measurements):
 
    print("---------------------------------- INDEXTRAME ----------------------------------", indexTrame)
    with open("frames_problematiques.txt","a") as prob:
      prob.write("frame Non traitée car il n'y a pas de capteur valide disponible     :   ")
      prob.write(str(indexTrame))
      prob.write("\n\n\n")
    return None

  else:
    l = int(measurements[k][0])
    timeStampL = seconds(float(measurements[k][1]))
    timeStampL = correctedTime(timeStampL, l)
    if sensorsTable[l]==1:
      timeStampL = correctedTimeGood(timeStampL, l)
    strongerSensor = l

    r = []
    serials = []

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
  
  try :
    lock.acquire()
    with open("frames_corrigees.txt","a") as fichier:
      fichier.write(str(indexTrame)+";")
      fichier.write(str(x[0])+";")
      fichier.write(str(x[1])+";")
      fichier.write(str(x[2])+"\n")
    lock.release()
    print(indexTrame)
  except IndexError :
    print("---------------------------------- INDEXTRAME", indexTrame)
    with open("frames_problematiques.txt","a") as prob:
      prob.write("frame  :  ")
      prob.write(str(indexTrame))
      prob.write("\n\n\n")


  # try:
  #     #if verbose:
  #       #print(x)
  #       #print(result.x)
  #       #print(result.success)
  #       #print(result.message)
  #   print(indexTrame,"---------->", result.x)
  #   df_sample.latitude.iloc[indexTrame] = x[0]
  #   df_sample.longitude.iloc[indexTrame] = x[1]
  #   df_sample.longitude.iloc[indexTrame] = x[2]

  # #T'as pas le droit de faire ça
  # except IndexError : 
  #   print("----------------------",indexTrame, df_sample.id.count(), "------------------------------------------")
  #   with open("frames_problematiques.txt","a") as fichier:
  #     fichier.write("frame  :  ")
  #     fichier.write(str(indexTrame))
  #     fichier.write(  '   ')
  #     fichier.write(str(df_sample.id.count()))
  #     fichier.write("\n\n\n")

  #     lock.acquire()
  #     try : 
  #         exportData()
  #     finally :
  #         lock.release()
  return result.x

  

#Là ou y'a tous les nouveaux trucs
  #~/retourCode-8-12-20/challenge/GAUSSNEWTON


# ---------------------- 1527155 632932 ------------------------------------------
# ---------------------- 2497593 632932 ------------------------------------------
# ---------------------- 3074110 632932 ------------------------------------------





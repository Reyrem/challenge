import geopy.distance as gd
from shapely.geometry import Point
import math

#turns a measurement entry (string) into a measurements list
def readMeasurements(meas):
  res = []
  meas = meas.split("],")
  for point in meas:
    point = point.replace("[","")
    point = point.replace("]","")
    res.append(point.split(","))
  return res
  
#3D distance between 2 tuples (altitude in meters)
def distance3D(p1, p2):
  flat_distance = gd.distance(p1[:2], p2[:2]).m
  euclidian_distance = math.sqrt(flat_distance**2 + (p2[2] - p1[2])**2)
  return euclidian_distance

#nanoseconds to seconds conversion
def seconds(t):
  factor = 10**(-9)
  return (factor * t)

  
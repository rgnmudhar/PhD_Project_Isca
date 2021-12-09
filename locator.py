"""
    Script for finding the indices for closest values to specific pressure, latitude or longitude.
    Takes user inputs.
    Outputs the closest values in the .nc dataset plus the index of the value in the lat, lon or pfull arrays.
    Best to run this script in a run000X directory.
"""

import xarray as xr
from bisect import bisect_left
import numpy as np

def closest(mylist, myval):
    if (myval > mylist[-1] or myval < mylist[0]):
        return False
    pos = bisect_left(mylist, myval)
    if pos == 0:
        return mylist[0]
    if pos == len(mylist):
        return mylist[-1]
    before = mylist[pos - 1]
    after = mylist[pos]
    if after - myval < myval - before:
        print('index=',pos-1)
        return after
    else:
        print('index=',pos-1)
        return before

ds = xr.open_dataset('Polvani_Kushner_2.0/run0001/atmos_monthly.nc', decode_times=False) 
p = ds.coords['pfull'].data
lat = ds.coords['lat'].data
lon = ds.coords['lon'].data

#lonval = float(input("Input longitude to find: "))
latval = float(input("Input latitude to find: "))
#pval = float(input("Input pressure to find (in hPa): "))

closestlat = print("Closest latitude:", closest(lat, latval))
#closestlon = print("Closest longitude:", closest(lon, lonval))
#closestp = print("Closest pressure:", closest(p, pval))

import os
import sys
import argparse
import datetime
import glob
import math
import numpy as np

from scipy.optimize import curve_fit
from dateutil import relativedelta

import statistics
import pandas as pd

def periodic_function(X, amplitude, bias, xbias):
    return amplitude * np.sin(((2*math.pi)/12) * X + xbias) + bias

def develop_curve(values):
    popt, pcov = curve_fit(
        periodic_function,  # our function
        np.arange(len(values)),  # measured x values
        values,  # measured y values
        p0=(30.0, 20, -1)
    )  # the initial guess for the two parameters
    return popt

parser = argparse.ArgumentParser()
parser.add_argument("--city", type=str, action="store", dest="city")
args = parser.parse_args()

city_dir = os.path.join(args.city)

landsat = pd.concat(
    map(
        pd.read_csv,
        glob.glob(os.path.join(city_dir,'landsat8*.csv'))
    ), 
    ignore_index= True
)

landsat.ST_B10 = landsat.ST_B10 * 0.00341802 + 149 - 273.15
# landsat.ST_QA = landsat.ST_QA * 0.01
landsat = landsat.dropna()

landsat.date = pd.to_datetime(landsat.date)
landsat = landsat.set_index("date")

unique_ids = landsat.id.unique()
popts = []

for unique_id in unique_ids:
    sample_landsat = landsat.loc[landsat.id == unique_id]["ST_B10"].resample('M').mean().ffill()
    popts.append(develop_curve(sample_landsat))
    
popts = np.array(popts)
periodic_regression = pd.DataFrame({
    "id":unique_ids.astype(str), 
    "amplitude":popts[:,0],
    "bias":popts[:,1],
    "xbias": popts[:,2]
#     "bias":popts[:,2]
})

periodic_regression.to_csv(
    os.path.join(
        city_dir,
        "sinusoid_compression.csv"
    )
)
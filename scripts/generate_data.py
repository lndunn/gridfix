import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import os

simyears = 100


def subset_by_day(weatherData):
    return [weatherData.index[i:i+24] for i in range(len(weatherData.index)) if i % 24 == 0]

def bootstrap_from_subsets(subsets, simyears=10, seed=2):
    np.random.seed(seed=seed)
    sampled_subsets = np.random.choice(range(len(subsets)), replace=True, size=simyears*365)

    sampled_index = []
    for i in sampled_subsets:
        sampled_index.extend(subsets[i])
    
    return sampled_index


if __name__ == '__main__':
    weatherData = pd.read_csv('data/weather_data.csv', usecols=['temperature','precipIntensity','windSpeed'])

    daily_subsets = subset_by_day(weatherData)
    sample_index = bootstrap_from_subsets(daily_subsets, simyears=simyears)
    
    X = weatherData.loc[sample_index].reset_index(drop=True).rename(columns={'temperature': 'Temp',
                                                           'precipIntensity': 'Precip',
                                                           'windSpeed': 'Wind'})

    X['Wind'].loc[X['Wind']==0] = 1e-1

    X = X.interpolate()

    X['DayPrecip'] = X[['Precip']].rolling(24).sum()
    X['DayPrecip'].loc[:24] = X['Precip'].loc[:24].sum()

    X['HotCalm'] = X['Temp'] * (1./X['Wind'])
    X['WindStorm'] = X['Wind'] * X['DayPrecip']

    if not os.path.exists('/inputs/'):
        os.mkdir(os.path.join(os.getcwd(),'inputs'))
        
    X.to_csv('inputs/weather.csv', index_label='time')

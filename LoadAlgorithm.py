import joblib
import numpy
import numpy as np
import requests
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

from src.Extractor import Extractor

def get_data(url):


    response = requests.get(url)
    readings_map = {}
    if response.status_code == 200:
        data = response.json()
        if 'metadata' in data and 'stations' in data['metadata']:
            stations = data['metadata']['stations']
            stations_map = process_stations(stations)

        if 'items' in data:
            items = data['items']
            if 'readings' in items[0]:
                readings = items[0]['readings']
                for reading in readings:
                    readings_map[stations_map[reading['station_id']]] = reading['value']
    else:
        print(f"Failed to fetch data")
    return readings_map

def process_stations(stations_data):
    unique_stations = {}
    for station in stations_data:
        pos = station['location']
        unique_stations[station['id']] = (pos['latitude'], pos['longitude'])
    return unique_stations

#if none in cluster , use average
temp_data = get_data(temp_url)
#if none in cluster, use average
humidity_data = get_data(humidity_url)
#if none in cluster, assume 0 rainfall
rainfall_data = get_data(rainfall_url)



# container = {}
#
# stations_map = process_stations(stations)

# for date, data in rainfall_data.items():
#     process_data(container,data,date)


def get_in_cluster(data:dict, clusters):
    arr = []
    for cluster in clusters:
        for pos, value in data.items():
            if cluster.in_hull(pos):
                arr.append(value)
    return np.array(arr)


#If invalid lat long was provided, throw an error
def get_prediction(long, lat):
    gradient: GradientBoostingRegressor = joblib.load('gradient.pkl')
    extractor: Extractor = joblib.load('extractor.pkl')
    humidity_url = 'https://api.data.gov.sg/v1/environment/relative-humidity'
    temp_url = 'https://api.data.gov.sg/v1/environment/air-temperature'
    rainfall_url = 'https://api.data.gov.sg/v1/environment/rainfall'
    clusters = extractor.get_clusters(long, lat)
    if len(clusters) == 0:
        return 0
    temp_arr = get_in_cluster(temp_data,clusters)
    humidity_arr = get_in_cluster(humidity_data,clusters)
    rainfall_arr = get_in_cluster(rainfall_data,clusters)
    density_arr = []

    # Iterate over clusters
    for cluster in clusters:
        # Append cluster density to density_arr
        density_arr.append(cluster.density)

    # Convert the list to a NumPy array
    density_arr = np.array(density_arr)

    # Calculate the mean density
    density = density_arr.mean()

    rainfall = 0
    temp = 32
    humidity = 60
    if humidity_arr.size != 0:
        humidity = humidity_arr.mean()
    if temp_arr.size != 0:
        temp = temp_arr.mean()
    if rainfall_arr.size != 0:
        rainfall = rainfall_arr.mean()

    data_array = [[rainfall, temp, humidity, density]]
    print(f"Data {data_array}")
    if density == 0:
        return 0

    cases = gradient.predict(data_array)[0]

    case_percent = cases/extractor.highest_cases

    density_percent = density/extractor.highest_cases

    return case_percent * density_percent

#print (get_prediction(103.893059,1.398))

# extractor.get_clusters()
#
# y = [[0,30,66,23]]
#
# print(gradient.predict(y))
# print(gradient.predict(y))
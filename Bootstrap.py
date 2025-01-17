import math
import datetime
import common
from common import LINE
from config import setting, log, bootlog
import statistics
from statistics import mean, median, pstdev, pvariance, stdev, variance

import random, time, os, sys, gc, collections, itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from numpy import diag
from pathlib import Path
from scipy import linalg
import logging, traceback

### HELPER FUNCTIONS
def sorted_by_value(xdict):
    import operator
    sorted_x = sorted(xdict.items(), key=operator.itemgetter(1))
    return sorted_x

def timestring(unixtime):
    from datetime import datetime
    ts = int(unixtime)
    s = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return s

def time_travel(distance):
    # distance: in metres
    # return num of hours
    speed = 60.0  # km/h
    return (distance / 1000 / speed)

def json_similar_list(list):
    data_set = {"POI_Prob": []}
    for (poi, prob) in list:
        data_set["POI_Prob"].append({"POI": poi, "Prob": prob})
    return data_set["POI_Prob"]

def get_distance(gps1, gps2, method="manhattan"):
    def haversine(coord1, coord2):
        R = 6372800  # Earth radius in meters
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    [lat1, lng1] = gps1
    [lat2, lng2] = gps2
    if method == "manhattan":
        return (haversine([lat1, lng1], [lat1, lng2])) + (haversine([lat1, lng2], [lat2, lng2]))
    else:
        return haversine([lat1, lng1], [lat2, lng2])

def get_distance_matrix(data):
    poi_distances = dict()
    for i in range(0, len(data)):
        id_i = data.iloc[i]['poiID']
        i_lat = data.iloc[i]['lat']
        i_lng = data.iloc[i]['long']
        for j in range(i, len(data)):
            id_j = data.iloc[j]['poiID']
            j_lat = data.iloc[j]['lat']
            j_lng = data.iloc[j]['long']
            distance = get_distance([i_lat, i_lng], [j_lat, j_lng], "euclidean")
            poi_distances[(id_i, id_j)] = distance
            poi_distances[(id_j, id_i)] = distance
    return poi_distances

def bootstrapping(weight_pop, alpha_pct=90, B=5000, iteration=10000, seed=None):
    ## Part 1. Bootstrap and Confidence Interval
    import numpy as np
    import random
    import time
    from datetime import datetime

    # Seeding randomness
    if seed is None:
        random.seed(int(datetime.now().timestamp()))
        np.random.seed(int(time.time()))
    else:
        random.seed(int(seed))
        np.random.seed(int(seed))

    # Remove samples with value 0 or 1
    weight_pop2 = [w for w in weight_pop if w >= 5 * 60]
    weight_pop = weight_pop2
    if len(weight_pop) == 0:
        return [15 * 60, 15 * 60]

    # Step 2: Sampling
    weight_sample = np.random.choice(weight_pop, size=len(weight_pop) * 5)
    sample_mean = np.mean(weight_sample)  # sample mean
    sample_std = np.std(weight_sample)  # sample std

    # Step 3: Bootstrap for iterations
    boot_means = [np.mean(np.random.choice(weight_sample, replace=True, size=B)) for _ in range(iteration)]
    boot_means_np = np.array(boot_means)

    # Step 4: Analysis and interpretation
    boot_std = np.std(boot_means_np)
    ci = np.percentile(boot_means_np, [(100 - alpha_pct) / 2, 100 - (100 - alpha_pct) / 2])
    return ci

def inferPOITimes(pois, userVisits, alpha_pct=90):
    logging.info("LINE %d Bootstrap.py -- inferPOITimes()", LINE())
    poitimes = dict()

    for poiid in sorted(pois['poiID']):
        poivisits = userVisits[userVisits['poiID'] == poiid]
        if poivisits.empty:
            continue
        user_visit_second = []

        users = poivisits['userID'].unique()
        for userid in users:
            t = poivisits[poivisits['userID'] == userid]

            seqids = t['seqID'].unique()
            for seqid in seqids:
                tt = t[t['seqID'] == seqid]
                timemin = tt['dateTaken'].min()
                timemax = tt['dateTaken'].max()
                timediff = timemax - timemin + 1
                user_visit_second.append(timediff)

        avg_time_ci = bootstrapping(user_visit_second, alpha_pct=alpha_pct)
        logging.debug("POI_ID %2d -> time_visits:  %d%% C.I.: [ %0.3f %0.3f ]", poiid, alpha_pct, avg_time_ci[0], avg_time_ci[1])
        poitimes[poiid] = (avg_time_ci[0], avg_time_ci[1])

    for poiid in pois['poiID']:
        if poiid not in poitimes and not userVisits[userVisits['poiID'] == poiid].empty:
            poitimes[poiid] = (5 * 60, 5 * 60)

    return poitimes

def main():
    print("inferPOITimes2(..)")
    from poidata import load_files
    (pois, userVisits, testVisits, costProfat) = load_files("Buda")
    print("inferPOITimes2(...)  ")

    times = inferPOITimes(pois, userVisits, alpha_pct=95)
    print("inferPOITimes2(....)  ")
    print(times)

if __name__ == "__main__":
    main()

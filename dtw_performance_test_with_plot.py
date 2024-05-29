# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 2021
@author: Stelios Lagaras
"""
import json
import logging
import random
import statistics
import time
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt

import numpy as np
from dtaidistance import dtw_ndim
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dtw_test.log"),  # Terminal output
    ],
    datefmt="%Y-%m-%d %H:%M:%S",  # New date format without milliseconds,
    force = True
)
logger = logging.getLogger(__name__)

# Log CPU info
cpu_info = psutil.cpu_percent(interval=1)
logging.info(f'CPU usage: {cpu_info}%')

# Log RAM info
ram_info = psutil.virtual_memory()
logging.info(f'Total RAM: {ram_info.total / (1024.0 ** 3)} GB')
logging.info(f'Available RAM: {ram_info.available / (1024.0 ** 3)} GB')
logging.info(f'Used RAM: {ram_info.used / (1024.0 ** 3)} GB')


simulation_sets = [{
    "number_of_ts": 1000,
    "length_of_ts": 100,
    "homogeneous": True,
    "dimensions": 3,
    "number_of_executions": 10
}, {
    "number_of_ts": 100,
    "length_of_ts": 100,
    "homogeneous": True,
    "dimensions": 3,
    "number_of_executions": 10
}, {
    "number_of_ts": 100,
    "length_of_ts": 100,
    "homogeneous": False,
    "dimensions": 3,
    "number_of_executions": 10
}, {
    "number_of_ts": 100,
    "length_of_ts": 20,
    "homogeneous": True,
    "dimensions": 3,
    "number_of_executions": 10
}]


class Simulation(object):
    def __init__(self, number_of_ts: int, length_of_ts:int, homogeneous:bool = True, dimensions: int = 3, number_of_executions: int = 10):
        self.number_of_ts = number_of_ts
        self.length_of_ts = length_of_ts
        self.homogeneous = homogeneous
        self.dimensions = dimensions
        self.number_of_executions = number_of_executions
        self.avg_sim_time = None
        self.std_dev_sim_time = None
    
    def execute(self):
        if self.homogeneous == True:
            generate_series(n=self.number_of_ts, k=self.length_of_ts, homogeneous=True)
            with open("dtw_data.json", "r") as f:
                data_json = json.load(f)
        else:
            generate_series(n=self.number_of_ts, k=self.length_of_ts, homogeneous=False)
            with open("dtw_data_non_homogeneous.json", "r") as f:
                data_json = json.load(f)
        sim_times = []
        for _ in range(self.number_of_executions):
            start_time = time.time()
            apply_fusion(data_json)
            sim_times.append(time.time() - start_time)

        self.avg_sim_time = statistics.mean(sim_times)
        self.std_dev_sim_time = statistics.stdev(sim_times)
        logger.info(f"Average:{self.avg_sim_time}, Standard deviation:{self.std_dev_sim_time}")

        return vars(self)
    
    def graph_simulation_n(self):

        sim_times = []
        for n in range(10, self.number_of_ts + 1, 10):
            generate_series(n=n, k=self.length_of_ts, homogeneous=True)
            with open("dtw_data.json", "r") as f:
                data_json = json.load(f)   
            start_time = time.time()
            apply_fusion(data_json)
            sim_times.append(time.time() - start_time)

        self.render_graph(sim_times, const_value = 'length_of_ts')
    
    def graph_simulation_k(self):

        sim_times = []
        for k in range(10, self.length_of_ts + 1, 10):
            generate_series(n=self.number_of_ts, k=k, homogeneous=True)
            with open("dtw_data.json", "r") as f:
                data_json = json.load(f)

            start_time = time.time()
            apply_fusion(data_json)
            sim_times.append(time.time() - start_time)

        self.render_graph(sim_times, const_value = 'number_of_ts')

    def render_graph(self, sim_times, const_value):
        
        n_values = range(10, self.number_of_ts+1, 10)
        # Create a figure and a set of subplots
        fig, ax1 = plt.subplots()

        # Plot sim_times on y axis and n_values on x axis
        ax1.plot(n_values, sim_times, color='blue', marker='o')

        # Set labels for x and y axis
        if (const_value == 'length_of_ts'):
            ax1.set_xlabel('number of ts')
            ax1.set_ylabel('dtw cost', color='blue')
            plt.title('Cost of DTW with respect to number of ts, length of ts = 100')
            plt.savefig('dtw_simulation_n.png')
        else:
            ax1.set_xlabel('length of ts')
            ax1.set_ylabel('dtw cost', color='blue')
            plt.title('Cost of DTW with respect to length of ts, number of ts = 100')
            plt.savefig('dtw_simulation_k.png')
        
def generate_series(n: int = 100, k=100, homogeneous=True):
    # Generate two series of 1000 tuples of (x, y, z) values between [0, 100]
    data = {}

    if homogeneous == True:
        for i in range(n):
            series = [
                (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
                for _ in range(k)
            ]
            data[f"series{i+1}"] = series

        with open("dtw_data.json", "w") as f:
            json.dump(data, f)
    else:
        for i in range(n):
            ts_length = random.randint(k/2, k)
            series = [
                (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
                for _ in range(ts_length)
            ]
            data[f"series{i+1}"] = series

        with open("dtw_data_non_homogeneous.json", "w") as f:
            json.dump(data, f)



class HacMethod(Enum):
    COMPLETE = "complete"
    WARD = "ward"


def apply_fusion(data_json, hac_method: HacMethod | str = "complete", similarity_barrier=0.95):
    """
    using dtw_ndim multi-dimensional algorithm

        Parameters
    ----------
    hac_method: HacMethod
        Method used for hierarchical agglomerative clustering, complete or ward

    similarity_barrier: float
        The similarity barrier to be used for the fusion process

    Returns
    -------
    fusion_res_matrix: np.ndarray
        The boolean fusion result matrix of matching drone packages

    """
    if isinstance(hac_method, str):
        hac_method = HacMethod[hac_method.upper()]

    if hac_method not in (HacMethod.COMPLETE, HacMethod.WARD):
        raise ValueError("hac_method must be 'complete' or 'ward'")

    if not 0 <= similarity_barrier <= 1:
        raise ValueError("similarity_barrier must be between 0 and 1")

    s = []

    for k, v in data_json.items():
        s.append(np.array(v, dtype=np.double))


    cost_matrix = dtw_ndim.distance_matrix_fast(s, parallel=True)
    return (cost_matrix)

if __name__ == "__main__":

    res_list = []

    for simulation_set in simulation_sets:
        simulation = Simulation(**simulation_set)
        res = simulation.execute()
        res_list.append(res)

    df = pd.DataFrame(res_list)
    df.to_csv("dtw_performance_test_results.csv", float_format='%.5f')
    df.to_excel("dtw_performance_test_results.xlsx", float_format='%.5f')

    # graph simulation
    
    simulation = Simulation(**simulation_sets[1])
    simulation.graph_simulation_n()
    simulation.graph_simulation_k()

import json
import os

import matplotlib.pyplot as plt

from math import *
from collections import defaultdict

class PrepareData():
    ''' 
    Class with methods for preparing data from JSON file
    '''
    def __init__(self) -> None:
        self.barcode_data = None

    def read_barcode_data(self, barcode_json_path: str) -> list:
        ''' 
        Returns data from JSON file, sorted with NNs layers
        Params:
            barcode_json_path - path to file with data
        Return:
            layer1, ..., layerN - N lists with data for each layer
        '''
        # Read data from JSON file
        with open(barcode_json_path, 'r') as data:
            barcodes = json.load(data)

        # Define a data structure to store the barcode data
        layers = defaultdict(list)

        # Collecting barcode data into defined data structure
        for epoch in barcodes:
            for layer, value in barcodes[epoch].items():
                layers[layer].append(value)

        return tuple(layers[layer] for layer in sorted(layers.keys()))

    def load_barcode_data(self, amounts: list, base_path: str) -> dict:
        ''' 
        Call read_barcode_data func and put it in one dict with amount of data
        as a key and a list of data as a value
        Params:
            amounts - list with data amount numbers;
            base_path - common path to all files with data 
        Return:
            Dictionary with data amounts as keys and the data as values 
        '''
        # Collecting data into the dictionary per amount of data
        barcode_data = {}
        for amount in amounts:
            path = f"{base_path}_DataAmount{amount}/barcodes/barcode_data.json"
            barcode_data[amount] = self.read_barcode_data(barcode_json_path=path)

        return barcode_data
    

class Calculations():
    ''' 
    Class with methods for calculation metrics: mean lifetime, entropy and norm entropy
    '''
    def __init__(self) -> None:
        self.mean_lifetime = None
        self.h = None 
        self.h_norm = None

    def calculate_homology_mean_lifetime(self, layer: list) -> float:
        ''' 
        Calculates the homogies mean lfietimes for one layer
        Params:
            layer - data for one layer.
        Return:
            Mean lifetime value
        '''
        last_epoch = layer[-1]['H0']
        lifetime = 0

        for i in range(len(last_epoch)):
            mean = last_epoch[i][1] - last_epoch[i][0]
            lifetime += mean 

        mean_lifetime = lifetime / len(layer)

        return mean_lifetime

    def calculate_homology_mean_lifetime_per_epochs(self, data_amount: list) -> dict:
        ''' 
        Calculates the homogies mean lfietimes for each epoch per one amount of data
        Params:
            data_amount - data for one data amount.
        Return:
            Dict with mean lifetime value per epochs
        '''
        total_lifetime = dict()
        epoch_num = 1

        for epoch in data_amount:
            curr_epoch = epoch['H0']
            lifetime = 0

            for i in range(len(curr_epoch)):
                mean = curr_epoch[i][1] - curr_epoch[i][0]
                lifetime += mean 

            mean_lifetime_part = lifetime / len(data_amount)

            total_lifetime[epoch_num] = mean_lifetime_part
            epoch_num += 1

        return total_lifetime

    def calculate_persist_entropy(self, layer: list) -> float:
        ''' 
        Calculates entropy for each layer in NNs
        Params:
            layer - data for one layer
        Return:
            Value of an entropy for with layer
        '''
        last_epoch = layer[-1]['H0']
        pair_difference = list()

        for pair in last_epoch:
            pair_difference.append(pair[1] - pair[0])
        
        result = list()
        for diff_pair in pair_difference:
            result.append(diff_pair / sum(pair_difference))

        h = 0
        for elem in result:
            if elem == 0:
                pass
            else:
                h =+ elem * log2(elem)

        return -h
    
    def calculate_norm_persist_entropy(self, layer: list) -> list:
        ''' 
        Calculates normalized persistent entropy from basic entropy
        Params:
            layer - list with entropy values for each layer and each data amount
        Return:
            Normilized entropy for specific layer and amount of data
        '''
        last_epoch = layer[-1]['H0']
        pair_difference = list()

        for pair in last_epoch:
            pair_difference.append(pair[1] - pair[0])
        
        result = list()
        for diff_pair in pair_difference:
            result.append(diff_pair / sum(pair_difference))

        h = 0
        for elem in result:
            if elem == 0:
                pass
            else:
                h =+ elem * log2(elem)

        return -h / log2(len(pair_difference))
    
    def calculate_norm_persist_entropy_per_epochs(self, data_amount: list) -> dict:
        ''' 
        Calculates normalized persistent entropy from basic entropy for each epoch per one data amount.
        Params:
            data_amount - list with entropy values for each epoch for one data amount.
        Return:
            Normilized entropy for specific data amount and each epoch.
        '''
        total_entropy = dict()
        epoch_num = 1

        for epoch in data_amount:
            curr_epoch = epoch['H0']
            pair_difference = list()

            for pair in curr_epoch:
                pair_difference.append(pair[1] - pair[0])
            
            result = list()
            for diff_pair in pair_difference:
                result.append(diff_pair / sum(pair_difference))

            h = 0
            for elem in result:
                if elem == 0:
                    pass
                else:
                    h =+ elem * log2(elem)
                    
            total_entropy[f'{epoch_num}'] = -h / log2(len(pair_difference))
            epoch_num += 1

        return total_entropy
    

class PlotResults():
    '''  
    Class with methods for plotting results calculated in Calculation class
    '''
    def __init__(self):
        self.response = None

    def plot_homologies_mean_lifetime_for_one_layer(self, mean_lifetime_value: float, data_amounts: list, layer_name: str) -> None:
        ''' 
        Build the graphs for data amounts and mean lifetime values
        Params:
            mean_lifetime_value - list of mean lifetime values;
            data_amounts - list of data amounts for this experiment;
            layer_name - name of a NNs hidden layer the graph belongs to
        Return:
            Graph for data amounts and mean lifetime values 
        '''
        plt.plot(data_amounts, mean_lifetime_value, marker='o')
        plt.title(layer_name)
        plt.xlabel('Data amounts')
        plt.ylabel('Mean lifetime')
        plt.grid(True)

        plt.show() 

    def plot_graphs_grid(self, data_amounts: list, layers: list, label: str, layers_num: int) -> None:
        ''' 
        Plots a grid of graphs for mean lifetime through the data amounts for each layer
        Params:
            data_amounts - list with data amounts;
            layers - list with lists with layer data
        Return:
            None
        '''
        # Create figure and grid with X rows and 1 column with common X axis
        fig, axes = plt.subplots(layers_num, 1, figsize=(10, 8), sharex=True)

        # Fill in the graphs
        for i, ax in enumerate(axes.flat):
            ax.plot(data_amounts, layers[i])
            ax.legend()
            ax.set_title(f'Layer {i+1}')
            ax.grid(True)

        # Label for common X axis
        fig.text(0.5, 0.04, 'Data amounts', ha='center', fontsize=12)
        fig.text(0.0001, 0.5, label, va='center', rotation='vertical', fontsize=12)

        # Add space between graphs
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Оставляем место для общей подписи
        plt.show()

    # ml - mean lifetime
    def plot_ml_first_points(self, layers: list, data_amount_index: int, data_amounts: list, layers_num: int) -> None:
        ''' 
        Plots graph for start mean lifetime points per each layer 
        Params:
            layers - list with mean lifetimes for each layer;
            data_amount - 0-0, 1-1000, 2-2000, 3-3000, 4-4000, 5-5000, 6-10000, 7-20000, 8-30000, 9-40000, 10-50000, 11-60000.
        Return:
            None
        '''
        l = [i for i in range(1, layers_num+1)]
        first_point = [lay[data_amount_index] for lay in layers]

        plt.plot(l, first_point, marker='o')
        plt.title(f'Mean lifetime start points per layer, batch={data_amounts[data_amount_index]}')
        plt.xlabel('Layer')
        plt.ylabel('Mean lfietime')
        plt.grid(True)

        plt.show()

    def plot_entropy_for_one_layer(self, ent_value: float, data_amounts: list, layer_name: str, name: str) -> None:
        ''' 
        Build the graphs for data amounts and entropy values
        Params:
            ent_value - list of entropy values;
            data_amounts - list of data amounts for this experiment;
            layer_name - name of a NNs hidden layer the graph belongs to
        Return:
            Graph for data amounts and entropy values 
        '''
        plt.plot(data_amounts, ent_value, marker='o')
        plt.title(layer_name)
        plt.xlabel('Data amounts')
        plt.ylabel(name)
        plt.grid(True)

        plt.show()


def save_model_metrics_to_json(json_path: str, model_name: str, accuracy: float, precision: float, recall: float, f1_score: float) -> None:
    ''' 
    This metod saves obtained models metrics into the JSON file
    Params:
        json_path - path to JSON file to store the data;
        model_name - models name as a key from saving data;
        accuracy - models classification metric;
        precision - models classification metric;
        recall - models classification metric;
        f1_score - models classification metric.
    Output: 
        None
    '''
    # Create a dict with model's metrics for saving im JSON file
    model_metrics = {
        model_name: {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    }

    # Save models metrics into JSON file
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)  # Load data from file

        # Add new data
        data.append(model_metrics)

        # Write new updated data into JSON file 
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    else:
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(model_metrics, file, ensure_ascii=False, indent=4)

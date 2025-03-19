# Topological Data Analysis (TDA) in Neural Networks

This repo created to study neural networks' structure using TDA methods. The main aim of this project is to study the NNs behaviour while training it on "small" datasets. To study the NNs structures while training we used TDA methods to obtain Betty's numbers (as called as "barcodes") from NNs weights (to obtain this numbers we used eXplain-NNs library - https://github.com/aimclub/eXplain-NNs). From Betty's numbers we calculated such metrics as mean life-time of homologies and persistent entropy in NNs structure.  The project is still in developing stage.


# Project structure

In this repo you can find a few folders with the names of dataset, on which we trained neural networks. Each such folder contains some Jupiter Notebooks with next pipelines: data preprocessing (not presented in such dataset as CIFAR-10 and MNIST), getting optimal NN hyperparameters, extracting "barcodes" from NNs weights and plotting the results. All the data structure we obtained from each experiment is described below.


# Output data structure

The repo is separated into few parts - each part for one dataset and Jupiter Notebooks to it. The folders for output data in this parts have the next structure:

- x_dataset - main folder
    
    - x_output - folder for experimental obtained data

        - exp_y - folder for each single experiment

            - weights_graphs_architecture_DataAmountn - folder with n/N datasets' parts and treir outputs 

                - barcodes - directory for barcodes' data

                    - epoch1

                    ...

                    - epochN - folders for each epochs during training loop

                        - layer1_barcode.png 
                        
                        ...

                        - layerN_barcode.png - barcode graph for each layer in each epoch

                    - barcode_data.json - JSON file with data about barcodes separated with epochs and layers of NN

                    - barcode_evaluation.json - JSON file with barcodes' evaluation also separated with epochs and layers of NN 

                - loss_and_rmse_architecture_DataAmountn.png - graph with train and test loss functions and Root Mean Square Error (RMSE) metric

                - weights_architecture_DataAmountn.pth - file with neural networks' weights
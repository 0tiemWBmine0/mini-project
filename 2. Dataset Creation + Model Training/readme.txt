This folder contains the entire process of model training. After preprocessing the wav audio data, we use the dataloader tool to process it, which is then used in the subsequent ResNet network.

test_picture, train_picture: 
		Audio feature pictures transferred here after data preprocessing, with the train folder storing the training dataset and the test folder storing the testing dataset, both supplied for retrieval by the Model.ipynb file.

index1.csv, index2.csv: 
		These files store the label data of the datasets and serve as markers in the creation of the dataloader.

Model.ipynb: 
		A runnable, complete model code that includes the creation of the dataloader dataset, the construction of the model network, training, and plotting of training metrics.

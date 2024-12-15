Explanation of the file contents for dataset preprocessing:

wav folder: 
		Audio raw data downloaded from the GitHub provided by QM.

index1.csv: 
		A CSV index established for the dataset; the original data is split, with seven-tenths of the original dataset used as training data, and index1 serves as the index for the training data.

index2.csv: 
		A CSV index established for the dataset; the original data is split, with three-tenths of the original dataset used as testing data, and index2 serves as the index for the testing data.

pre_process_1.py, pre_process_2.py: 
		Scripts for extracting audio features from the raw data in the wav folder, and saving the relationship graphs of features and time, which will be used as data for constructing the Dataloader later on. The two different files are for extracting different features.

data_1, data_2, data_3, data_4, data_5: 
		Preprocessed datasets composed of different audio features extracted from the wav files.

# DeLEG
**Deep Learning for EpiGenomics data to predict phenotype**


`00_model.lua` : Model architecture file with kernel sizes and layer details. The model architecture is loaded in model_training.lua file


`01_model_training.lua` : Consists of data loading, manipulation and training part. Saves the model and prints/plots the training loss vs iteration curve


`02_model_testing.lua` : Consists of loading test data and calculates accuracy. Loads the saved model from "model_training.lua" file. Prints the classwise and overall accuracy.

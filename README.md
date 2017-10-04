# DeLEG
**Deep Learning for EpiGenomics data to predict phenotype**


`00_model.lua` : Model architecture file with kernel sizes and layer details. The model architecture is loaded in model_training.lua file


`01_model_training.lua` : Consists of data loading, manipulation and training part. Saves the model and prints/plots the training loss vs iteration curve


`02_model_testing.lua` : Consists of loading test data and calculates accuracy. Loads the saved model from "model_training.lua" file. Prints the classwise and overall accuracy.

## Inspiration
The concept of implementing AI with Bioinformatics data.

## What it does?
The generated model is trained to distinguish the healthy and the diseased individuals in a set of mixed samples.

## How we built it?
Our workflow consists of 3 parts; 

i) Otsu thresholding for segmentation, 
ii) enrichment score of the window around TSS and 
iii) train a CNN (Convolutional Neural Network) for classification.

Otsu thresholding for segmentation: ==> It takes as an input the ChIP-seq data (Chromatin Immunoprecipitation massively parallel DNA sequencing to identify the binding sites of DNA-associated proteins). In this step, Otsu is used to remove the background and filter out peaks by thresholding the ChIP-seq data. It finally returns the "important" regions (peaks) of the raw data.

Enrichment score of the window around TSS: ==> In this step the input are the ChIP-seq data and peak regions that were identified in the previous step. The genes and their corresponding IDs were retrieved from the NCBI (National Center for Biotechnology Information) database and the previously identified peaks were annotated based on their corresponding genes and the specific window/region was extracted (data manipulation process). This step returns the enrichment score from the ChIP-seq data for each window around the TSS (Transcriptional Start Site) for each gene corresponding to the peaks.

Train a Convolutional Neural Network for classification: ==> The input now is the enrichment score of the fixed length sequences. This step we train the CNN with windows from 34 subjects in each class, totaling to about 20k windows in each class. After we get the trained model, we visualize the features learned by the network to determine the regions in the input that influenced the network's decision. Finally, it returns the classified probabilities for the window to lie in each phenotype.

## Challenges we ran into
The detection of “important” regions (the distinction between background noise and real peaks) in ChIP-seq data. These regions are further used for prediction, classification of the two different phenotypes and a better understanding of human epigenome.
Computational power; for example, space and RAM.
Accomplishments that we're proud of
Bridging the gap between Bioinformatics and Machine Learning. This model is a novelty in itself and could be used for various future research in health care and other sectors.

## What we learned?
1) Epigenetic modifications can be found not only in gene and TSS but also in intergenic regions. 
2) How to learn the prospect of Deep Learning in Massive Parallel DNA Sequencing data.

## What's next for DeLEG Deep Learning EpiGenomic model?
This model can be sold to hospitals, clinics, research centers, health and other laboratories.
It can be also sold to people to determine their health condition.
In general, epigenetics is the study of the heritable changes in gene expression that does not involve changes in the DNA sequence. It causes changes in the phenotype without affecting the genotype.Our model can further be trained for specific individual's lifestyle, such as drinking alcohol, smoking and it can be useful for studies which link the specific epigenetic change with a particular (or more than one) environmental factor.

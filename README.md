# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about various demographic characteristics (age, marital status, education, etc.) as well as what I believe is loan application data (loan duration, default). The goal is to use the various data to predict default rate.

The best performing model, by accuracy rating, was a VotingEnsemble model, generated via AutoML processes. This was identified by running both a HyperDrive and AutoML process in Azure, performing a classification task, and selecting the model with the highest accuracy value.

## Scikit-learn Pipeline
The HyperDrive processes included hyperparameter tuning. HyperDrive performed uniform sampling and grid search sampling on the model hyperparameters. These were chosen to strike a balance between proper space exploration and calculation time. Additionally, early stopping was specified with the BanditPolicy being selected to not too dramatically overshoot the best-fit iteration.

**What are the benefits of the early stopping policy you chose?**

## AutoML
The AutoML process, however, was black-boxed, generating an ensemble of 9 models, with weights ranging between 0.06 and 0.2.

## Pipeline comparison
The resulting models varied dramatically, which makes sense, given that AutoML checks across a wide range of models, whereas HyperDrive only iterates different hyperparameters across a single model. The end result is that the AutoML model is superior in the accuracy metric.

## Future work
A blending of the two techniques would give a stronger assessment of the 'true' optimal model; AutoML may have found the most optimal architecture, but HyperDrive could then pick up the slack with additional tuning to identify the best overall hyperparameters for this optimal model.

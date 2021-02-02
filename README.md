# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about various demographic characteristics (age, marital status, education, etc.) as well as what I believe is loan application data (loan duration, default). The goal is to use the various data to predict default rate.

The best performing model, by accuracy rating, was a VotingEnsemble model, generated via AutoML processes. This was identified by running both a HyperDrive and AutoML process in Azure, performing a classification task, and selecting the model with the highest accuracy value.

## Scikit-learn Pipeline
The data used for the SKLearn pipeline was provided pre-seeded within the AzureML instance.

The model used is a Logistic Regression, which was evaluated using the model's accuracy value.

The pipeline included hyperparameter tuning on the maximum iterations, as well as the regularization strength. It performed uniform sampling and grid search sampling on the model hyperparameters. Uniform sampling would select a value within a specified range randomly, and run the model from that. Grid Search would select same-spaced intervals within a specified range. The former tends to provide similar results for less computational cost.

Additionally, early stopping was specified with the BanditPolicy being selected to not too dramatically overshoot the best-fit iteration. Generally an early-stopping policy is good to have as to not spend too much time running iterations on the ML training that don't actually contribute to the training effectiveness. BanditPolicy makes sure that current training is within a certain threshold of the best run, ensuring that training doesn't drift too far away from the best run. The policy also does not kick in until a specified number of iterations have passed.

## AutoML
The AutoML process, however, was black-boxed, generating an ensemble of 9 models, with weights ranging between 0.06 and 0.2. These weights determine how influential that particular model is on the final verdict. The ensemble aggregates the results of several models to come to a stronger result than any single model could alone.

The AutoML process was set to solve a classification algorithm, and assess each variant on its accuracy, with the best accuracy rising to the top. Additionally 2-fold cross validation was used on the models.

Each model was run for a maximum of 100 iterations, with an l2 penalty, using the `saga` solver.

## Pipeline comparison
The resulting models varied dramatically, which makes sense, given that AutoML checks across a wide range of models, whereas HyperDrive only iterates different hyperparameters across a single model. The end result is that the AutoML model is superior in the accuracy metric.

## Future work
A blending of the two techniques would give a stronger assessment of the 'true' optimal model; AutoML may have found the most optimal architecture, but HyperDrive could then pick up the slack with additional tuning to identify the best overall hyperparameters for this optimal model. This would leverage a 'best of both worlds' of both approaches.

Also, due to the imbalance in the dataset's target, accuracy is likely a poor metric. F1 score, which is an aggregation of precision and sensitivity, would be a better metric.

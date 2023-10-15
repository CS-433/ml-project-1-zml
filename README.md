# CS-433 Machine Learning Project 1 
# Higgs Boson Classification

## Results 
The final result of this project yields 0.84 categorization accuracy and an F1 score of 0.757, placing us in the 5th out of 200 teams participating in the Higgs Boson challenge.

## Team
The project is accomplished by team **ZML** with members:

Marko Lisicic: @mrfox99

Lazar Milikic: @Lemmy00

Yurui Zhu: @ruiarui

## Guideline

To successfully run this project, modules including `numpy`, `matplotlib` and `csv` are required. And training and test 
file should be placed into `resources` folder with name `train.csv` and `test.csv`

 One can find in our project report the full description of the project. In addition to the report, we provide several notebooks that go even further in data analyses (to which we refer to the conclusions made in our report) and plot visualization of feature distributions, outliers, correlation, etc. 

## Project Outline

 The purpose of this project is to implement basic Machine learning models
  which include linear, ridge, and logistic regression, and then use them
  to train and later predict whether a particle with given features 
  is classified as a Higgs boson or not. As our input data has an abundance of missing values,
  outliers, and features with heavy-tailed distributions, besides the aforementioned models, preprocessing and feature engineering play a crucial role in improving classification accuracies. 
  The final result of this project yields 0.84 categorization accuracy and an F1 score of 0.757 `submission \#204396`.

## Code Structure
```
├── implementaions.py: Implementations of 6 ML function
├── run.py: Python file for regenerating our final/best prediction.
├── preprocessing.py: Preprocessing pipeline, including filling missing values, build sin/cos and polynomial feature expansion
├── utils.py: All utils functions such as MSE and cross-entropy, gradient calculation function, K-cross validation etc.
├── helper.py: Functions to help load the data, generate mini-batch iterator and create *.csv* files for submission.
├── cross_vaildation.py: Python file for running cross validaion to find the best hyper-parameter.
├── Exploratory.ipynb: Data exploratory analysis, where it contains plots and conclusions to support our preprocessing and model selection.
├── fine_tuning.ipynb: fine tuning the model and plot the result
├── learning_curves.ipynb: plot the learning curves for Rigde and logistic regression.
├── CS_433_Project_1.pdf: a 2-pages report of the complete solution.
├── README.md
└── resources
    ├── train.csv
    ├── test.csv
```


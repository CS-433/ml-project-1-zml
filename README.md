# CS-433 Machine Learning Project 1 
# Higgs Boson Classification

## Team
The project is accomplished by team **ZML** with members:

Marko Lisicic: @mrfox99

Lazar Milikic: @Lemmy00

Yurui Zhu: @ruiarui

## Project Outline

todo
With a Test Accuracy of 0.84


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
├── Report.pdf: a 2-pages report of the complete solution.
├── README.md
└── resources
    ├── train.csv
    ├── test.csv
```
## Guideline

To successfully run this project, modules including `numpy`, `matplotlib` and `csv` are required. And training and test 
file should be placed into `resources` folder named `train.csv` and `test.csv`

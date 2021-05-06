# Deep Reinforcement Learning Applied to Atari MsPacman

* Created a tabular Reinforcement Learning environment.
* Applied Q-learning in the environment created.
* Applied Soft Actor Critic to Atari MsPacman.
* Applied Deep Q-Network to Atari MsPacman.

## Packages
**Python Version**: 3.8.3

**Main Packages used**:
* opencv-python==4.5.1.48
* torch==1.8.1
* torchvision==0.9.1

**Requirements**: 
`<pip install -r requirements.txt>`

## Original Data
The original data was from [OpenAI Gym](https://gym.openai.com).

## Results


## Conclusion 
* SVM outperformed MLP
* MLP is more capable of computational training than SVM and has potential to outperform SVM when it is trained with bigger size of hidden layers.


## Specifications

### Jupyter Notebooks

All jupyter notebooks are available as ipynb.

* data_preparation: Clean the dataset
* MLP_optimisation: Apply a grid search for MLP
* SVM_optimisation: Apply a grid search for SVM
* MLP_testing: Test the MLP model
* SVM_testing: Test the SVM model

### Folders

* Code: Contains all of the codes implemented

### Files

* MLP_optimised.joblib: The optimised MLP model
* SVM_optimised.joblib: The optimised SVM model
* Report: The report for the whole project

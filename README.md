# KCNQ-PyTorch
Implementation of the KCNQ1 project in PyTorch


This was written in CoLab as a series of code blocks where I tried to comment in my reasoning. Briefly:
1. The problem is framed as a multitarget regression problem
2. The data is manipulated as specified
3. The NN is implemented with 2 hidden layers, leaky ReLU, dropout neurons before each hidden layer of size 32 (2 layers), with 4 output parameters that are continuous
4. The model is trained with a 80/20 train test split of the data
5. The model is trained with a k-fold cross validation (k=5) of the data.

I think some major outstanding issues are:
  - not normalizing the output data potentially leading to astronomical losses
  - syntax errors in the k-fold cross validation with generating the train and test splits for each fold
  - I didn't completely figure out the syntax for generating the model that is an average of all of the trained models in the k-fold cross validation
  - 
------------------------------------
Update: 231117

KCNQ1_script.py is updated python script, separate from KCNQ1.ipynb. 

- The basic architecture listed above is the same. 
- K-fold cross-validation has been correctly implemented and the data has been normalized. 
- The network does a binary prediction based on the cutoff values for normal/dysfunctional variant as listed in S. Phul 2022. Reference the github from this paper for training data. 
- Inference is performed in a separate script using 23 "new" variants unseen by the network. 

Outstanding issues: 

- Update model architecture to predict one (benign/pathogenic) classification per variant. If any of the 4 electrophysiological outputs are classified as dysfunctional, then mutant overall is classified as pathogenic. 
- Implement transfer learning, recycling 

KCNQ1_script.py is updated python script, separate from KCNQ1.ipynb. 
- Architecure updated , various tests performed 
<<<<<<< HEAD

* 241003
- Added penetrance-test/ and ikps_only/ folders with scripts to train respective models in each and training/testing data

* 241104
- Added fourEP_metrics/ with updated scripts predict the 4 EP metrics with model architecture changes
- Cutoff for classification for each EP metric is automated. Model now uses features from two cryoEM structures 
=======
- 
- Added penetrance-test/ and ikps_only/ folders with scripts to train respective models in each and training/testing data 
>>>>>>> 22571ca377f12d72c3e0c0f5fdb75a82834cf326

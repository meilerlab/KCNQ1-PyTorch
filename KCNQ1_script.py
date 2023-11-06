''' 
accg.
Writing script following suggestions from Eric's tutorial.
Main changes: added a dataset object, added batch normalization during training

12 biophys input params. 4 EP outputs. checks if output from training model is functional or dysfunction based on cutoff criteria from the S Phul paper - see isdysfunc(). 

- checking MCC  , # truths / total (e.g. # predicitions made by the model that agree with func/dysfunc criteria, divided by # total predicitions) 

* updates 20231102 to model_best_231101.py:
  - create trainAll() . copy of trainModel(), excluding validation part. 
  - TODO: add in additional criteria that if peak curr density is <17%, all 4 biophys params are considered dysfunctional 

# IMPORTANT: if i change trainModel(), I'll need to update trainAll() !!! 
# will also need to update model_inference.py if I update the training model here. 

'''
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import sklearn
import sklearn.metrics as skm
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import train_test_split

def binarizeTensor(intensor):
    ccrit = [0.55,1.15,1.30,0.80,1.70,0.70,0.75,1.25]
    Ikscut = 0.17 # if peak I is < Ikscut, then all 4 outputs should be considered pathogenic 

    binTensor = torch.vstack((torch.logical_or(intensor[:,0] < ccrit[0], intensor[:,0] > ccrit[1]),
                              torch.logical_or(intensor[:,1] > ccrit[2], intensor[:,1] < ccrit[3]),
                              torch.logical_or(intensor[:,2] > ccrit[4], intensor[:,2] < ccrit[5]),
                              torch.logical_or(intensor[:,3] < ccrit[6], intensor[:,3] > ccrit[7]))).to(torch.float).t()


    return binTensor

class BioData(Dataset):
    
    def __init__(self, evolFile, biophysFile): 
        self.inputs, self.outputs, self.binout = self.readFile(evolFile,biophysFile)

    def __len__(self):
        return self.outputs.size(0)

    #Returns a (featureTensor, label) tuple given an index
    def __getitem__(self, idx):
        return (self.inputs[idx,:], self.outputs[idx,:], self.binout[idx,:])
        
    # Reads input files, processes, returns 0/1 (dysfunc/func)
    def readFile(self,evolFile,biophysFile):
        #This code makes no checks that the rows of the CSVs are in order
        #If this is not the case, use a dictionary with the first column as the key
        featureTensor = torch.empty(0, 12)
        f = open(evolFile)
        
        for line in f:
            features = torch.tensor([float(x) if x != "nan" else 0. for x in line.strip().split(",")[1:13]])
            featureTensor = torch.vstack((featureTensor, features))
        f.close()

        outTensor = torch.empty(0, 4)
        f = open(biophysFile)
        for line in f:
            outputs = torch.tensor([float(x) if x != "nan" else 0. for x in line.strip().split(",")[1:]])
            outTensor = torch.vstack((outTensor, outputs))
        binTensor = binarizeTensor(outTensor)
        
        return (featureTensor, outTensor, binTensor)
        
class MODEL(nn.Module): # create NN class
    def __init__(self,hiddenNodes):
        super().__init__()
        
        self.model = nn.Sequential(

            nn.Linear(12, 32),
            nn.LeakyReLU(), # hidden layer activation
            nn.Dropout(0.2),
            
            nn.Linear(32, 8),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(8, 4),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.model(x)
        return x

def trainModel(trainLoader,validLoader,device="cuda" if torch.cuda.is_available() else "cpu"):

    device = torch.device("cpu")
    KCNQ = MODEL(hiddenNodes=32)
    KCNQ = KCNQ.to(device)
    #loss_fn = nn.MSELoss()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(KCNQ.parameters())
    epoch=0
    losses = []
    nTruths = []
    mccs = np.empty((0,4))
    KCNQ.train()
    nEpochs = 1200
    while epoch < nEpochs: 
        totalLoss=0.
        #for features, labels in dl:
        for inputs, labels, binary in trainLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            binary = binary.to(device)
            optimizer.zero_grad()
            outputs=KCNQ(inputs) 
            #print(torch.vstack((outputs,labels)))
            #loss = loss_fn(outputs,labels)
            #print(torch.vstack((outputs,binary)))
            loss = loss_fn(outputs,binary)
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
        epoch+=1 
        losses.append(totalLoss)

        # VALIDATION 
        with torch.no_grad():
            KCNQ.eval() # changes how some layers behave for evaluation
            
            preds = np.empty((0,4))
            truths = np.empty((0,4))
            for inputs, labels, binary in validLoader:
                inputs = inputs.to(device)
                outputs = KCNQ(inputs)
                #print(torch.vstack((outputs.cpu(),binary)))
                binout = torch.round(outputs)
                preds = np.vstack((preds,binout.cpu().numpy()))
                truths = np.vstack((truths,binary.numpy()))
            mcc = np.array([skm.matthews_corrcoef(truths[:,i],preds[:,i]) for i in range(preds.shape[1])])    
            mccs = np.vstack((mccs,mcc))
            match = np.equal(preds,truths)
            nTruths.append(np.sum(match)/match.size)
            KCNQ.train() # set back to training mode 

        print(f"Epoch {epoch}: Loss={losses[-1]} %Match={nTruths[-1]} MCC1={mccs[-1,0]} MCC2={mccs[-1,1]} MCC3={mccs[-1,2]} MCC4={mccs[-1,3]}")
        
def trainAll(dataset,fsave,device="cuda" if torch.cuda.is_available() else "cpu"):
    # Train based on all data. Use this to save model for inference.
    # This should be exactly the same as the first part of trainModel()

    device = torch.device("cpu")
    KCNQ = MODEL(hiddenNodes=32)
    KCNQ = KCNQ.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(KCNQ.parameters())
    epoch=0
    losses = []
    nTruths = []
    mccs = np.empty((0,4))
    KCNQ.train()
    nEpochs = 1200
    while epoch < nEpochs: 
        totalLoss=0.
        #for features, labels in dl:
        for inputs, labels, binary in dataset:
            inputs = inputs.to(device)
            labels = labels.to(device)
            binary = binary.to(device)
            optimizer.zero_grad()
            outputs=KCNQ(inputs) 
            loss = loss_fn(outputs,binary)
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
        epoch+=1 
        losses.append(totalLoss)

    torch.save(KCNQ,fsave)


def crossValidation(dataset):
    kf = KFold(n_splits=5, shuffle=True)
    for ifold, (trainIdx, validIdx) in enumerate(kf.split(range(len(dataset)))):
        print("FOLD %d"%(ifold+1))
        trainSet = Subset(dataset, trainIdx)
        validSet = Subset(dataset, validIdx)
        trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True)
        validLoader = DataLoader(validSet, batch_size=64, shuffle=True)
        trainModel(trainLoader, validLoader)


######################################################################
if __name__ == "__main__":
    #fig, ax = plt.subplots(2) 
    ds = BioData("./orig_data/full.whet.multimer.csv","./orig_data/a1q1.model_data.csv")

    #crossValidation(ds)

    # after crossValidation, train w/ all data . save trained model. 
    fsave="temp.pth" 
    dl = DataLoader(ds, shuffle=True, batch_size=64)
    trainAll(dl,fsave) 

    #display.plot(ax=ax1,name="precision recall")
    #plt.show()

# model copied from model_240604.first4wt.py
# predict a peak current density , adjust cutoffs 
# author: Ana C Chang-Gonzalez

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.model_selection import KFold
from torch.utils.data import Subset, Dataset, DataLoader, WeightedRandomSampler
from scipy import stats 

# for plots
import seaborn as sbn
from matplotlib.colors import LinearSegmentedColormap

##############################

class VariantDataset(Dataset):
    def __init__(self, featurecsv, labelcsv, excludeWT = False):
        self.featureDict = {}
        self.labelDict = {}
        self.readFeatures(featurecsv)
        self.readLabels(labelcsv)

        # remove variants not found in both input files 
        self.featureDict = {k: v for k, v in self.featureDict.items() if k in self.labelDict}
        self.labelDict = {k: v for k, v in self.labelDict.items() if k in self.featureDict}

        if excludeWT:
            self.variantNames = [name for name in self.featureDict.keys() if name[0] != name[-1]]

        else:
            self.variantNames = [name for name in self.featureDict.keys()]
            if set(self.variantNames) != set([name for name in self.labelDict.keys()]):
                # sort to make the two dictionaries the same order
                keys = self.featureDict.keys()
                dict2 = {key: self.labelDict[key] for key in keys}
                self.labelDict=dict2 
            # check again 
            if set(self.variantNames) != set([name for name in self.labelDict.keys()]):
                print("UH OH SISTERS!")
                exit()
        
    def __len__(self):
        return len(self.variantNames)

    def __getitem__(self, idx):
        return (self.featureDict[self.variantNames[idx]], self.labelDict[self.variantNames[idx]])

    def readFeatures(self, featurecsv):
        f = open(featurecsv)
        for line in f:
            parts = line.split(",")
            sel_features = parts[1:] # parts[1:11] + parts[12:]  
            features = torch.tensor([float(x) for x in sel_features])
            name = parts[0].strip()
            self.featureDict[name] = features

        f.close()
    
    def readLabels(self, labelcsv):
        f = open(labelcsv)
        for line in f:
            parts = line.split(",")
            name = parts[0].strip()
            #labels = torch.tensor([float(x) if x.strip() != "nan" else 0. for x in parts[1:]])
            raw = [float(x) if x.strip() != "nan" else 0. for x in parts[1:2]]
            #labels = torch.ones(4)
            labels = torch.ones(1)
            #labels[0]=raw[0]
            #if raw[0] < 0.17:
            #    labels[0] = 0.
            #    labels[1] = 0.
            #    labels[2] = 0.
            #    labels[3] = 0.
            #if raw[0] < 0.36 or raw[0] > 1.36: ## accg. use distribution of these variants 
            if raw[0] < 0.55 or raw[0] > 1.15: # published . train with above. test with this. predictions did better.. 
                labels[0] = 0.
            #if raw[1] > 1.30 or raw[1] < 0.80:
            #    labels[1] = 0.
            #if raw[2] > 1.70 or raw[2] < 0.70:
            #    labels[2] = 0.
            #if raw[3] < 0.75 or raw[3] > 1.25:
            #    labels[3] = 0.

            self.labelDict[name] = labels
        f.close()

    def znormFeat(self):  
        # exlcude first 4 cols since these identify perturbing vars
        stacked_features = torch.stack([v[4:] for v in self.featureDict.values()]) 
        #stacked_features = torch.stack(list(self.featureDict.values()))
        # calc mean and std of stacked tensor
        mean = stacked_features.mean(dim=0)
        std = stacked_features.std(dim=0)
        znorm_stacked_feat = (stacked_features - mean) / std
        minval=torch.min(znorm_stacked_feat)
        maxval=torch.max(znorm_stacked_feat)
        normtensor = (znorm_stacked_feat - minval) / (maxval - minval)        

        for i,key in enumerate(self.featureDict):
            #self.featureDict[key] = (self.featureDict[key] - mean) / (std + 1e-8)  
            self.featureDict[key] = torch.cat((self.featureDict[key][0:4],normtensor[i]),dim=0)
            #self.featureDict[key] = torch.cat((self.featureDict[key][0:4],znorm_stacked_feat[i]),dim=0)
        return mean,std,minval,maxval


    def znormFeatTrans(self,mean,std,minval,maxval):
        stacked_features = torch.stack([v[4:] for v in self.featureDict.values()]) 
        znorm_stacked_feat = (stacked_features - mean) / std
        normtensor = (znorm_stacked_feat - minval) / (maxval - minval)        
        for i,key in enumerate(self.featureDict):
            self.featureDict[key] = torch.cat((self.featureDict[key][0:4],normtensor[i]),dim=0)
        

    def znormContLabel(self):   # only use if label is continuous 
        stacked_features = torch.stack(list(self.labelDict.values()))
        ## calc mean and std of stacked tensor
        #mean = stacked_features.mean(dim=0)
        #std = stacked_features.std(dim=0)
        #znorm_stacked_feat = (stacked_features - mean) / std
        #self.labelDict = {key: (value - mean) / std for key, value in self.labelDict.items()}

        # compress to 0-1: 
        minval=min(stacked_features)
        maxval=max(stacked_features)
        self.labelDict = {key: (value - minval) / (maxval - minval) for key, value in self.labelDict.items()}
        
        #print(self.labelDict)
        #quit()

# random forest
#class Network(

class L2Normalization(nn.Module):
    def __init__(self, dim=1):
        """
        L2 Normalization layer.

        Parameters:
        dim (int): The dimension along which to normalize.
        """
        super(L2Normalization, self).__init__()
        self.dim = dim

    def forward(self, x):
        norm = x.norm(p=2, dim=self.dim, keepdim=True)
        return x / norm

### for logistic regression 
#class Network(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.linin = nn.Linear(14, 1) 
#        self.fa = nn.Sigmoid()
#    def forward(self, x):
#        x = self.linin(x)
#        x = self.fa(x)
#        return x

## for ANN - from model_240605.py version  , best after 240930 testing, 26 feat
#class Network(nn.Module):
#    def __init__(self):
#        super().__init__()
#        #super(Network,self).__init__()
#        self.linin = nn.Linear(26,4)
#        self.layer2 = nn.Linear(4,1)
#        self.relu = nn.LeakyReLU()
#        self.fa = nn.Sigmoid()
#        self.dropout1 = nn.Dropout(p=0.33)
#        self.dropout2 = nn.Dropout(p=0.1)
#        self.dropout3 = nn.Dropout(p=0.05)
#        self.norm = nn.BatchNorm1d(4) 
#
#    def forward(self, x):
#        x = self.dropout2(x)
#        x = self.linin(x)
#        ## don't need batch normalization during testing w only perturbing variants
#        #x = self.norm(x)  
#        x = self.relu(x)
#        x = self.dropout3(x)
#        x = self.layer2(x)
#        x = self.fa(x)
#        return x


### for ANN - ORIGINAL
#class Network(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.linin = nn.Linear(12, 32) 
#        self.layer1 = nn.Linear(32, 12)
#        self.layer2 = nn.Linear(12, 4)
#        self.relu = nn.LeakyReLU()
#        self.fa = nn.Sigmoid()
#        self.dropout1 = nn.Dropout(p=0.33)
#        self.dropout2 = nn.Dropout(p=0.2)
#        self.dropout3 = nn.Dropout(p=0.05)
#
#    def forward(self, x):
#        x = self.dropout3(x)
#        x = self.linin(x)
#        x = self.relu(x)
#        x = self.dropout1(x)
#        x = self.layer1(x)
#        x = self.relu(x)
#        x = self.dropout1(x)
#        x = self.layer2(x)
#        x = self.fa(x)
#        return x

### for ANN - from model_240605.py version  - 240930: this model predicted every value in test set as dysfunctional. then I removed dropout, and it behaved the same as the 12->4 model.  
## with 26 input features, loss using this ann is lower than the 26-4-1 model, but more variables of the test set are incorrect . 
#class Network(nn.Module): 
#    def __init__(self):
#        super().__init__()
#        #super(Network,self).__init__()
#        self.linin = nn.Linear(26, 13) #26,32
#        self.layer1 = nn.Linear(13,4) #32,32)
#        self.layer2 = nn.Linear(4,1) #32,1)
#        self.relu = nn.LeakyReLU()
#        self.fa = nn.Sigmoid()
#        self.l2_norm = L2Normalization(dim=1)  # l2norm along feature dimension
#        self.norm = nn.BatchNorm1d(20) 
#        self.dropout1 = nn.Dropout(p=0.33)
#        self.dropout2 = nn.Dropout(p=0.1)
#        self.dropout3 = nn.Dropout(p=0.05)
#
#    def forward(self, x):
#        x = self.dropout2(x)
#        x = self.linin(x)
#        #x = self.norm(x)
#        x = self.relu(x)
#        #x = self.l2_norm(x)  
#        x = self.dropout3(x)
#        x = self.layer1(x)
#        x = self.relu(x)
#        x = self.layer2(x)
#        x = self.fa(x)
#        return x

## for ANN - test on 240926
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        #super(Network,self).__init__()
        self.linin = nn.Linear(26, 64)
        self.layer2 = nn.Linear(64,32)
        self.layer3 = nn.Linear(32,16)
        self.layer4 = nn.Linear(16,1)
        self.relu = nn.LeakyReLU()
        self.fa = nn.Sigmoid()
        self.dropout1 = nn.Dropout(p=0.33)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.05)
        self.norm1 = nn.BatchNorm1d(64) 
        self.norm2 = nn.BatchNorm1d(32)
        self.norm3 = nn.BatchNorm1d(16)  

    def forward(self, x):
        x = self.dropout2(x)
        x = self.linin(x)
        #x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.layer2(x)
        #x = self.norm2(x)
        x = self.relu(x)
        x = self.layer3(x)
        #x = self.norm3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.fa(x)
        return x


def trainModel(dl, valid=None): #,save=False):
    net = Network()
    #criterion = nn.MSELoss(reduction="mean")
    criterion = nn.BCELoss() # mean is default reduction . (reduction="sum")
    #optimizer = optim.Adam(net.parameters(),lr=0.001) # good w logistic regression, 1k epochs
    #optimizer = optim.Adam(net.parameters(),lr=0.0001,weight_decay=1e-5)  # good w ann, 5k epochs 
    optimizer = optim.Adam(net.parameters(),lr=0.00001,weight_decay=1e-5) # lr=1e-5 better with ann 14-4-1
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
    epoch = 0
    trainLoss = []
    validLoss = []
    endPreds = None
    endTrues = None
    #while epoch < 100000:  # for 12-13
    while epoch < 20000:  # for 12-4-1
    #while epoch < 15000:  #5000:  # for cross validation 5k was ok with batch size=100
    #while epoch < 2000: 
        running_loss = 0.
        running_num = 0
        net.train()
        for data, labels in dl:
            #optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            #print("num perturb:" , sum(torch.sum(data[:,0:4],1) != 0.))
            for param in net.parameters():
                param.grad = None
            loss.backward()
            # to help with exploding gradients? 
            #torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            #clip_value=1
            #for p in net.parameters():
            #    p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
            optimizer.step()
            running_loss += loss.item()
            running_num += data.size(0)
        trainLoss.append(running_loss/running_num)
        if valid != None:
            valid_loss = 0.
            valid_num = 0
            preds = torch.empty(0,1)
            trues = torch.empty(0,1)
            net.eval()
            with torch.no_grad():
                for data, labels in valid:
                    outputs = net(data)
                    select = torch.sum(data[:,0:4],1) != 0.
                    #print(valid.variantNames)
                    #exit()
                    preds = torch.vstack((preds,outputs[select,:]))
                    trues = torch.vstack((trues,labels[select,:]))
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    valid_num += sum(select).item()
                    #data.size(0)
                #print(trues.size(0))
                endTrues = trues
                endPreds = preds
                
            validLoss.append(valid_loss/valid_num)
            #print(epoch, trainLoss[-1], validLoss[-1])
        epoch += 1
    plt.plot([i for i in range(len(trainLoss))],trainLoss,label='train')
    plt.plot([i for i in range(len(validLoss))],validLoss,label='validation')
    #plt.ylim([0,1])
    plt.legend()
    plt.savefig('loss.png')
    net.eval()

    #print("per fold:")
    #mcc=skm.matthews_corrcoef(endTrues[:],(endPreds[:]>0.5))
    #print("regular mcc:",mcc) 

    return (endTrues, endPreds,net)

def makeWeightVec(ds): #,pwt):
    vec = []

    # increase weight of perturbing variants
    for data, label in ds:
        if torch.sum(data[0:4]) != 0.:
            vec.append(3.)
        else:
            vec.append(1.)

    ## this is not great for performance.. set random subset of non-perturbing variants to 0 
    #ones = [i for i, value in enumerate(vec) if value == 1]
    #nChange = int(len(ones) * 0.5)
    #iChange = random.sample(ones, nChange)
    #for i in iChange:
    #    vec[i] = 2

    return vec

def crossValid(ds, fold=5):
    splits = KFold(n_splits=fold, shuffle=True)
    trues = torch.empty(0,1)
    preds = torch.empty(0,1)
    #perturb_wts=np.arange(1,fold+1)
    #print(perturb_wts)
    #quit()
    for ifold, (trainIdx, validIdx) in enumerate(splits.split(np.arange(len(ds)))):
        #print(len(trainIdx))
        trainSamp = Subset(ds, trainIdx)
        validSamp = Subset(ds, validIdx)
        #sampler = WeightedRandomSampler(makeWeightVec(trainSamp,perturb_wts[ifold]),len(trainSamp)) #,replacement=False)
        sampler = WeightedRandomSampler(makeWeightVec(trainSamp),len(trainSamp)) #,replacement=False)
        trainLoader = DataLoader(trainSamp, batch_size=100, sampler=sampler)
        validLoader = DataLoader(validSamp, batch_size=100, shuffle=True)
        foldTrues, foldPreds , _ = trainModel(trainLoader, valid=validLoader)
        trues = torch.vstack((trues, foldTrues))
        preds = torch.vstack((preds, foldPreds))
    #print(trues.size())
    #print(preds)

    figauprc,axauprc=plt.subplots(1)    
    xlabels=["Iks","V1/2","tau_act","tau_deact"]    
    colors=['blue','orange','green','red']

    mccs , auprc = [],[]
    for i in range(1):
        mccThresh = [skm.matthews_corrcoef(trues[:,i], (preds[:,i] > thresh).to(torch.float)) for thresh in np.arange(0,1,0.01)]
        #print("thresh:",mccThresh)
        maxMcc = max(mccThresh)
        print("mcc:",maxMcc, mccThresh.index(maxMcc))
        mccs.append(maxMcc)

        ## regular MCC
        #mcc=skm.matthews_corrcoef(trues[:,i],(preds[:,i]>0.5))
        #mccs.append(mcc)        

        precision,recall,_ = skm.precision_recall_curve(trues[:,i],preds[:,i])
        auprc.append(skm.auc(recall,precision))
        
        # plot prc 
        axauprc.plot(recall,precision,linestyle='-',marker='o',c=colors[i],lw=2,label='%s AUPRC=%.2f' % (xlabels[i],auprc[i]))

    print("max mcc:",mccs)
    rcut=0.5 
    mcc=skm.matthews_corrcoef(trues[:]>rcut,(preds[:]>rcut))
    print("regular mcc:",mcc) 

    #print(foldRes)
    #print(np.mean(foldRes),np.std(foldRes))

    # auprc
    print("auprc:")
    print(auprc)

    plt.legend()
    plt.show()

    return trues,preds

##############################
def runTestSet(ifnModel,testDS,noval=0):
    KCNQ1 = torch.load(ifnModel)
    criterion = nn.BCELoss(reduction="sum")
    test_loss = 0.
    test_num = 0
    testLoss = []
    preds = torch.empty(0,1)
    trues = torch.empty(0,1)
    with torch.no_grad():
        KCNQ1.eval()
        for data, labels in testDS:
            outputs = KCNQ1(data)
            preds = torch.vstack((preds,outputs))
            trues = torch.vstack((trues,labels))
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_num += data.size(0)
            #print(trues.size(0))
            endTrues = trues
            endPreds = preds
            
        testLoss.append(test_loss/test_num)
        #print(testLoss[-1])

    if (noval): # the "GT" is dummy for this dataset, so don't need to perform validation. 
        # Rather, print predictions for each variant 
        analyzePreds(testDS,preds)
        quit()

    plt.plot([i for i in range(len(testLoss))],testLoss)
    mccs = []
    #thresh=[0.57,0.80,0.81,0.61] # thresholds determined from train/crossVal of original data 
    #thresh=[0.6,0.69,0.7,0.72]
    thresh=[0.5,0.5,0.5,0.5] # generic
    #thresh=[0.5,0.44,0.65,0.53] 
    for i in range(1):
        mcc = skm.matthews_corrcoef(trues[:,i], (preds[:,i] > thresh[i]).to(torch.float))
        mccs.append(mcc)
    print(mccs)

    ##########
    # for auroc
    xlabels=["Iks","V1/2","tau_act","tau_deact"]    
    colors=['blue','orange','green','red']
    figauc,axauc=plt.subplots(1) 
    for i in range(1):
        fpr,tpr,t=skm.roc_curve(trues[:,i],preds[:,i])
        auc=skm.auc(fpr,tpr)
        axauc.plot(fpr,tpr,linestyle='-',marker='o',c=colors[i],lw=2,label='%s AUROC=%.2f' % (xlabels[i],auc))
    figauc.text(0.5,0.03,'False Positive Rate',ha='center')
    axauc.set_ylabel('True Positive Rate')
    #for i in range(4):
    axauc.plot([0,1],[0,1],color='k',linestyle='--',lw=0.5)
    axauc.legend(loc='lower right')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.title('ROC, test set')
    #plt.savefig('auroc.png')

    ##########
    # for auprc    
    figauprc,axauprc=plt.subplots(1)
    ftp=sum(trues)/len(trues) # fraction true positives 
    for i in range(1):
        precision,recall,_ = skm.precision_recall_curve(trues[:,i],preds[:,i])
        auprc = skm.auc(recall,precision)
        axauprc.plot(recall,precision,linestyle='-',marker='o',c=colors[i],lw=2,label='%s AUPRC=%.2f' % (xlabels[i],auprc))
        axauprc.plot([0,1],[ftp[i],ftp[i]],c=colors[i],linestyle='--',lw=0.5)
        axauprc.text(0,ftp[i],xlabels[i])

    axauprc.set_ylabel('Precision')
    axauprc.set_xlabel('Recall')
    #for i in range(4):
    axauprc.legend(loc='lower right')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.title('Precision-Recall, test set')    
    #plt.savefig('auprc.png')
    #plt.show()

    ##########
    # variant-specific results 
    residues=testDS.variantNames[:]

    binaryPred = torch.zeros(len(preds), 1)
    for j in range(len(preds)): 
        for i in range(1): 
            binaryPred[j,i] =  1 if preds[j,i]>thresh[i] else 0

    match = np.equal(binaryPred,trues) # individual outputs 
    func=np.where((binaryPred == 1 ) , 'normal', 'dysfunctional')

    # define colors
    colors = ( (1.0,0,0) , ( 0,0,1) ) 
    cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
    
    #A = np.array([[f'{func[i, j]:.2f}' for j in range(4)] for i in range(23)])
 
    plt.figure()
    ax = sbn.heatmap(match,annot=func,fmt='',cmap=cmap,xticklabels=xlabels,yticklabels=residues)
    #ax = sbn.heatmap(match,annot=True,cmap=cmap,xticklabels=xlabels,yticklabels=residues)
    #plt.xticks(rotation=10) 
    
    # colorbar labels 
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25,0.75])
    colorbar.set_ticklabels(['Wrong','Correct'])
    plt.title("Inference results from pytorch model") 

    ##### 
    # print list of variants model got at least EP metric wrong
    right,wrong=[],[]
    for i in range(len(residues)): 
        if (residues[i][0]==residues[i][len(residues[i])-1]):
            continue 
        #print(residues[i]," :",end=" " )
        for j in range(1):
            #print(xlabels[j],"= pred=",func[i][j].item()," prob=",preds[i][j].item()," gt=",trues[i][j].item()," match?=",match[i][j].item())
            print(residues[i]," ",xlabels[j]," prob=",preds[i][j].item()," gt=",trues[i][j].item()," match?=",match[i][j].item())

            ## just plot wrong ones
            #if (match[i][j].item()==0):
            #    wrong.append(residues[i])
            #    print(xlabels[j],"= pred=",func[i][j].item()," prob=",preds[i][j].item()," gt=",trues[i][j].item()," match?=",match[i][j].item())
            #else: 
            #    right.append(residues[i])

    #plt.savefig('heatmap.png')
    plt.show()
    return 

##############################
## for performance of previous models , see model_240604.py cfmodels function 

##############################
def analyzePreds(ds,preds):
    # these variants don't have known EP metrics. Therefore I will analyze prediction scores 

    residues=ds.variantNames[:]

    thresh=[0.5,0.5,0.5,0.5] # generic    

    pair = zip(residues, preds)
    sorted_pairs = sorted(pair, key=lambda x: extract_number(x[0]))
    sort_residues, sort_preds = zip(*sorted_pairs)
    #print(sorted_residues)
    #print(sorted_preds)

    for j in range(len(sort_residues)): 
        flag=0
        print(sort_residues[j],": " , sort_preds[j][0].item(), " " ,sort_preds[j][1].item(), " " ,sort_preds[j][2].item(), " " ,sort_preds[j][3].item())
        for i in range(4): 
            if (sort_preds[j][i]>0.3 and sort_preds[j][i]<0.7): 
                flag=flag+1
        if (flag>2): 
            resid=int(sort_residues[j][1:4])
            if (resid>245 and resid<362):
                print("  resid ", resid, " is candidate for checking")

##########
def extract_number(var):
    return int(var[1:4])

    #for i, char in enumerate(var):
    #    if char.isdigit():
    #        return int(var[i:])

##############################
def plotDist(ax,ds,color,scale=1,text=0): 


    sorted_items = sorted(ds.labelDict.items(), key=lambda item: item[1].item())  # Sort by value
    # Unpack the sorted keys and values
    sorted_keys = [key for key, value in sorted_items]
    sorted_values = [value.item() for key, value in sorted_items]

    ikps=[ds.labelDict[key].item() for key in ds.labelDict.keys()]
    ikps=sorted(ikps)
    xval=np.arange(0,len(ikps))    

    ax.scatter(xval*scale,sorted_values,s=5*scale,c=color)
    if (text==1): 
        for i, (key, value) in enumerate(zip(sorted_keys, sorted_values)):
            ax.text(i*scale, value, key, ha='right', fontsize=9)

    plt.figure()
    counts, edges, _ = plt.hist(ikps, bins=3, edgecolor='black')  
    mdpt = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
    print("midpoint:",mdpt)

    #plt.show()

##############################

### train w orig+updated july 2024 set. I will use intermediate set for test
#ds = VariantDataset("/dors/meilerlab/data/changga/kcnq1/get_feat/7xnk/filled_amber/features_results_7xnk_240911.csv","./in/train-val.model_data.csv") #,excludeWT=True)
ds = VariantDataset("./in/features_results_7xnk_240911.w_am_w_rmsf.csv","./in/train-val.model_data.csv") #,excludeWT=True)
ds1 = VariantDataset("/dors/meilerlab/data/changga/kcnq1/get_feat/8sik/filled_amber/features_results_8sik_240911.csv","./in/train-val.model_data.csv") #,excludeWT=True)

for key in ds1.featureDict:
    if key in ds.featureDict:
        ds.featureDict[key] = torch.cat((ds.featureDict[key], ds1.featureDict[key]))
    else:
        print("concatenated tensors don't have same # keys. aborting.")
        break

#print(ds.featureDict['A300A'])
#quit()
#tmean,tstd,zmin,zmax=ds.znormFeat() # z normalize features  # need mean,std to znorm test data 
#ds.znormContLabel() # z normalize labels

#fig,ax=plt.subplots(1)
#plotDist(ax,ds,'gray',1,0) # plot distribution of experimental values 
#exit()
##
trues,preds=crossValid(ds) 
quit()
##
sampler = WeightedRandomSampler(makeWeightVec(ds),len(ds))
#trainLoader = DataLoader(ds, batch_size=len(ds), sampler=sampler)
trainLoader = DataLoader(ds, batch_size=len(ds), sampler=sampler)
#modelTrues, modelPreds , _ = trainModel(trainLoader,valid=trainLoader,save=True) 
_,_,kcnq1=trainModel(trainLoader)
torch.save(kcnq1,'temp.pth')

#plt.show()
#quit()


#_,_,kcnq1=trainModel(ds)

##############################
## test model trained on original data on new variants

## test set here is "first" set of new variants 
#dtest = VariantDataset("/dors/meilerlab/data/changga/kcnq1/get_feat/7xnk/filled_amber/features_results_7xnk_240911.csv","./in/test.model_data.csv")
#dtest = VariantDataset("./in/features_results_7xnk_240911.w_am_w_rmsf.csv","./in/test.model_data.csv")
dtest = VariantDataset("./in/features_results_7xnk_240911.w_am_w_rmsf.csv","./in/test.model_data.csv") #,excludeWT=True)
dtest1 = VariantDataset("/dors/meilerlab/data/changga/kcnq1/get_feat/8sik/filled_amber/features_results_8sik_240911.csv","./in/test.model_data.csv") #,excludeWT=True)

for key in dtest1.featureDict:
    if key in dtest.featureDict:
        dtest.featureDict[key] = torch.cat((dtest.featureDict[key], dtest1.featureDict[key]))
    else:
        print("concatenated tensors don't have same # keys. aborting.")
        break



#dtest.znormFeatTrans(tmean,tstd,zmin,zmax) # z normalize features  # need mean,std to znorm test data 

#plotDist(ax,dtest,'cornflowerblue',7,1)
#ax.axhline(y=0.55, color='gray', linestyle='--', label='published',alpha=0.5)
#ax.axhline(y=1.15, color='gray', linestyle='--',alpha=0.5)
#ax.axhline(y=0.36, color='magenta', linestyle=':', label='new',alpha=0.5)
#ax.axhline(y=1.36, color='magenta', linestyle=':',alpha=0.5)
#ax.legend()
#plt.show()
##quit()

ifnModel='temp.pth' 
#ifnModel='saved_ann_26-4-1.pth' # used to show current progress for nu/vandy call 241001

## 
#testLoader = DataLoader(dtest, batch_size=len(dtest)) #, sampler=sampler)
#runTestSet(ifnModel,testLoader,0)
#quit()
runTestSet(ifnModel,dtest,0)



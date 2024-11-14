# model copied from ../240604/model_240604.first4wt.py & modified 
# predict all 4 EP metrics, auatomate cutoff selection 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset, Dataset, DataLoader, WeightedRandomSampler
from scipy import stats 
from scipy.stats import linregress

# for plots
import seaborn as sbn
from matplotlib.colors import LinearSegmentedColormap
##############################

torch.set_printoptions(precision=6, sci_mode=False)
##############################

class VariantDataset(Dataset):
    #def __init__(self, featurecsv, labelcsv, labelCut,excludeWT = False):
    def __init__(self, featurecsv, labelcsv, EPcut, nFeat,excludeWT = False):
        self.featureDict = {}
        self.labelDict = {}
        self.readFeatures(featurecsv)
        #self.reduceFeaturesPCA(nFeat) # reduce # features to {input} by PCA 
        self.readLabels(labelcsv,EPcut)

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

    def reduceFeaturesPCA(self,nFeat):
        feature_matrix = np.array([tensor.numpy() for tensor in self.featureDict.values()])
        # Normalize (standardize) the feature matrix.  This WILL NOT change the the feature dictionary. 
        scaler = StandardScaler()
        feature_matrix_normalized = scaler.fit_transform(feature_matrix)
        pca = PCA(n_components=nFeat)
        reduced_features = pca.fit_transform(feature_matrix_normalized)
        reduced_dict = {key: reduced_features[i] for i, key in enumerate(self.featureDict)}
        for key, features in reduced_dict.items():
            print(f"{key}: {features}")

        print(self.featureDict['V205M'])
        print(reduced_dict['V205M'])
        
        self.featureDict=reduced_dict 
            
        ## which features were kept?
        ## Feature importance analysis
        #print("\nFeature Importance in Each Principal Component:")
        #for i, component in enumerate(pca.components_):
        #    # Get the indices of features sorted by importance (descending order of absolute value)
        #    sorted_indices = np.argsort(np.abs(component))[::-1]
        #    print(f"\nPrincipal Component {i+1} - Top Contributing Features:")
        #    for idx in sorted_indices[:5]:  # Change '5' to see more features if needed
        #        print(f"Feature {idx}: Weight {component[idx]:.4f}")
        #
        #explained_variance = pca.explained_variance_ratio_
        #print("\nExplained Variance by Each Component:")
        #for i, variance in enumerate(explained_variance):
        #    print(f"Component {i+1}: {variance:.4%}")
        #quit()
    
    def readLabels(self, labelcsv, EPcut): 
        f = open(labelcsv)
        for line in f:
            parts = line.split(",")
            name = parts[0].strip()
            raw = [float(x) if x.strip() != "nan" else 1. for x in parts[1:]]
            labels = torch.zeros(4)
            if raw[0] < 0.17: # this cutoff is pretty important.. mcc of v1/2, tau* drops ~< 0.2 without it. 
                labels[0] = 1.
                labels[1] = 1.
                labels[2] = 1.
                labels[3] = 1.
            if raw[0] < EPcut[0][0] or raw[0] > EPcut[0][1]: 
                labels[0] = 1.                
            if raw[1] > EPcut[1][1] or raw[1] < EPcut[1][0]:
                labels[1] = 1.
            if raw[2] > EPcut[2][1] or raw[2] < EPcut[2][0]:
                labels[2] = 1.
            if raw[3] < EPcut[3][0] or raw[3] > EPcut[3][1]:
                labels[3] = 1.

            self.labelDict[name] = labels
        f.close()

### for logistic regression 
class LogRegModel(nn.Module):
    def __init__(self): 
        super(LogRegModel,self).__init__()
        self.linin = nn.Linear(20, 4)
        self.fa = nn.Sigmoid()
        # save param for z-normalization & loss 
        self.mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.std = nn.Parameter(torch.ones(1), requires_grad=False)
        
    def forward(self, x):
        x = self.linin(x)
        x = self.fa(x)
        return x

class ANNModel0(nn.Module):
    def __init__(self):
        super(ANNModel,self).__init__()
        self.linin = nn.Linear(20, 30)
        self.layer3 = nn.Linear(30, 10)                
        self.layer2 = nn.Linear(10, 4)
        self.relu = nn.LeakyReLU()
        self.fa = nn.Sigmoid()
        self.dropout1 = nn.Dropout(p=0.33)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.05)
        # save param for z-normalization & loss 
        self.mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.std = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x):
        x = self.linin(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.fa(x)
        return x
            
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0): 
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    
## for ANN - test on 241028
class ANNModel(nn.Module):
    def __init__(self):
        super().__init__()
        #super(Network,self).__init__()
        #self.linin = nn.Linear(14, 64)
        self.linin = nn.Linear(20, 40)
        self.layer2 = nn.Linear(40,16)
        self.layer3 = nn.Linear(16,8)
        self.layer4 = nn.Linear(8,4)
        self.relu = nn.LeakyReLU()
        self.fa = nn.Sigmoid()
        self.dropout1 = nn.Dropout(p=0.33)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.05)
        # save param for z-normalization & loss 
        self.mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.std = nn.Parameter(torch.ones(1), requires_grad=False)
        
    def forward(self, x):
        x = self.dropout2(x)
        x = self.linin(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.layer2(x)
        x = self.relu(x)
        #x = self.dropout3(x)        
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.fa(x)
        return x

def trainModel(dl, valid=None): #,save=False):

    ####################
    ### FIRST TRAIN LOGISTIC REGRESSION 
    #net = LogRegModel()
    net = ANNModel()
    criterion = nn.BCELoss() #reduction="sum")
    #optimizer = optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-5) #,lr=0.00001,weight_decay=1e-5)
    optimizer = optim.Adam(net.parameters(),lr=1e-5,weight_decay=1e-5)
    early_stopper = EarlyStopper(patience=20, min_delta=.0001)

    ##########
    # znormalize : Use training data to determine mean & std for normalizing 
    batch_samples,tot_samples,tot_sum = 0,0,0
    mean,std = 0.,0.

    #select = torch.sum(data[:,0:4],1) != 0.
    for data,_ in dl:
        batch_samples = data[:,4:].size(0)
        tot_samples += batch_samples
        tot_sum += data[:,4:].sum(dim=0)
        std += data[:,4:].std(dim=0) * batch_samples
    mean = tot_sum / tot_samples
    std /= tot_samples

    # fill last 3 entries so last 3 features are not normalized
    mean=torch.cat((torch.tensor([0,0,0,0]),mean))
    std=torch.cat((torch.tensor([1,1,1,1]),std))
    ## variables to save 
    net.mean.data = mean
    net.std.data = std

    ##########    
    epoch = 0
    trainLoss = []
    validLoss = []
    endPreds = None
    endTrues = None
    while epoch < 10000: #1000:
        running_loss = 0.
        running_num = 0
        net.train()
        for data, labels in dl:
            optimizer.zero_grad()
            data_norm=(data-mean)/std
            outputs = net(data_norm)
            loss = criterion(outputs, labels)
            #print("num perturb:" , sum(torch.sum(data[:,0:4],1) != 0.))
            #for param in net.parameters():
            #    param.grad = None
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_num += data.size(0)
        trainLoss.append(running_loss/running_num)
        if valid != None:
            valid_loss = 0.
            valid_num = 0
            preds = torch.empty(0,4)
            trues = torch.empty(0,4)
            net.eval()
            with torch.no_grad():
                for data, labels in valid:
                    select = torch.sum(data[:,0:4],1) != 0.
                    if (sum(select).item() == 0):
                        #print("none")
                        continue
                    data_norm = (data[select,:]-mean)/std                    
                    outputs = net(data_norm)
                    preds = torch.vstack((preds,outputs))
                    trues = torch.vstack((trues,labels[select,:]))
                    loss = criterion(outputs,labels[select,:])
                    valid_loss += loss.item()
                    valid_num += data_norm.size(0) # sum(select).item()
                #print(trues.size(0))
                endTrues = trues
                endPreds = preds
            validLoss.append(valid_loss/valid_num)
            if early_stopper.early_stop(valid_loss/valid_num):
                print("early stop at epoch=",epoch) 
                break                    
            
            #print(epoch, trainLoss[-1], validLoss[-1])
        epoch += 1
    plt.plot([i for i in range(len(trainLoss))],trainLoss,label='train')
    plt.plot([i for i in range(len(validLoss))],validLoss,label='validation')
    #plt.ylim([0,1])
    plt.legend()
    plt.savefig('loss.png')
    net.eval()

    mcc=[]
    for i in range(4):
        mcc.append(skm.matthews_corrcoef(endTrues[:,i],(endPreds[:,i]>0.5)))
    print("trainModel fold mcc:",mcc)


    #####################    
    #####################
    ## Initialize ANN w weights from logreg for training
    #ann_net = ANNModel()
    #criterion = nn.BCELoss() #reduction="sum")
    #optimizer = optim.Adam(ann_net.parameters(),lr=1e-4) #,lr=0.00001,weight_decay=1e-5)
    ## Transfer weights from the logistic regression model to the first layer of the ANN
    #with torch.no_grad():
    #    ann_net.linin.weight[:,:] = net.linin.weight  # Copy weights
    #    ann_net.linin.bias[:] = net.linin.bias        # Copy bias
    #
    #print(ann_net.linin.weight)
    #quit()

    
    return (endTrues, endPreds,net)

def makeWeightVec(ds): #,pwt):
    vec = []

    # increase weight of perturbing variants
    for data, label in ds:
        if torch.sum(data[0:4]) != 0.:
            vec.append(3.)
        else:
            vec.append(1.)
    return vec

def crossValid(ds, fold=5):
    splits = KFold(n_splits=fold, shuffle=True)
    trues = torch.empty(0,4)
    preds = torch.empty(0,4)
    plt.figure()
    for ifold, (trainIdx, validIdx) in enumerate(splits.split(np.arange(len(ds)))):
        #print(len(trainIdx))
        trainSamp = Subset(ds, trainIdx)
        validSamp = Subset(ds, validIdx)
        sampler = WeightedRandomSampler(makeWeightVec(trainSamp),len(trainSamp))
        trainLoader = DataLoader(trainSamp, batch_size=100, sampler=sampler)
        validLoader = DataLoader(validSamp, batch_size=100, shuffle=True)
        foldTrues, foldPreds , _ = trainModel(trainLoader, valid=validLoader)
        trues = torch.vstack((trues, foldTrues))
        preds = torch.vstack((preds, foldPreds))

    fig,ax=plt.subplots(2)
    figscat,axscat=plt.subplots(4)    
    xlabels=["Iks","V1/2","tau_act","tau_deact"]    
    colors=['blue','orange','green','red']
    
    maxmccs, mccs , auprc,auroc = [],[],[],[]
    for i in range(4):

        # for last fold, print preds v trues
        axscat[i].scatter(foldTrues[:,i],foldPreds[:,i],c=colors[i],edgecolors='k',label='%s'%(xlabels[i]))
        axscat[i].legend()
        
        #mccThresh = [skm.matthews_corrcoef(trues[:,i], (preds[:,i] > thresh).to(torch.float)) for thresh in np.arange(0,1,0.01)]
        ##print("thresh:",mccThresh)        
        #maxMcc = max(mccThresh)
        #maxmccs.append(maxMcc)

        # regular MCC
        mcc=skm.matthews_corrcoef(trues[:,i],(preds[:,i]>0.5))
        mccs.append(mcc)        

        # plot prc         
        precision,recall,_ = skm.precision_recall_curve(trues[:,i],preds[:,i])
        auprc.append(skm.auc(recall,precision))        
        ax[0].plot(recall,precision,linestyle='-',marker='o',c=colors[i],lw=2,label='%s AUPRC=%.2f' % (xlabels[i],auprc[i]))

        # plot roc 
        fpr,tpr,t=skm.roc_curve(trues[:,i],preds[:,i])
        auroc.append(skm.auc(fpr,tpr))
        ax[1].plot(fpr,tpr,linestyle='-',marker='o',c=colors[i],lw=2,label='%s AUROC=%.2f' % (xlabels[i],auroc[i]))
        
    #print("max mcc:",maxmccs)
    print("regular mcc:",mccs) 

    # auprc
    print("auprc:")
    print(auprc)
    # auroc
    print("auroc:")
    print(auroc)

    ax[0].legend()
    ax[1].legend()
    
    plt.show()

    return trues,preds

##############################
def runTestSet(ifnModel,testDS,noval=0):
    KCNQ1 = torch.load(ifnModel)
    criterion = nn.BCELoss() #reduction="sum")    
    test_loss = 0.
    test_num = 0
    testLoss = []
    preds = torch.empty(0,4)
    trues = torch.empty(0,4)
    with torch.no_grad():
        KCNQ1.eval()
        mean,std = KCNQ1.mean.detach(), KCNQ1.std.detach()
        for data, labels in testDS:
            data_norm =  (data - mean) / std
            outputs = KCNQ1(data_norm)
            preds = torch.vstack((preds,outputs))
            trues = torch.vstack((trues,labels))
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_num += data_norm.size(0)
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
    for i in range(4):
        mcc = skm.matthews_corrcoef(trues[:,i], (preds[:,i] > thresh[i]).to(torch.float))
        mccs.append(mcc)
    print(mccs)

    ##########
    # for auroc
    xlabels=["Iks","V1/2","tau_act","tau_deact"]    
    colors=['blue','orange','green','red']
    figauc,axauc=plt.subplots(1) 
    for i in range(4):
        fpr,tpr,t=skm.roc_curve(trues[:,i],preds[:,i])
        auc=skm.auc(fpr,tpr)
        axauc.plot(fpr,tpr,linestyle='-',marker='o',c=colors[i],lw=2,label='%s AUROC=%.2f' % (xlabels[i],auc))
        axauc.legend(loc='lower right')        
    figauc.text(0.5,0.03,'False Positive Rate',ha='center')
    axauc.set_ylabel('True Positive Rate')
    axauc.plot([0,1],[0,1],color='k',linestyle='--',lw=0.5)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.title('ROC, test set')
    #plt.savefig('auroc.png')
    #plt.show()
    #quit()
    ##########
    # for auprc    
    figauprc,axauprc=plt.subplots(1)
    ftp=sum(trues)/len(trues) # fraction true positives
    print(ftp)
    for i in range(4):
        precision,recall,_ = skm.precision_recall_curve(trues[:,i],preds[:,i])
        auprc = skm.auc(recall,precision)
        axauprc.plot(recall,precision,linestyle='-',marker='o',c=colors[i],lw=2,label='%s AUPRC=%.2f' % (xlabels[i],auprc))
        axauprc.plot([0,1],[ftp[i],ftp[i]],c=colors[i],linestyle='--',lw=0.5)
        axauprc.text(0,ftp[i],xlabels[i])
    axauprc.set_ylabel('Precision')
    axauprc.set_xlabel('Recall')
    for i in range(4):
        axauprc.legend(loc='lower right')
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.05,1.05])
    plt.title('Precision-Recall, test set')    
    #plt.savefig('auprc.png')
    #plt.show()
    #quit()
    ##########
    # variant-specific results 
    residues=testDS.variantNames[:]

    binaryPred = torch.zeros(len(preds), 4)
    for j in range(len(preds)): 
        for i in range(4): 
            binaryPred[j,i] =  1 if preds[j,i]>thresh[i] else 0

    match = np.equal(binaryPred,trues) # individual outputs 
    func=np.where((binaryPred == 0 ) , 'normal', 'dysfunctional')

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
def findEPCutoff(ifname):
    # read continuous labels for each EP metric and set an appropriate min/max cutoff to consider a variant benign usign the current perturbing variant trainign data 
    
    labels=[]
    f = open(ifname)
    for line in f:
        parts = line.split(",")
        name = parts[0].strip()
        raw = [float(x) if x.strip() != "nan" else 1. for x in parts[1:]]
        if (name[0]==name[-1]): 
            continue
        labels.append(raw)

    rcut=np.ones((len(labels[0]),2))
    window_size = 5
    lower,upper = 0.05,0.95     # percentiles to exclude 
    for ifeat in range(4):
        plt.figure()
        label = [array[ifeat] for array in labels]
        sorted_labels=np.array(sorted(label))

        # exclude the bottom/top XX %
        lower_bound = np.quantile(sorted_labels, lower)
        upper_bound = np.quantile(sorted_labels, upper)     

        ###########
        ## cubic fit
        #x=np.arange(0,len(sorted_labels))
        #coefficients = np.polyfit(x, sorted_labels, 3)  # Degree 3 for cubic fit
        #cubic_fit = np.poly1d(coefficients)
        ## Generate points for the fitted curve
        #x_fit = np.linspace(min(x), max(x), 200)
        #y_fit = cubic_fit(x_fit)
        ## Plotting the scatter points and the fitted curve
        plt.plot(sorted_labels,marker='o')
        #plt.plot(x_fit, y_fit, color='red', label=f'Cubic Fit')
        ###########

        # Calculate slope for every overlapping set of window_size points
        for j in range(2):
            if (j==0):
                # min cutoff
                data=sorted_labels[sorted_labels < 1.0]
                data=data[data>=lower_bound]
            else:
                # max cutoff
                data=sorted_labels[sorted_labels > 1.0]
                data=data[data<=upper_bound]
                #data=data[data<100] # . there are some entries = 10k.. exclude
            slopes=[]
            for i in range(len(data) - window_size + 1):
                x = np.arange(window_size) 
                y = data[i:i + window_size] 
                slope, _, _, _, _ = linregress(x, y)  # Perform linear regression
                slopes.append(slope)
    
            # Step 2: Calculate the difference between consecutive slopes
            slope_changes = np.diff(slopes)
        
            # Step 3: Find the index of the maximum change in slope
            # highest slope change: 
            max_index = np.argmax(np.abs(slope_changes)) # max
            change_index=max_index
            #del_max = np.delete(slope_changes, max_index)
            #change_index = np.argmax(np.abs(del_max))
            
            # Step 4: Identify the data points corresponding to this window
            start_index = change_index
            end_index = change_index + window_size
            change_points = data[start_index:end_index]
            
            print(f"Maximum slope change between windows occurs at index {change_index}")
            print("Data points in the window with maximum slope change:", change_points)
            rcut[ifeat][j]=np.mean(change_points)

        plt.axhline(y=rcut[ifeat][0], color='cyan', linestyle='--', label='min',alpha=0.5)
        plt.axhline(y=rcut[ifeat][1], color='cyan', linestyle='--', label='max',alpha=0.5)
        plt.ylim([-0.1,4])
        
        if ifeat==0:
            plt.axhline(y=0.55, color='gray', linestyle='--', label='published',alpha=0.5)
            plt.axhline(y=1.15, color='gray', linestyle='--',alpha=0.5)
        if ifeat==1:
            plt.axhline(y=0.80, color='gray', linestyle='--', label='published',alpha=0.5)
            plt.axhline(y=1.30, color='gray', linestyle='--',alpha=0.5)
        if ifeat==2:
            plt.axhline(y=0.70, color='gray', linestyle='--', label='published',alpha=0.5)
            plt.axhline(y=1.70, color='gray', linestyle='--',alpha=0.5)            
        if ifeat==3:
            plt.axhline(y=0.75, color='gray', linestyle='--', label='published',alpha=0.5)
            plt.axhline(y=1.25, color='gray', linestyle='--',alpha=0.5)

    plt.legend()
    #plt.show()

    print(rcut)
    return rcut
        
    
##############################
##############################
# main 

## select max/min cutoff values for EP data from training feature set
trainLabelsCSV="train-val.model_data.csv"
EPcut=findEPCutoff(trainLabelsCSV)
#EPcut=0

## Reduce feature inputs to nFeat
nFeat=10  

## train w orig+updated july 2024 set. I will use intermediate set for test
ds = VariantDataset("features_results_7xnk_240911.w_am_w_rmsf.csv",trainLabelsCSV,EPcut,nFeat) #,excludeWT=True)
ds1 = VariantDataset("features_results_8sik_240911_manual_reduce.csv",trainLabelsCSV,EPcut,nFeat) 

for key in ds1.featureDict:
    if key in ds.featureDict:
        ds.featureDict[key] = torch.cat((ds.featureDict[key], ds1.featureDict[key]))
    else:
        print("concatenated tensors don't have same # keys. aborting.")
        break

#print(ds.featureDict['V310S'])
#print(ds.featureDict['V310V'])
#quit()
trues,preds=crossValid(ds) 
quit()
###
#sampler = WeightedRandomSampler(makeWeightVec(ds),len(ds))
##trainLoader = DataLoader(ds, batch_size=len(ds), sampler=sampler)
#trainLoader = DataLoader(ds, batch_size=len(ds), sampler=sampler)
##modelTrues, modelPreds , _ = trainModel(trainLoader,valid=trainLoader,save=True) 
#_,_,kcnq1=trainModel(trainLoader)
#torch.save(kcnq1,'temp.pth')
#
##plt.show()
#quit()


#_,_,kcnq1=trainModel(ds)

##############################
## test model trained on original data on new variants

## test set here is "first" set of new variants 
dtest = VariantDataset("features_results_7xnk_240911.w_am_w_rmsf.csv","./in/test.model_data.csv",EPcut,nFeat) #,excludeWT=True)
dtest1 = VariantDataset("features_results_8sik_240911_manual_reduce.csv","./in/test.model_data.csv",EPcut,nFeat) #,excludeWT=True)

for key in dtest1.featureDict:
    if key in dtest.featureDict:
        dtest.featureDict[key] = torch.cat((dtest.featureDict[key], dtest1.featureDict[key]))
    else:
        print("concatenated tensors don't have same # keys. aborting.")
        break



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



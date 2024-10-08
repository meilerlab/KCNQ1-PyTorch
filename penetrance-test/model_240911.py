# author: ana c. chang-gonzalez 
# biophys+evolutionary to penentrance. regression 
# first test: this will take structural features & predict penentrance

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.model_selection import KFold
from torch.utils.data import Subset, Dataset, DataLoader, WeightedRandomSampler
from scipy import stats 

# for plots
import seaborn as sbn
from matplotlib.colors import LinearSegmentedColormap
#from adjustText import adjust_text

##############################

class PenetranceDataset(Dataset):
    def __init__(self, featurecsv, labelcsv, excludeWT = False):
        self.featureDict = {}
        self.labelDict = {}
        self.readFeatures(featurecsv)
        self.readLabels(labelcsv)

        # remove variants not found in both input files 
        self.featureDict = {k: v for k, v in self.featureDict.items() if k in self.labelDict}
        self.labelDict = {k: v for k, v in self.labelDict.items() if k in self.featureDict}
        # exclude WT entries?
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
            features = torch.tensor([float(x) if x.strip() != "nan" else 0. for x in parts[1:]]) # 1:13
            name = parts[0].strip()
            self.featureDict[name] = features
        f.close()
    
    def readLabels(self, labelcsv):
        f = open(labelcsv)
        next(f) # skip header row . 
        for line in f:
            parts = line.split(",")
            name = parts[0].strip()
            raw = [float(x) if x.strip() != "nan" else 0. for x in parts[1:]]
            labels = torch.ones(1) # penetrance
            labels[0]=raw[2]*.01
            self.labelDict[name] = labels
        f.close()

    def znorm(self):  
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

        #mean = torch.cat((torch.tensor([0,0,0,0]),mean),dim=0) # to not change last 4 cols
        #std = torch.cat((torch.tensor([1,1,1,1]),std),dim=0) # to not change last 4 cols
        for i,key in enumerate(self.featureDict):
            #self.featureDict[key] = (self.featureDict[key] - mean) / (std + 1e-8)  
            self.featureDict[key] = torch.cat((self.featureDict[key][0:4],normtensor[i]),dim=0)

#### for logistic regression 
#class Network(nn.Module):
#    def __init__(self,isize):
#        super(Network,self).__init__()
#        self.linin = nn.Linear(isize, 1) 
#        self.fa = nn.Sigmoid()
#    def forward(self, x):
#        x = self.linin(x)
#        x = self.fa(x)
#        return x

## for ANN : for biophys+evol features --> penetrance 
class Network(nn.Module):
    def __init__(self,isize):
        super(Network,self).__init__()
        self.linin = nn.Linear(isize, 64)
        self.layer2 = nn.Linear(64,32)
        self.layer3 = nn.Linear(32,1)
        #self.activation = nn.LeakyReLU()
        self.activation = nn.ReLU()
        self.fa = nn.Sigmoid() # sigmoid not recommended for regression tasks
        self.dropout1 = nn.Dropout(p=0.33)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.05)
        self.softmax = nn.Softmax()

    def forward(self, x):
        #x = self.dropout2(x)
        x = self.linin(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.fa(x)
        return x

def trainModel(sel_feat,dl, valid=None,save=False):
    f0,f1=sel_feat[0],sel_feat[1]
    num_feat = f1-f0
    assert(num_feat>0)
    net = Network(num_feat)
    criterion = nn.MSELoss() # MSE for regression task 
    #criterion = nn.BCELoss(reduction="sum")
    optimizer = optim.Adam(net.parameters(),lr=0.00001) #,weight_decay=0.5) 
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    epoch = 0
    trainLoss = []
    validLoss = []
    endPreds = None
    endTrues = None
    while epoch < 1000: # 2000: # 500: #1200:
        running_loss = 0.
        running_num = 0
        for data, labels in dl:
            optimizer.zero_grad()
            outputs = net(data[:,f0:f1])
            #print(data[:,f0:f1])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_num += data.size(0)
        trainLoss.append(running_loss/running_num)
        if valid != None:
            valid_loss = 0.
            valid_num = 0
            preds = torch.empty(0,1)
            trues = torch.empty(0,1)
            with torch.no_grad():
                net.eval()
                for data, labels in valid:                    
                    #outputs = net(data[:,f0:f1])
                    outputs = net(data[:,f0:f1])
                    select = torch.sum(data[:,0:4],1) != 0. # excl. nonperturb, biophys
                    preds = torch.vstack((preds,outputs[select,:])) 
                    trues = torch.vstack((trues,labels[select,:]))
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    valid_num += sum(select).item() 
                net.train()
                #print(trues.size(0))
                endTrues = trues
                endPreds = preds
                
            validLoss.append(valid_loss/valid_num)
            #print(epoch, trainLoss[-1], validLoss[-1])
        #scheduler.step(valid_loss) # update learning rate after every epoch
        epoch += 1

    plt.plot([i for i in range(len(trainLoss))],trainLoss,marker='o',markersize=4,label='train')
    plt.plot([i for i in range(len(validLoss))],validLoss,marker='x',markersize=4,label='validation')
    plt.legend()
    plt.ylim([0,0.03])
    plt.savefig('loss.png')

    #print(endTrues)
    #print(endPreds)

    ##### Fold validation metrics
    print("next fold")
    #figfold,axfold=plt.subplots(1)
    #print(trues)
    #axfold.scatter(trues,preds)
    #axfold.set_xlim([0,1])
    #axfold.set_ylim([0,1])
        
    # regular MCC
    rcut=0.5 
    mcc=skm.matthews_corrcoef(endTrues[:]>rcut,(endPreds[:]>rcut))
    print(" regular MCC:",mcc)
    
    precision,recall,_ = skm.precision_recall_curve(endTrues[:]>rcut,endPreds[:])
    auprc=skm.auc(recall,precision)
    tpr_base=(endTrues[:]>rcut).sum().item()/len(endTrues)
    print(" auprc=",auprc,"tpr_base=",tpr_base)
    fpr,tpr,t=skm.roc_curve(endTrues[:]>rcut,endPreds[:])
    auroc=skm.auc(fpr,tpr)
    print(" auroc=",auroc)
    r2 = r2_score(endTrues,endPreds)
    print(f' R²: {r2}')
    mae = mean_absolute_error(endTrues,endPreds)
    print(f' MAE: {mae}')
    ###########

    net.eval()

    if (save):
        torch.save(net.state_dict(),'temp.pth')

    return (endTrues, endPreds,net)

def makeWeightVec(ds):
    vec = []
    for data, label in ds:
        #if torch.sum(data[-4:]) != 0.:
        if torch.sum(data[0:4]) != 0.:
            vec.append(3.)
        else:
            vec.append(1.)
    return vec

def crossValid(sel_feat, ds, fold=5): 
    splits = KFold(n_splits=fold, shuffle=True)
    trues = torch.empty(0,1)
    preds = torch.empty(0,1)

    # for visualizing fold performance 
    xlabels=["penetrance"]
    figfold,axfold=plt.subplots(3)            
    rcut=0.5 

    for trainIdx, validIdx in splits.split(np.arange(len(ds))):
        trainSamp = Subset(ds, trainIdx)
        validSamp = Subset(ds, validIdx)
        sampler = WeightedRandomSampler(makeWeightVec(trainSamp),len(trainSamp))
        #trainLoader = DataLoader(trainSamp, batch_size=64, shuffle=True)
        trainLoader = DataLoader(trainSamp, batch_size=64, sampler=sampler) # shuffle=True) 
        validLoader = DataLoader(validSamp, batch_size=64, shuffle=True)
        foldTrues, foldPreds , _ = trainModel(sel_feat,trainLoader, valid=validLoader)
        trues = torch.vstack((trues, foldTrues))
        preds = torch.vstack((preds, foldPreds))

        print("fold metrics:")
        mcc=skm.matthews_corrcoef(foldTrues[:]>0.5,(foldPreds[:]>0.5))
        print(" mcc:",mcc)
        print(" r2_score:",r2_score(foldTrues,foldPreds))
        print(" pearson:",stats.pearsonr(foldTrues,foldPreds))
        print(' MAE:',mean_absolute_error(foldTrues,foldPreds))

        precision,recall,_ = skm.precision_recall_curve(foldTrues[:]>rcut,foldPreds[:])
        auprc=skm.auc(recall,precision)
        tpr_base=(foldTrues[:]>rcut).sum().item()/len(foldTrues)
        #axfoldprc.axhline(y=tpr_base,color='gray',alpha=0.5,linestyle='--',label='baseline')
        axfold[0].plot(recall,precision,linestyle='-',marker='o',lw=2,label='AUPRC=%.2f' % (auprc)) 
        fpr,tpr,t=skm.roc_curve(foldTrues[:]>rcut,foldPreds[:])
        auroc=skm.auc(fpr,tpr)
        axfold[1].plot(fpr,tpr,linestyle='-',marker='o',label='AUROC=%.2f' % (auroc))
    
    #print(trues) #.size())
    #print(preds.size())

    # settings for per fold plot
    axfold[0].legend()
    axfold[1].legend()
    axfold[0].set_xlabel('recall')
    axfold[0].set_ylabel('precision')
    axfold[1].plot([0, 1], [0, 1], linestyle='--',color='gray', alpha=0.5, label='Random (y = x)')
    axfold[1].set_xlabel('FPR')
    axfold[1].set_ylabel('TPR')


    ##### VALIDATION METRICS. 
    print("metrics from all folds:") 

    # max mcc
    mccThresh = [skm.matthews_corrcoef(trues[:]>thresh, (preds[:] > thresh).to(torch.float)) for thresh in np.arange(0,1,0.01)]
    maxMcc = max(mccThresh)
    print("max MCC:",maxMcc)

    # regular MCC
    mcc=skm.matthews_corrcoef(trues[:]>rcut,(preds[:]>rcut))
    print("regular MCC:",mcc)

    figauprc,axauprc=plt.subplots(1)            
    precision,recall,_ = skm.precision_recall_curve(trues[:]>rcut,preds[:])
    auprc=skm.auc(recall,precision)
    tpr_base=(trues[:]>rcut).sum().item()/len(trues)
    # plot prc 
    axauprc.plot(recall,precision,linestyle='-',marker='o',lw=2,label='%s AUPRC=%.2f' % (xlabels,auprc))
    axauprc.axhline(y=tpr_base,color='gray',alpha=0.5,linestyle='--',label='baseline')
    axauprc.set_xlabel('recall')
    axauprc.set_ylabel('precision')
    axauprc.legend()

    figauroc,axauroc=plt.subplots(1)        
    fpr,tpr,t=skm.roc_curve(trues[:]>rcut,preds[:])
    auroc=skm.auc(fpr,tpr)
    axauroc.plot(fpr,tpr,linestyle='-',marker='o',label='%s AUROC=%.2f' % (xlabels,auroc))
    axauroc.plot([0, 1], [0, 1], linestyle='--',color='gray', alpha=0.5, label='Random (y = x)')
    axauroc.set_xlabel('FPR')
    axauroc.set_ylabel('TPR')
    axauroc.legend()

    r2 = r2_score(trues,preds)
    print(f'R²: {r2}')
    mae = mean_absolute_error(trues,preds)
    print(f'MAE: {mae}')
    pcorr, pval =  stats.pearsonr(trues,preds) 

    # plot trues v preds
    figreg,axreg=plt.subplots(1)        
    axreg.scatter(trues,preds,label='pearson=%.2f pval=%.6f' % (pcorr,pval))
    axreg.set_ylabel('predictions')
    axreg.set_xlabel('true')
    #axreg.set_xlim([0,1])
    #axreg.set_ylim([0,1])
    axreg.plot([rcut,rcut], [0, 1], linestyle='--',color='gray', alpha=0.5)
    axreg.plot([0,1], [rcut, rcut], linestyle='--',color='gray', alpha=0.5)

    axreg.legend()
    #plt.show()

##############################
#def runTestSet(ifnModel,testDS,noval=0):
def predTestSet(ifname,sel_feat,testDS,noval=0):
    f0,f1=sel_feat[0],sel_feat[1]
    num_feat = f1-f0
    assert(num_feat>0)
    preds = torch.empty(0,1)
    trues = torch.empty(0,1)
    
    KCNQ1 = Network(num_feat)
    KCNQ1.load_state_dict(torch.load(ifname))  # saved model 
    KCNQ1.eval()
    with torch.no_grad():
        for data, labels in testDS:
            outputs = KCNQ1(data[f0:f1])
            preds = torch.vstack((preds,outputs))
            trues = torch.vstack((trues,labels))
            
   ##### VALIDATION METRICS. 
    print("metrics from all folds:") 
    xlabels=["penetrance"]

    # max mcc
    mccThresh = [skm.matthews_corrcoef(trues[:]>thresh, (preds[:] > thresh).to(torch.float)) for thresh in np.arange(0,1,0.01)]
    maxMcc = max(mccThresh)
    print("max MCC:",maxMcc)

    # regular MCC
    rcut=0.5 
    mcc=skm.matthews_corrcoef(trues[:]>rcut,(preds[:]>rcut))
    print("regular MCC:",mcc)

    figauprc,axauprc=plt.subplots(1)            
    precision,recall,_ = skm.precision_recall_curve(trues[:]>rcut,preds[:])
    auprc=skm.auc(recall,precision)
    tpr_base=(trues[:]>rcut).sum().item()/len(trues)
    # plot prc 
    axauprc.plot(recall,precision,linestyle='-',marker='o',lw=2,label='%s AUPRC=%.2f' % (xlabels,auprc))
    axauprc.axhline(y=tpr_base,color='gray',alpha=0.5,linestyle='--',label='baseline')
    axauprc.set_xlabel('recall')
    axauprc.set_ylabel('precision')
    axauprc.legend()

    figauroc,axauroc=plt.subplots(1)        
    fpr,tpr,t=skm.roc_curve(trues[:]>rcut,preds[:])
    auroc=skm.auc(fpr,tpr)
    axauroc.plot(fpr,tpr,linestyle='-',marker='o',label='%s AUROC=%.2f' % (xlabels,auroc))
    axauroc.plot([0, 1], [0, 1], linestyle='--',color='gray', alpha=0.5, label='Random (y = x)')
    axauroc.set_xlabel('FPR')
    axauroc.set_ylabel('TPR')
    axauroc.legend()

    r2 = r2_score(trues,preds)
    print(f'R²: {r2}')
    p=num_feat #12
    adj_r2 = 1 - (1-r2_score(trues,preds)) * (len(trues)-1)/(len(trues)-p-1)
    print(f'adjusted R²: {adj_r2}')
    mae = mean_absolute_error(trues,preds)
    print(f'MAE: {mae}')
    mse = mean_squared_error(trues,preds)
    print(f'MSE: {mse}')
    rmse = sqrt(mean_squared_error(trues,preds))
    print(f'RMSE: {rmse}')
    pcorr, pval =  stats.pearsonr(trues,preds) 


    # plot trues v preds
    figreg,axreg=plt.subplots(1)        
    axreg.scatter(trues,preds,edgecolors='k',label='pearson=%.2f pval=%.6f' % (pcorr,pval))
    axreg.set_ylabel('predictions')
    axreg.set_xlabel('true')
    #axreg.set_xlim([0,1])
    #axreg.set_ylim([0,1])
    axreg.plot([rcut,rcut], [0, 1], linestyle='--',color='gray', alpha=0.5)
    axreg.plot([0,1], [rcut, rcut], linestyle='--',color='gray', alpha=0.5)

    axreg.legend()
    #plt.show()

    ##########
    # variant-specific results 
    residues=testDS.variantNames[:]
    #for i,resi in enumerate(residues):
    #    #axreg.text(trues[i],preds[i],resi)
    #    resis.append(axreg.text(trues[i],preds[i],resi))
    #resis = [axreg.text(trues[i], preds[i], '%s' %residues[i], ha='center', va='center') for i in range(len(residues))]
    [axreg.text(trues[i], preds[i], '%s' %residues[i]) for i in range(len(residues))]
    #adjust_text(resis)

    return 

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

######################################################################
# These are for predicting penentrance . 
# A different class will be needed to readLabels of EP data. 

# start/end of column index for features to use: 
# 13=rmsf, 14=am, 
sel_feat=[0,12] # this will just use original 11+1 features 

#####
# biophys+AM+RMSF

# for lqt1_penetrance.csv, remove any "nan" entries , exclude variants with "tot_carriers=0" from training data . This is *clean*.csv file 
ds = PenetranceDataset("features_results_240909_order.csv","lqt1_penetrance_clean.csv") #,excludeWT=True)
crossValid(sel_feat,ds)
plt.show()
quit()
##
#### train on all data 
##sampler = WeightedRandomSampler(makeWeightVec(ds),len(ds))
###trainLoader = DataLoader(ds, batch_size=len(ds), sampler=sampler)
##trainLoader = DataLoader(ds, batch_size=64, sampler=sampler)
##modelTrues, modelPreds , _ = trainModel(sel_feat,trainLoader,valid=trainLoader,save=True) 
##
##plt.show()
##quit()

######
# train model on all data, test on unseen data


# test model on unseen data . 

# "order" is just the same csv file (in get_feat), but with the non perturbing-marked features in front
dt = PenetranceDataset("features_results_240909_order.csv","lqt1_penetrance_test_clean.csv") #,excludeWT=True)

# test model on new/unseen test set. 
# saved_ann_bp2pen_240916.pth # 64->32->32->1 w relu , 1k epochs, lr=1e-5
#predTestSet('temp.pth',sel_feat,dt)
predTestSet('saved_ann_bp2pen_240916.pth',sel_feat,dt)


plt.show()

quit()


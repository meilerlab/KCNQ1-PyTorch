# basic model from Eric, analyses from me. 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
        if excludeWT:
            self.variantNames = [name for name in self.featureDict.keys() if name[0] != name[-1]]

        else:
            self.variantNames = [name for name in self.featureDict.keys()]
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
            features = torch.tensor([float(x) for x in parts[1:13]])
            name = parts[0].strip()
            self.featureDict[name] = features
        f.close()
    
    def readLabels(self, labelcsv):
        f = open(labelcsv)
        for line in f:
            parts = line.split(",")
            name = parts[0].strip()
            #labels = torch.tensor([float(x) if x.strip() != "nan" else 0. for x in parts[1:]])
            raw = [float(x) if x.strip() != "nan" else 0. for x in parts[1:]]
            labels = torch.ones(4)
            if raw[0] < 0.17:
                labels[0] = 0.
                labels[1] = 0.
                labels[2] = 0.
                labels[3] = 0.
            if raw[0] < 0.55 or raw[0] > 1.15: #6:
                labels[0] = 0.
            if raw[1] > 1.30 or raw[1] < 0.80:
                labels[1] = 0.
            if raw[2] > 1.70 or raw[2] < 0.70:
                labels[2] = 0.
            if raw[3] < 0.75 or raw[3] > 1.25:
                labels[3] = 0.
            self.labelDict[name] = labels
        f.close()

### for logistic regression 
#class Network(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.linin = nn.Linear(12, 4) 
#        self.fa = nn.Sigmoid()
#    def forward(self, x):
#        x = self.linin(x)
#        x = self.fa(x)
#        return x

## for ANN 
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.linin = nn.Linear(12, 32) 
        self.layer1 = nn.Linear(32, 12)
        self.layer2 = nn.Linear(12, 4)
        self.relu = nn.LeakyReLU()
        self.fa = nn.Sigmoid()
        self.dropout1 = nn.Dropout(p=0.33)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.05)

    def forward(self, x):
        x = self.dropout3(x)
        x = self.linin(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.fa(x)
        return x

def trainModel(dl, valid=None):
    net = Network()
    #criterion = nn.MSELoss(reduction="mean")
    criterion = nn.BCELoss(reduction="sum")
    optimizer = optim.Adam(net.parameters())
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
    epoch = 0
    trainLoss = []
    validLoss = []
    endPreds = None
    endTrues = None
    while epoch < 1200:
        running_loss = 0.
        running_num = 0
        for data, labels in dl:
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
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
            with torch.no_grad():
                net.eval()
                for data, labels in valid:
                    outputs = net(data)
                    select = torch.sum(data[:,-4:],1) != 0.
                    #print(valid.variantNames)
                    #exit()
                    preds = torch.vstack((preds,outputs[select,:]))
                    trues = torch.vstack((trues,labels[select,:]))
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    valid_num += data.size(0)
                net.train()
                #print(trues.size(0))
                endTrues = trues
                endPreds = preds
                
            validLoss.append(valid_loss/valid_num)
            #print(epoch, trainLoss[-1], validLoss[-1])
        epoch += 1
    plt.plot([i for i in range(len(trainLoss))],trainLoss)
    plt.plot([i for i in range(len(validLoss))],validLoss)
    plt.savefig('loss.png')
    net.eval()
    return (endTrues, endPreds,net)

def makeWeightVec(ds):
    vec = []
    for data, label in ds:
        if torch.sum(data[-4:]) != 0.:
            vec.append(3.)
        else:
            vec.append(1.)
    return vec

def crossValid(ds, fold=5):
    splits = KFold(n_splits=fold, shuffle=True)
    trues = torch.empty(0,4)
    preds = torch.empty(0,4)
    for trainIdx, validIdx in splits.split(np.arange(len(ds))):
        #print(len(trainIdx))
        trainSamp = Subset(ds, trainIdx)
        validSamp = Subset(ds, validIdx)
        sampler = WeightedRandomSampler(makeWeightVec(trainSamp),len(trainSamp))
        trainLoader = DataLoader(trainSamp, batch_size=64, sampler=sampler)
        validLoader = DataLoader(validSamp, batch_size=64, shuffle=True)
        foldTrues, foldPreds , _ = trainModel(trainLoader, valid=validLoader)
        trues = torch.vstack((trues, foldTrues))
        preds = torch.vstack((preds, foldPreds))
    #print(trues.size())
    #print(preds)

    figauprc,axauprc=plt.subplots(1)    
    xlabels=["Iks","V1/2","tau_act","tau_deact"]    
    colors=['blue','orange','green','red']

    mccs , auprc = [],[]
    for i in range(4):
        mccThresh = [skm.matthews_corrcoef(trues[:,i], (preds[:,i] > thresh).to(torch.float)) for thresh in np.arange(0,1,0.01)]
        #print("thresh:",mccThresh)
        maxMcc = max(mccThresh)
        #print("mcc:",maxMcc, mccThresh.index(maxMcc))
        mccs.append(maxMcc)
        precision,recall,_ = skm.precision_recall_curve(trues[:,i],preds[:,i])
        auprc.append(skm.auc(recall,precision))
        
        # plot prc 
        axauprc.plot(recall,precision,linestyle='-',marker='o',label=xlabels[i],c=colors[i]) 
        # ,lw=2,label='%s AUPRC=%.2f' % (xlabels[i],auprc))

    print("max mcc:") 
    print(mccs)
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
    preds = torch.empty(0,4)
    trues = torch.empty(0,4)
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
    figauc.text(0.5,0.03,'False Positive Rate',ha='center')
    axauc.set_ylabel('True Positive Rate')
    #for i in range(4):
    axauc.plot([0,1],[0,1],color='k',linestyle='--',lw=0.5)
    axauc.legend(loc='lower right')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.title('ROC, test set')
    plt.savefig('auroc.png')

    ##########
    # for auprc    
    figauprc,axauprc=plt.subplots(1)
    ftp=sum(trues)/len(trues) # fraction true positives 
    for i in range(4):
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
    plt.savefig('auprc.png')
    #plt.show()

    ##########
    # variant-specific results 
    residues=testDS.variantNames[:]

    binaryPred = torch.zeros(len(preds), 4)
    for j in range(len(preds)): 
        for i in range(4): 
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

            # just plot wrong ones
            if (match[i][j].item()==0):
                wrong.append(residues[i])
                print(xlabels[j],"= pred=",func[i][j].item()," prob=",preds[i][j].item()," gt=",trues[i][j].item()," match?=",match[i][j].item())
            else: 
                right.append(residues[i])

    plt.show()
    plt.savefig('heatmap.png')
    return 

#####
def cfModels():
    # compare models . Each model architecture is ran 5x. AUPRC values are saved. Then we compare.
    # orig, orig w/ ADam, log reg w/ adam, and orig w/ Adam trained on orig+new, log reg trained on orig+new data     

    ##############################
    ## MCCs for KCNQ1 progress report 
    # train w/ orig data 
    origlog=[
        [0.4558676957679617, 0.6676322629936194, 0.5406835754490181, 0.6344935007484221],
        [0.4075309682072006, 0.6962165787240718, 0.5406835754490181, 0.6384389258971761],
        [0.41699804389425804, 0.6895828271676112, 0.4922778427988095, 0.546803062212687],
        [0.426423867216943, 0.6702578662976232, 0.5442455138715881, 0.5699074825848987],
        [0.4235119057737274, 0.6962165787240718, 0.5701323657106178, 0.5890333195090929]
    ]
    origAdam=[
        [0.488658025936577, 0.6760940618785495, 0.49976966004534357, 0.5959563949995046],
        [0.4030691395342715, 0.6627538877727291, 0.5200003504295492, 0.5918127348113934],
        [0.3847132649418082, 0.6133903233530341, 0.5082884899635816, 0.6276766897142139],
        [0.3572324287458255, 0.6074431545171679, 0.5560964123597755, 0.5416036221738872],
        [0.4489949259330556, 0.6901441116922944, 0.5326947387438172, 0.6468768290890939]        
    ]
    # train w/ orig+new data
    wnewlog=[
        [0.3741323553640879, 0.614860262582297, 0.49837376507254666, 0.5893492934112678],
        [0.3969717462618035, 0.6295380040802205, 0.5123955964618677, 0.5949929597650145],
        [0.4073515663460344, 0.6150857474650101, 0.4524307888959688, 0.5794043872625172],
        [0.400801226651248, 0.6406869223960758, 0.5007953872596048, 0.5906922207189379],
        [0.39364326414298934, 0.6589415298355806, 0.5084754435076584, 0.5893492934112678]
    ]
    wnewAdam=[
        [0.42677438197194567, 0.6200590370957261, 0.5114554193528571, 0.5636030579477402],
        [0.37319836334990797, 0.6006836910845077, 0.45450644447825583, 0.5055985284702472],
        [0.3799003893249917, 0.5955182198536311, 0.4775886953454905, 0.5508013688119326],
        [0.40794886818883336, 0.5299397152181071, 0.4850978485592254, 0.4786677782426137],
        [0.43717943350653343, 0.572325827136512, 0.4522003275565816, 0.49238133540623885]
    ]
    # ESM from David
    

    

    ###############################
    ## AUPRCs for kcnq1 vu call box plots
    #orig=[
    #    [0.5967926183510255, 0.8383872704320945, 0.7988692700762219, 0.7610476642944451],
    #    [0.522032087491464, 0.7744937344581649, 0.7417090609067483, 0.687312943690045],
    #    [0.5760296228112579, 0.7937738477525871, 0.746732109272032, 0.735445336277469],
    #    [0.49533419817079555, 0.799103382082438, 0.7499377693597433, 0.7028466356523104],
    #    [0.5490800149976789, 0.8135396822381783, 0.7697164508812303, 0.7353572916189892]
    #]
    #
    #origWAdam=[
    #    [0.5128769182539386, 0.8014650275468451, 0.7582864454216579, 0.7057644277550155],
    #    [0.6213141901826736, 0.8869754020469368, 0.7591144392593884, 0.7997909505359335],
    #    [0.6663666839646315, 0.8727975550663839, 0.7492837867408428, 0.7648031216008531],
    #    [0.5642893017178316, 0.7938537751134404, 0.7564238888046966, 0.731695865190494],
    #    [0.5488601677686036, 0.83296007250526, 0.7483287821507504, 0.7826034822941581]
    #]
    #
    #logReg=[
    #    [0.5480170656958282, 0.8656515531951275, 0.8065462257809627, 0.7720394963534014],
    #    [0.5877527382610159, 0.8773942946302519, 0.8301236226793712, 0.832351521246399],
    #    [0.5965341567510343, 0.9004658691643671, 0.7936328950705909, 0.8337455401294382],
    #    [0.5864310678429767, 0.8872798637028007, 0.8160268549833375, 0.8171088125045171],
    #    [0.6158860936654137, 0.8888848984002933, 0.7866996724463093, 0.8214223217829784]
    #]
    #
    ### train w/ orig+new data
    #origWAdam1=[
    #    [0.5914652491907011, 0.7877425315176573, 0.7507651186971631, 0.7713938942442096],
    #    [0.5651076908123247, 0.7820333932964706, 0.7387644092302833, 0.7397608211903126],
    #    [0.507176741974495, 0.7646816579671745, 0.7413718519159049, 0.7517048859819125],
    #    [0.5934933042223518, 0.843716759742529, 0.7220379874162615, 0.7883144219187298],
    #    [0.47821798439265545, 0.7034164824460575, 0.6678713049892242, 0.6700372669776764]
    #]
    #
    #logReg1=[
    #    [0.5680711722872944, 0.8371813980306881, 0.7621857862397283, 0.8240823333000366],
    #    [0.5463176146881048, 0.8130422842579992, 0.7440690047181892, 0.7988429268637687],
    #    [0.6051832886724453, 0.8339894285246167, 0.7546188691315883, 0.7835544459172471],
    #    [0.5687830003199076, 0.8150894064683553, 0.7550830569232556, 0.793520531939987],
    #    [0.6033169968962208, 0.8172640393582279, 0.7594423589323633, 0.793556255510582]
    #]
    #
    ###### add values from David's ESM model (from ppw he sent in Slack)
    #esm_ann=[
    #    [0.856,0.871,0.648,0.804]
    #]
    #esm_logreg=[
    #    [0.744,0.834,0.590,0.694]
    #]
    #esm_attnlayer=[
    #    [0.674,0.601,0.499,0.633]
    #]
    ######    
    #
    ## get avg values & plot
    #for i in range(4):
    #    origarr,origWAdamarr,logRegarr=[],[],[]
    #    origWAdam1arr,logReg1arr=[],[]
    #    for j in range(len(logReg)):
    #        origarr.append(orig[j][i])
    #        origWAdamarr.append(origWAdam[j][i])
    #        logRegarr.append(logReg[j][i])
    #        origWAdam1arr.append(origWAdam1[j][i])
    #        logReg1arr.append(logReg1[j][i])
    #    # esm
    #    esmannarr,esmlogregarr,esmattnarr=[],[],[]
    #    for j in range(len(esm_ann)): 
    #        esmannarr.append(esm_ann[j][i])
    #        esmlogregarr.append(esm_logreg[j][i])
    #        esmattnarr.append(esm_attnlayer[j][i])
    #        
    #    # box plot
    #    plt.figure()
    #    data=[origarr,origWAdamarr,logRegarr,origWAdam1arr,logReg1arr,esmannarr,esmlogregarr,esmattnarr]
    #    plt.boxplot(data)
    #    plt.savefig('boxplot'+str(i)+'.png')
    #
    #    ### get t-test p value 
    #    #print("t-test p-values:")
    #    #t_stat, p_value = stats.ttest_ind(origarr,origWAdamarr)
    #    #print(" train w/ orig, ANN+SGD -- ANN+Adam : ","{:.4f}".format(p_value))
    #    #t_stat, p_value = stats.ttest_ind(origWAdamarr,logRegarr)
    #    #print(" train w/ orig, ANN+Adam -- Log Reg : ","{:.4f}".format(p_value))
    #    #t_stat, p_value = stats.ttest_ind(origarr,logRegarr)
    #    #print(" train w/ orig, ANN+SGD -- Log Reg : ","{:.4f}".format(p_value))
    #    #t_stat, p_value = stats.ttest_ind(origWAdam1arr,logReg1arr)
    #    #print(" train w/ orig+new, ANN+Adam -- LogReg : ","{:.4f}".format(p_value))
    #    #t_stat, p_value = stats.ttest_ind(logRegarr,logReg1arr)
    #    #print(" Log Reg train w/ orig -- Log Reg train w/ orig+new: ","{:.4f}".format(p_value))
    #
    ## t-test

    #t_stat, p_value = stats.ttest_ind(origWAdamarr,logRegarr)
    #t_stat, p_value = stats.ttest_ind(origWAdam1arr,logReg1arr)

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

## train on original
ds = VariantDataset("../orig_data/full.whet.multimer.csv","../orig_data/a1q1.model_data.csv",excludeWT=True)

## train w/ original+new
#pwd='/dors/meilerlab/data/changga/kcnq1/ml_model/pytorch/240313/'
#ds = VariantDataset(pwd+"biophys_features_all.csv",pwd+"a1q1.model_data.csv")

trues,preds=crossValid(ds) 
quit()
#
#_,_,kcnq1=trainModel(ds)
#
#
#fsave="temp.pth" 
#torch.save(kcnq1,fsave)
#quit()
#
##cfModels()
##quit()

##############################
## test model trained on original data on new variants

#ifnModel="kcnq1_ann_model_sgd.pth" # architecture: ann as in paper 
#ifnModel="kcnq1_ann_model_adam.pth" # architecture: ann as in paper 
ifnModel="kcnq1_logreg_model.pth" # logistic regression 
#ifnModel="kcnq1_logreg_orig-new-data.pth"

pwd='/dors/meilerlab/data/changga/kcnq1/ml_model/pytorch/240313/'
#dtest = VariantDataset(pwd+"biophys_features_all_new_only.csv",pwd+"a1q1.model_data_new_vars_only.csv")
#dtest = VariantDataset(pwd+"biophys_features_all.csv",pwd+"a1q1.model_data.csv")
dtest = VariantDataset(pwd+"biophys_features_vus_from_al_only.csv",pwd+"a1q1.vus_from_al_only.csv") # noval=1
dtest = VariantDataset("biophys_features_try_all_aas_try_only.csv","a1q1.try_all_aas_try_only.csv")

runTestSet(ifnModel,dtest,1)


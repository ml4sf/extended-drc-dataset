import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn import utils

OF = open('ANN1.log', 'w')

def get_data(csv_file, training_size_frac=0.75):
    df = pd.read_csv(csv_file)
    
    #_cols = [ _ for _ in df.columns[2:-1]]
    _cols = ['E_LUMO_PM6[eV]', 'IsotropicAverageAlpha [A.U.]', ' AverageGamma [A.U.]', 'S1(CAS2-1+CIS) [eV]', 'INDOS-OscStr(CAS2-1+CIS)', 'Si', 'Mv', 'Mare', 'Mi', 'nHBAcc', 'nHBDon', 'n6Ring', 'n8HeteroRing', 'nF9HeteroRing', 'nT5HeteroRing', 'nT6HeteroRing', 'nT7HeteroRing', 'nT8HeteroRing', 'nT9HeteroRing', 'nT10HeteroRing', 'nRotB', 'RotBFrac', 'nRotBt', 'RotBtFrac']

   # _cols = [
   #         "DipoleMoment [D]",
   #         "IsotropicAverageAlpha [A.U.]",
   #         "S1(CAS2-1+CIS) [eV]",
   #         "S2(CAS2-1+CIS) [eV]",
   #         "CI-Coef**2(CAS2-1+CIS)",
   #         "T1(CAS2-1+CIS) [eV]",
   #         "T2(CAS2-1+CIS) [eV]",
   #         "nAcid",
   #         "naAromAtom",
   #         "nHeavyAtom",
   #         "nS",
   #         "nF",
   #         "nBondsD",
   #         "nBondsD2",
   #         "nBondsT",
   #         "C2SP1",
   #         "Sv",
   #         "Sp",
   #         "Mv",
   #         "Mpe",
   #         "Mp",
   #         "Mi",
   #         "nHBAcc_Lipinski",
   #         "nHBDon",
   #         "nAtomP",
   #         "nAtomLAC",
   #         "n3Ring",
   #         "n4Ring",
   #         "n6Ring",
   #         "n7Ring",
   #         "n8Ring",
   #         "n10Ring",
   #         "nF6Ring",
   #         "n3HeteroRing",
   #         "nF8HeteroRing",
   #         "nF9HeteroRing",
   #         "nT4HeteroRing",
   #         "nT9HeteroRing",
   #         "RotBFrac",
   #         "nRotBt"
   #     ]
 
    print(_cols)
    X = df[_cols].to_numpy()
    _Y = np.array(df[df.columns[-1]])
    #Y = np.array([1 if _>.05 and _<.5 else 0 for _ in _Y])
    Y = np.array([1 if _>.05 else 0 for _ in _Y])
    X, Y = utils.shuffle(X,Y)
    OF.write('Y={}\n'.format(Y))
    print(Y)
    training_set_len = int(training_size_frac*X.shape[0])

    X_train = X[:training_set_len,:]
    Y_train = Y[:training_set_len]
    
    X_eval = X[training_set_len:,:]
    Y_eval = Y[training_set_len:]
    
    OF.write('Sum(Y_train) = {}\n'.format(sum(Y_train)))
    OF.write('Sum(Y_train) = {}\n'.format(sum(Y_train)))
    print('Sum(Y_train) = {}'.format(sum(Y_train)))
    print('Sum(Y_eval) = {}'.format(sum(Y_eval)))

    X_train, Y_train, X_eval, Y_eval = map(torch.tensor, (X_train, Y_train, X_eval, Y_eval))
    #return X_train.float(), Y_train.long(), X_eval.float(), Y_eval.long()
    return X_train.float(), Y_train.float(), X_eval.float(), Y_eval.float()

class dataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]  

    def __len__(self):
        return self.length

class Net(nn.Module):
    def __init__(self,input_shape):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,40)
        self.fc2 = nn.Linear(40,20)
        self.fc3 = nn.Linear(20,1)  
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data(sys.argv[1])

    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    L2_REGULARIZATION = 0.025
    EPOCHS = 1000
    
    x = X_train
    y = y_train
    
    sc = StandardScaler()
    x = sc.fit_transform(x)
    X_test = sc.fit_transform(X_test)
    

    trainset = dataset(x,y)#DataLoader
    trainloader = DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=False)

    model = Net(input_shape=x.shape[1])
    optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
    loss_fn = nn.BCELoss()

    #forward loop
    losses = []
    accur = []
    OF.write('X shape = {}\n'.format(x.shape))
    print(x.shape)
    for i in range(EPOCHS):
        for j,(x_train,y_train) in enumerate(trainloader):

            #calculate output
            output = model(x_train)

            #calculate loss
            loss = loss_fn(output,y_train.reshape(-1,1))

            #accuracy
            predicted = model(torch.tensor(X_test, dtype=torch.float32))
            acc = (predicted.reshape(-1).detach().round() == y_test.round()).float().mean()
            '''
            sum_zeros = 0
            sum_ones = 0
            _prd = predicted.reshape(-1).detach().round()
            for k in range(y_test.shape[0]):
                if y_test.round()[k] == 0 and _prd[k] == 0:
                    sum_zeros += 1
                elif y_test.round()[k] == 1 and _prd[k] == 1:
                    sum_ones += 1

            print('Correct Zeros:{}; Correct Ones:{}'.format(sum_zeros/((y_test.round() == 0).sum().float()),
                                    sum_ones/((y_test.round() == 1).sum().float())))
            '''
            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.append(loss)
        accur.append(acc)
        #if i%50 == 0:
        OF.write("epoch {}\tloss : {}\t accuracy : {}\n".format(i,loss,acc))
        print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))
        
        sum_zeros = 0
        sum_ones = 0
        _prd = predicted.reshape(-1).detach().round()
        for k in range(y_test.shape[0]):
            if y_test.round()[k] == 0 and _prd[k] == 0:
                sum_zeros += 1
            elif y_test.round()[k] == 1 and _prd[k] == 1:
                sum_ones += 1

        print('-----Num Corr 0: {}; Num Corr 1: {}'.format(sum_zeros, sum_ones))
        print('Correct Zeros:{}; Correct Ones:{}'.format(sum_zeros/((y_test.round() == 0).sum().float()),
                                    sum_ones/((y_test.round() == 1).sum().float())))
    #plt.plot(accur, label = 'Accuracy over test set')
    plt.plot(accur, label = 'Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy over the test set')
    #plt.ylim(.5, 1)
    #plt.legend(loc='center right')
    plt.savefig('TestCurve', dpi=300)
    plt.show()
    OF.close()

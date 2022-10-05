import torch
class Dataset1dto1d(torch.utils.data.Dataset):
    def __init__(self, X,y,nclasses=5,shift=2):
        super(Dataset1dto1d, self).__init__()
        self.X = X
        self.y = y
        self.l = len(self.X[0])
        self.nclasses = nclasses
        self.shift=shift

        self.preprocess()

    def preprocess(self):
        self.X = [torch.tensor(self.X[i]).reshape(1,self.l).float() for i in range(len(self.X))]
        #self.y = [torch.swapaxes(Fun.one_hot(torch.tensor(y[i])+self.shift,
        #                                  self.nclasses),1,0) for i in range(len(y))]
        self.y = [torch.tensor(self.y[i])+self.shift for i in range(len(self.y))]


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx],self.y[idx]

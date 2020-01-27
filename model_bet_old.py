import torch.nn as nn
import torch.nn.functional as F
from layer import GNN_Layer
from layer import GNN_Layer_Init
import torch 


class GNN_Bet(nn.Module):
    def __init__(self, ninput, nhid, dropout):
        super(GNN_Bet, self).__init__()

        self.gc1 = GNN_Layer_Init(ninput,nhid)
        self.gc2 = GNN_Layer(nhid,nhid)
        self.gc3 = GNN_Layer(nhid,nhid)
        self.gc4 = GNN_Layer(nhid,nhid)
        self.gc5 = GNN_Layer(nhid,nhid)
        #self.gc6 = GNN_Layer(nhid,nhid)

        self.dropout = dropout

        self.linear1 = nn.Linear(nhid,2*nhid)
        self.linear2 = nn.Linear(2*nhid,2*nhid)
        self.linear3 = nn.Linear(2*nhid,1)



    def forward(self,adj1,adj2):

        #Layers for aggregation operation
        x_1 = F.normalize(F.relu(self.gc1(adj1)),p=2,dim=1)
        x2_1 = F.normalize(F.relu(self.gc1(adj2)),p=2,dim=1)


        x_2 = F.normalize(F.relu(self.gc2(x_1, adj1)),p=2,dim=1)
        x2_2 = F.normalize(F.relu(self.gc2(x2_1, adj2)),p=2,dim=1)


        x_3 = F.normalize(F.relu(self.gc3(x_2,adj1)),p=2,dim=1)
        x2_3 = F.normalize(F.relu(self.gc3(x2_2,adj2)),p=2,dim=1)
        #x_3 = F.relu(self.gc3(x_2,adj1))
        #x2_3 = F.relu(self.gc3(x2_2,adj2))
        
        x_4 = F.normalize(F.relu(self.gc4(x_3,adj1)),p=2, dim=1)
        x2_4 = F.normalize(F.relu(self.gc4(x2_3,adj2)),p=2,dim=1)

        x_5 = F.relu(self.gc5(x_4,adj1))
        x2_5 = F.relu(self.gc4(x2_4,adj2))

        #Score Calculations
        #to-do: make a MLP layer and import here
        score1_1 = F.relu(self.linear1(x_1))
        score1_1 = F.dropout(score1_1,self.dropout)
        score1_1 = F.relu(self.linear2(score1_1))
        score1_1 = F.dropout(score1_1,self.dropout)
        score1_1 = self.linear3(score1_1)

        
        score1_2 = F.relu(self.linear1(x_2))
        score1_2 = F.dropout(score1_2,self.dropout)
        score1_2 = F.relu(self.linear2(score1_2))
        score1_2 = F.dropout(score1_2,self.dropout)
        score1_2 = self.linear3(score1_2)
        
        score1_3 = F.relu(self.linear1(x_3))
        score1_3 = F.dropout(score1_3,self.dropout)
        score1_3 = F.relu(self.linear2(score1_3))
        score1_3 = F.dropout(score1_3,self.dropout)
        score1_3 = self.linear3(score1_3)
        
        
        score1_4 = F.relu(self.linear1(x_4))
        score1_4 = F.dropout(score1_4,self.dropout)
        score1_4 = F.relu(self.linear2(score1_4))
        score1_4 = F.dropout(score1_4,self.dropout)
        score1_4 = self.linear3(score1_4)

        score1_5 = F.relu(self.linear1(x_5))
        score1_5 = F.dropout(score1_5,self.dropout)
        score1_5 = F.relu(self.linear2(score1_5))
        score1_5 = F.dropout(score1_5,self.dropout)
        score1_5 = self.linear3(score1_5)
        


        score2_1 = F.relu(self.linear1(x2_1))
        score2_1 = F.dropout(score2_1,self.dropout)
        score2_1 = F.relu(self.linear2(score2_1))
        score2_1 = F.dropout(score2_1,self.dropout)
        score2_1 = self.linear3(score2_1)

        
        score2_2 = F.relu(self.linear1(x2_2))
        score2_2 = F.dropout(score2_2,self.dropout)
        score2_2 = F.relu(self.linear2(score2_2))
        score2_2 = F.dropout(score2_2,self.dropout)
        score2_2 = self.linear3(score2_2)

        score2_3 = F.relu(self.linear1(x2_3))
        score2_3 = F.dropout(score2_3,self.dropout)
        score2_3 = F.relu(self.linear2(score2_3))
        score2_3 = F.dropout(score2_3,self.dropout)
        score2_3 = self.linear3(score2_3)
        
        
        score2_4 = F.relu(self.linear1(x2_4))
        score2_4 = F.dropout(score2_4,self.dropout)
        score2_4 = F.relu(self.linear2(score2_4))
        score2_4 = F.dropout(score2_4,self.dropout)
        score2_4 = self.linear3(score2_4)
        
        score2_5 = F.relu(self.linear1(x2_5))
        score2_5 = F.dropout(score2_5,self.dropout)
        score2_5 = F.relu(self.linear2(score2_5))
        score2_5 = F.dropout(score2_5,self.dropout)
        score2_5 = self.linear3(score2_5)
        
        
        score1 = score1_1 + score1_2 + score1_3 + score1_4 + score1_5
        score2 = score2_1 + score2_2 + score2_3 + score2_4 + score2_5

        #score1 = score1_1 + score1_2 + score1_3 
        #score2 = score2_1 + score2_2 + score2_3 

        x = torch.mul(score1,score2)

        return x

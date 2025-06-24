import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader

class TwoTower(nn.Module) :
    def __init__(self , user_input_dim , item_input_dim , embedding_dim = 128):
        super().__init__() 
        
        self.user_network = nn.Sequential(
            nn.Embedding(user_input_dim , embedding_dim) , 
            nn.Linear(embedding_dim , 256) , 
            nn.ReLU() , 
            nn.Linear(256,embedding_dim)
        ) 

        self.item_network = nn.Sequential(
            nn.Embedding(item_input_dim , embedding_dim) , 
            nn.Linear(embedding_dim , 256) , 
            nn.ReLU() , 
            nn.Linear(256,embedding_dim)
        ) 

        
    def forward(self , user , pos_item , neg_item) :
        user_output = self.user_network(user) 
        pos_item_output = self.item_network(pos_item)
        neg_item_output = self.item_network(neg_item)


        pos_result = F.cosine_similarity(user_output , pos_item_output)
        neg_result = F.cosine_similarity(user_output , neg_item_output) 

        return pos_result , neg_result 

class PairWiseLossFunc(nn.Module):
    def __init__(self , a):
        super().__init__()
        self.a = a 

    def forward(self , pos_result , neg_result):
        loss = torch.log(1 + torch.exp(self.a * (neg_result - pos_result)))
        return loss .mean()

class MyModel :
    def __init__(self , user_input_dim, item_input_dim, lr=0.01, epoches=1000):
        self.a = 1 
        self.lr = lr
        self.epoches = epoches 
        self.model = TwoTower(user_input_dim , item_input_dim)
        self.optimer = torch.optim.Adam(params = self.model.parameters() , lr = self.lr)
        self.loss_fn = PairWiseLossFunc(self.a) 

    def train(self , dataloader) :
        for epoch in range(self.epoches) :
            for x , pos_y , neg_y in dataloader :
                pos_result , neg_result = self.model(x , pos_y , neg_y) 
                loss = self.loss_fn(pos_result , neg_result)

                self.optimer.zero_grad()
                loss.backward()
                self.optimer.step()

                
            if epoch % 10 == 0 :
                print(f"-----第{epoch}轮 loss等于{loss}-----")

class RecommendationDataset(Dataset):
    def __init__(self, num_users, num_items, num_samples):
        self.num_users = num_users
        self.num_items = num_items
        self.num_samples = num_samples

        self.users = torch.randint(0, num_users, (num_samples,))
        self.pos_items = torch.randint(0, num_items, (num_samples,))
        self.neg_items = torch.randint(0, num_items, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.users[idx], self.pos_items[idx], self.neg_items[idx]
    


num_users = 1000     
num_items = 5000      
num_samples = 10000     
batch_size = 64         


dataset = RecommendationDataset(num_users, num_items, num_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main() :
    mymodel = MyModel(num_users , num_items) 
    mymodel.train(dataloader)
    
main()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import mean_squared_error
from torchvision import transforms

from DTIDataset import DTADataset
from metrics import mse, rmse, ci, rm2
from model import net
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
torch.cuda.set_device(3)
# Define training parameters
batch_size = 4
learning_rate = 0.001
num_epochs = 1000
dropout=0.2
# Create instances of the dataset and dataloader
dataset_train=DTADataset('Davis',r'/mnt/sdb/home/hjy/Summary-DTA/data/Davis/processed/train/fold1/idx_id.csv',r'/mnt/sdb/home/hjy/Summary-DTA/data/Davis/label.json')
dataset_test=DTADataset('Davis',r'/mnt/sdb/home/hjy/Summary-DTA/data/Davis/processed/test/fold1/idx_id.csv',r'/mnt/sdb/home/hjy/Summary-DTA/data/Davis/label.json')
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=dataset_train.collate_fn,shuffle=True, drop_last=True)
dataloader_test=DataLoader(dataset_test, batch_size=batch_size, collate_fn=dataset_train.collate_fn,shuffle=True, drop_last=True)
print('data load end')
# Create an instance of the SDTAnet model
model = net.SDTNET_NEW(protein_len=1200, compound_len=85,
                protein_emb_dim=100, compound_emb_dim=64,compound_fea_dim=44,protein_fea_dim=41, out_dim=1,dropout=0.4,model_name='Graph Transformer')
#model = torch.load( r'/mnt/sdb/home/hjy/Summary-DTA/model/save/Graph_Transforme_model2.pth')
model.to(device)
print('Graph Transformer')
# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = Adam(model.parameters(), lr=learning_rate)
best_mse=1000
# Training loop

for epoch in range(num_epochs):
    print(epoch)
    total_loss = 0
    for batch in dataloader_train:
        # Transfer batch data to device
        batch = [item.to(device) for item in batch]

        # Forward pass
        compound_onehot, compound_graph, compound_img,compound_3d,protein_onehot, protein_word2vec, protein_graph,protein_3d,labels = batch
        #compound_onehot = compound_onehot.to(torch.long)
        #protein_onehot = protein_onehot.to(torch.long)
        #compound_adj=compound_adj.to(torch.float64)
        #print(compound_img.shape)
        #print(labels)
        outputs,l = model(compound_onehot, protein_word2vec,compound_graph,protein_graph,compound_img,compound_3d,protein_3d)

        # Compute the loss
        loss = loss_fn(outputs, labels)+l

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print the average loss for the epoch
    avg_loss = total_loss / len(dataloader_train)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for batch in dataloader_test:
            batch = [item.to(device) for item in batch]
            compound_onehot, compound_graph, compound_img,compound_3d,protein_onehot, protein_word2vec, protein_graph, protein_3d,labels = batch
            output,l=model(compound_onehot, protein_word2vec,compound_graph,protein_graph,compound_img,compound_3d,protein_3d)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, labels.view(-1, 1).cpu()), 0)

        total_labels = total_labels.numpy().flatten()
        total_preds = total_preds.numpy().flatten()

        MSE = mse(total_labels, total_preds)
        RMSE = rmse(total_labels, total_preds)
        CI = ci(total_labels, total_preds)
        RM2 = rm2(total_labels, total_preds)
        print('MSE: ',MSE)
        print('RMSE: ',RMSE)
        print('CI: ',CI)
        print('RM2: ',RM2)
        st='EPOCH: '+str(epoch)+'  MSE: '+str(MSE)+' RMSE: '+str(RMSE)+' CI: '+str(CI)+' RM2: '+str(RM2)
        with open(r"/mnt/sdb/home/hjy/Summary-DTA/train_log/Graph_Transforme.txt", "a") as file:
            file.write(st)
        if MSE < best_mse:
            # Save the trained model
            torch.save(model, r'/mnt/sdb/home/hjy/Summary-DTA/model/save/Graph_Transforme_model2.pth')
            best_mse=MSE
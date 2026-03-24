import numpy as np 
import pandas as pd 
import geopandas as gpd
import utils.cross_validation as cval
import torch
from torch import nn
import matplotlib as plt
# from captum.attr import IntegratedGradients

import yaml


#Neural network
class MLP(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential( #Passes through layers sequentially
             nn.Linear(17, 30, bias= True), #100 inputs into 200 perceptrons
             nn.ReLU(), #Activation function
             nn.Linear(30, 100, bias= True), # 200 inputs 100 outputs
             nn.ReLU(),
             nn.Linear(100, 2, bias= True), # 200 inputs 100 outputs
             #nn.Dropout(0.05)
         )

    def forward(self, x): #Feed input X into layers mentioned above
      return self.layers(x)
    
#The dataloader, tensor conversions etc
class Dataset(torch.utils.data.Dataset): #Initialsing a structure to pass each line of data through

    def __init__(self, X, y, scale_data=True):
      if not torch.is_tensor(X) and not torch.is_tensor(y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

if __name__ == "__main__":

    with open("parameters.yaml", "r") as f:
        config = yaml.safe_load(f)

    N_EPOCHS = config['training']['epochs']
    TEST_SIZE = config['data']['test_size']
    BATCH_SIZE = config['training']['batch_size']
    # DATA_PATH = config['data']['path']
    EXPLAINABLE = config['integrated']

    df= pd.read_csv("data/final/fd_df.csv")
    PID_loc= pd.read_csv("data/lookup/PID_location_all.csv")

    ecoregions=cval.process_ecoregion("data/Ecoregions/Ecoregions2017.shp")

    df=df.merge(PID_loc, on="PID", how="left")
    df.dropna(subset=["lat", "lon"], inplace=True)

    df = df.drop(columns=[col for col in df.columns if "_y" in col])
    df.columns = df.columns.str.replace('_x$', '', regex=True)

    df=cval.assign_spatial_groups(df, grid_size=1.0)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"  # WGS84
    )

    train, test= cval.ecoregion_cross_validation(gdf, ecoregions, TEST_SIZE, BATCH_SIZE)

    train = train.loc[:, :'csc']
    test = test.loc[:, :'csc']

    y=['transformed npp', 'csc'] 

    X_train=train.drop(columns=y+['PID']).values
    y_train = np.column_stack([train['transformed npp'], train['csc']])

    X_test=test.drop(columns=y+['PID']).values
    y_test = np.column_stack([test['transformed npp'], test['csc']])

    dataset = Dataset(X_train, y_train)
    testset= Dataset(X_test, y_test)

    #Saving results
    results = {} 
    vloss=[]
    tloss=[]
    testloss=[]
    rv=[]
    rt=[]
    lrs = []

    #Initialize the NN
    model = MLP()
    criterion = nn.L1Loss() #Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001) #Gradient descent
    # lambda1 = lambda epoch: 0.65 ** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    trainloader = torch.utils.data.DataLoader( dataset,  batch_size=BATCH_SIZE, num_workers=8)
    valloader = torch.utils.data.DataLoader( dataset, batch_size=BATCH_SIZE, num_workers=8)
    testloader = torch.utils.data.DataLoader( testset, batch_size=BATCH_SIZE, num_workers=8)

     
    for epoch in range(N_EPOCHS): 
        train_loss = 0.0
        valid_loss = 0.0

        #train the model -------------------------------
        model.train() # prep model for training

        for data, target in trainloader:
            data=data.float()
            target=target.float()
            # target = target.reshape((target.shape[0], 1))

            output= model(data.float()) #Feedforward
            optimizer.zero_grad()
            loss = criterion(output, target)

            loss.backward() #backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step() #perform a single optimization step (parameter update)
            lrs.append(optimizer.param_groups[0]["lr"])
            # scheduler.step()
            train_loss += loss.item()
        print('finish training')
        #validate the model
        model.eval()
        for data, target in valloader:
            with torch.no_grad():
                data, target = data.float(), target.float()
                output = model(data)  # forward pass: compute predicted outputs by passing inputs to the model
                loss1 = criterion(output, target) # calculate the loss
                valid_loss += loss1.item()

                train_loss = train_loss/len(trainloader)
                tloss.append(train_loss)
                valid_loss = valid_loss/len(valloader)
                vloss.append(valid_loss)
                print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format( epoch+1, train_loss, valid_loss))
        
    
    torch.save(model.state_dict(), 'model.pt')

    test_loss = 0.0

    model.eval() # prep model for evaluation
    for data, target in testloader:
            data, target = data.float(), target.float()
            target = target.reshape((target.shape[0], 1))
            #forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            #calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()

        #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

        # print training/validation statistics
        # calculate average loss over an epoch
        
    test_loss = test_loss/len(testloader)
    for i in range(N_EPOCHS):
        testloss.append(test_loss)

    print('the loss on the Test data is', test_loss)

    length= range(0, len(vloss))
    leng= range(0, len(tloss))
    tleng=range(0, len(testloss))

    plt.plot(length, vloss,  color="blue", label= "Validation loss")
    plt.plot(leng, tloss, color='red', label= 'Trainingloss')
    plt.plot(tleng, testloss, color='green', label= 'Test loss')
    plt.title( "Loss function over time")
    plt.legend()
    plt.savefig('training.png')

    plt.plot(N_EPOCHS,lrs, color='black', label= 'learning rate')
    plt.savefig('idk,png')

    # if EXPLAINABLE:
    #     model.eval()
    #     input = torch.rand(1195, 1122)
    #     baseline = torch.zeros(1195, 1122)

    #     ig = IntegratedGradients(model)
    #     attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
    #     attr = attributions.detach().numpy()
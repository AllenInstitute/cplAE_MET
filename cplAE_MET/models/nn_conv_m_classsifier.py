import numpy as np
import pandas as pd
import argparse
import csv
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter

from cplAE_MET.utils.dataset import MET_exc_inh
from cplAE_MET.utils.utils import set_paths
from cplAE_MET.models.augmentations import get_padded_im, get_soma_aligned_im

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--output_filename',           default='50k_classification_results_test.csv',  type=str,   help='config file with data paths')

def main(output_filename="50k_classification_results_test.csv"):
   
    # classifier class ##########################################################

    # def make_weights_for_balanced_classes(dataobj, n_met, met_subclass_id , batch_size):
    #     train_classes = dataobj.group.copy()
    #     train_subclasses = dataobj.subgroup.copy()

    #     # desired prob of choosing a met cell from each class
    #     met_prob = n_met/(batch_size/2)
    #     others_prob = 1 - met_prob
          
    #     # class and subclass relationship: dic={0:0, 1:1, 2:0, 3:1}
    #     dic = {}
    #     for i, j in zip(train_classes, train_subclasses):
    #         dic[j] = i

    #     # count of class and subclasses and the total number of cells  
    #     class_counts = Counter(train_classes)
    #     subclass_counts = Counter(train_subclasses)
    #     N = float(sum(class_counts.values()))
        
    #     # to each cell, we assign a weight based on their classes. so if we use these weights then we can sample 50% from each class
    #     weight_per_class = {}  
    #     for k, v in class_counts.items():    
    #         weight_per_class[k] = N/float(v)

    #     # renormalize in such a way so that we still sample 500 from each class but the expected value for met cell sampling is 54 
    #     # in each class 
    #     weight_per_subclass = {}   
    #     for k, v in subclass_counts.items():
    #         c = dic[k]
    #         if k in met_subclass_id:
    #             weight_per_subclass[k] = (weight_per_class[c] * met_prob/v) * class_counts[c]
    #         else: 
    #             weight_per_subclass[k] = (weight_per_class[c] * others_prob/v) * class_counts[c]

    #     weights = [0] * len(train_subclasses)
    #     for idx, val in enumerate(train_subclasses):                                          
    #         weights[idx] = weight_per_subclass[val]

    #     return weights
    
    def make_weights_for_balanced_classes(images, nclasses):                        
      count = [0] * nclasses                                                      
      for item in images: 
          count[item] += 1                                                     
      weight_per_class = [0.] * nclasses                                      
      N = float(sum(count))                                                   
      for i in range(nclasses):                                                   
          weight_per_class[i] = N/float(count[i])                                 
      weight = [0] * len(images)                                              
      for idx, val in enumerate(images):                                          
          weight[idx] = weight_per_class[val]                                  
      return weight


    class conv_classifier(nn.Module):
        """ conv3d classifier which takes the images and return .
        - `xm` expected in [N, C=1, D=240, H=4, W=4] format, with C = 1, D=240, H=4, W=4
        - Elements of xm expected in range (0, ~40). Missing data is encoded as nans.
        - Output is an intermediate representation, `zm_int`
        """

        def __init__(self, out_dim=1):
            super().__init__()
            # self.conv_0 = nn.Conv3d(1, 4, kernel_size=(7, 3, 1), padding=(3, 1, 0))
            # self.pool_0 = nn.MaxPool3d((4, 1, 1), return_indices=True)
            # self.conv_1 = nn.Conv3d(10, 10, kernel_size=(7, 3, 1), padding=(3, 1, 0))
            # self.pool_1 = nn.MaxPool3d((4, 1, 1), return_indices=True)
            self.drp = nn.Dropout(p=0.7)
            self.fc_0 = nn.Linear(1920, 10)
            self.fc_1 = nn.Linear(10, out_dim)
            self.bn = nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
            self.relu = nn.ReLU()
            self.Sigmoid = nn.Sigmoid()
            return

        def forward(self, x):
            # x, self.pool_0_ind = self.pool_0(self.relu(self.conv_0(x)))
            # x, self.pool_1_ind = self.pool_1(self.relu(self.conv_1(x)))
            # x = x.view(x.shape[0], -1)
            x = self.drp(x)
            x = x.reshape(x.shape[0], -1)
            x = self.relu(self.fc_0(x))
            output = self.Sigmoid(self.bn(self.fc_1(x)))
            return output


    #defining dataset class
    class dataset(Dataset):
      def __init__(self,x,y, device=None):
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        self.x = torch.tensor(x,dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y,dtype=torch.float32, device=self.device)
        self.length = self.x.shape[0]
    
      def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
      def __len__(self):
        return self.length
        

    def compute_BCE_loss(x, target):
        bcel = nn.BCELoss()
        return bcel(x, target)

    def optimizer_to(optim, device):
        '''function to send the optimizer to device, this is required only when loading an optimizer 
        from a previous model'''
        for param in optim.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)
        


    # Main code ###############################################################
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    config_file = 'config.toml'
    exp_name="nn_conv_m_classifier"
    fold_n=10
    epochs=10

        
    # Read input data ---------------------------------------------------------
    dir_pth = set_paths(config_file, exp_name=exp_name, fold_n=fold_n)
    tb_writer = SummaryWriter(log_dir="/home/fahimehb/Local/new_codes/cplAE_MET/data/results/tb_logs/test/")
    dat, D = MET_exc_inh.from_file(dir_pth['MET_data'])

    # Preprocess M data for convolutiuon --------------------------------------
    dat.XM = np.expand_dims(dat.XM, axis=1)
    dat.Xsd = np.expand_dims(dat.Xsd, axis=1)
    # pad = 60
    # norm2pixel_factor = 100
    # padded_soma_coord = np.squeeze(dat.Xsd * norm2pixel_factor + pad)
    # dat.XM = get_padded_im(im=dat.XM, pad=pad)
    # dat.XM = get_soma_aligned_im(im=dat.XM, soma_H=padded_soma_coord)

    # Choose only exc cells with m data and remove the rest -------------------
    is_m_1d = np.all(~np.isnan(dat.XM), axis=(1,2,3,4))
    is_exc_1d = dat.class_id==0
    is_m_exc_1d = np.logical_and(is_m_1d, is_exc_1d)
    new_dat = dat[is_m_exc_1d, :]
    x = new_dat.XM
    y = new_dat.platform
    print("shape of x: {}\nshape of y: {}".format(x.shape,y.shape))

    # Loop over em cells to predict their labels ------------------------------
    all_em_cells_ind = np.where(y)[0] 
    predicted_em_cells = np.array([])
    predicted_labels_em_cells = np.array([])
    true_labels_em_cells = np.array([])

    ## so we predict 100 em cells labels at a time, train on the remaing patch and em cells
    em_batch=20000
    for i in range(0, len(all_em_cells_ind), em_batch):
      if len(all_em_cells_ind)-i < em_batch:
        j = len(all_em_cells_ind)-i
      else:
        j=em_batch
      print(i, i+j)
      em_cells_test_ind = all_em_cells_ind[i:i+j]
      test_cells_mask = np.array([False] * len(y))
      test_cells_mask[em_cells_test_ind] = True
      train_cells_mask = ~test_cells_mask
      print(Counter(y[train_cells_mask]))

      # Model , Optimizer, Loss -------------------------------------------------
      model = conv_classifier()
      optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
      loss_fn = nn.BCELoss()

      model.to(device)
      optimizer_to(optimizer,device)

      # Weighted sampling strategy -----------
      # weights = make_weights_for_balanced_classes(new_dat[np.where(train_cells_mask), :], n_met = 54, met_subclass_id = [2, 3], batch_size=500)             
      weights = make_weights_for_balanced_classes(new_dat[np.where(train_cells_mask), :].platform, nclasses=2)                                                                                                             
      weights = torch.DoubleTensor(weights)                                       
      sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) 

      #DataLoader
      trainset = dataset(x[train_cells_mask], y[train_cells_mask], device=device)
      trainloader = DataLoader(trainset, batch_size=500, shuffle=False, sampler=sampler)

      testset = dataset(x[test_cells_mask], y[test_cells_mask], device=device)
      testloader = DataLoader(testset, batch_size=x[test_cells_mask].shape[0], shuffle=False)


      #forward loop
      losses = []
      accur = []
      model.train()
      for i in range(epochs):
        train_loss=[]
        accuracy=[]
        for _,(x_train, y_train) in enumerate(trainloader):
          
          #calculate output
          output = model(x_train)
      
          #calculate loss
          loss = loss_fn(output,y_train.reshape(-1,1))
      
          #accuracy
          predicted = model(torch.tensor(x,dtype=torch.float32,device=device))
          acc = (predicted.reshape(-1).detach().cpu().numpy().round() == y).mean()
          #backprop
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          train_loss.append(loss)
          accuracy.append(acc)
        
        train_loss = torch.stack(train_loss).detach().cpu().numpy().mean()
        accuracy = np.mean(accuracy)
        

        print("epoch {}\tloss: {}\t accuracy: {} \t training_classes: {}".format(i,train_loss, accuracy, Counter(y_train.cpu().numpy())))
        tb_writer.add_scalar('Train/loss', train_loss, i)
        tb_writer.add_scalar('Train/accuracy', accuracy, i)

        model.eval()
        for _, (x_test, y_test) in enumerate(testloader):
          y_pred = model(x_test)
          val_loss = loss_fn(y_pred,y_test.reshape(-1,1))
          val_acc=(y_pred.reshape(-1).detach().cpu().numpy().round() == y_test.cpu().numpy()).mean()
          print("test accuracy:", val_acc)
        tb_writer.add_scalar('Test/loss', val_loss, i)
        tb_writer.add_scalar('Test/accuracy', val_acc, i)

      predicted_em_cells = np.append(predicted_em_cells, dat.specimen_id[em_cells_test_ind])
      predicted_labels_em_cells = np.append(predicted_labels_em_cells, y_pred.reshape(-1).detach().cpu().numpy().round())
      true_labels_em_cells = np.append(true_labels_em_cells, y_test.cpu().numpy())

    df=pd.DataFrame(columns= ["specimen_id", "predicted_class", "true_class"])
    df['specimen_id']= predicted_em_cells
    df['predicted_class']= predicted_labels_em_cells
    df['true_class'] = true_labels_em_cells


    df.to_csv("/home/fahimehb/Local/new_codes/cplAE_MET/data/results/classification_results/"+output_filename)
       

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))



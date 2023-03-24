import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import torch
from get_data import get_data

class StockDataset(pl.LightningDataModule):
  def __init__(self, csv_file, inp_len = 128, out_len = 8):
    self.csv_file = csv_file
    self.inp_len = inp_len
    self.out_len = out_len
    self.organizations = []
    self.URLs = []

    with open(self.csv_file, 'r') as f:
      lines = f.readlines()
    
    print(f'loading {self.csv_file}')
    for line in tqdm(lines):
      self.organizations.append(line.split(',')[0])
      self.URLs.append(line.split(',')[-1])

  def __len__(self):
    return len(self.organizations)

  def __getitem__(self, index):
    if torch.is_tensor(index):
      index = index.tolist()

    data = get_data(self.URLs[index])
    input = torch.FloatTensor(data)[:, :self.inp_len]
    output = torch.FloatTensor(data)[:, self.inp_len:self.inp_len+self.out_len]
    return input, output


class StockPred(nn.Module):

  def __init__(self):
    super (StockPred, self).__init__()

    #define layers
    self.lin1 = nn.Linear(in_features = 128, out_features = 128)
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (2, 5), stride=1)
    self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (2, 5), stride=1)
    self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (2, 5), stride=1)
    self.conv4 = nn.Conv1d(in_channels = 32, out_channels = 8, kernel_size = 5, stride=2)
    self.conv5 = nn.Conv1d(in_channels = 8, out_channels = 4, kernel_size = 5, stride=2)
    # self.conv6 = nn.Conv1d(in_channels = 4, out_channels = 2, kernel_size = 5, stride=2)
    # self.conv7 = nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = 5, stride=2)
    self.lin2 = nn.Linear(in_features = 26, out_features = 12)
    self.lin3 = nn.Linear(in_features = 12, out_features = 6)
    self.lin4 = nn.Linear(in_features = 6, out_features = 2)
    # self.activation = nn.Sigmoid()
  
  def forward(self, x):
    '''

    '''
    
    x = x.unsqueeze(1)
    y = self.lin1(x)
    y = self.conv1(y)
    # y = self.activation(y)
    y = self.conv2(y)
    y = self.conv3(y).squeeze(2)
    y = self.conv4(y)
    y = self.conv5(y)
    y= self.lin2(y)
    y= self.lin3(y)
    y= self.lin4(y)
    y = y.abs()
    #y = y.squeeze()
    # y = self.transformer_encoder(y)
    return y
  
class StockPred_out1(nn.Module):

  def __init__(self):
    super (StockPred_out1, self).__init__()

    #define layers
    #self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first = True)
    #self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
    self.lin1 = nn.Linear(in_features = 128, out_features = 128)
    self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 32, kernel_size = 5, stride=2)
    self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 5, stride=2)
    self.conv3 = nn.Conv1d(in_channels = 64, out_channels = 32, kernel_size = 5, stride=2)
    self.conv4 = nn.Conv1d(in_channels = 32, out_channels = 8, kernel_size = 5, stride=2)
    self.conv5 = nn.Conv1d(in_channels = 8, out_channels = 1, kernel_size = 3, stride=2)
    self.lin2 = nn.Linear(in_features = 2, out_features = 2)
  
  def forward(self, x):
    '''

    '''
    
    x = x.unsqueeze(1)
    y = self.lin1(x)
    y = self.conv1(y)
    y = self.conv2(y)
    y = self.conv3(y)
    y = self.conv4(y)
    y = self.conv5(y)
    y= self.lin2(y)
    y = y.abs()
    y = y.squeeze()
    # y = self.transformer_encoder(y)
    return y

class StockPred_t(nn.Module):

  def __init__(self):
    super (StockPred_t, self).__init__()

    #define layers
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=5, nhead=5, batch_first = True)
    self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
    self.lin1 = nn.Linear(in_features = 128, out_features = 128)
    self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 32, kernel_size = 5, stride=2)
    self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 5, stride=2)
    self.conv3 = nn.Conv1d(in_channels = 64, out_channels = 32, kernel_size = 5, stride=2)
    self.conv4 = nn.Conv1d(in_channels = 32, out_channels = 8, kernel_size = 5, stride=2)
    self.conv5 = nn.Conv1d(in_channels = 8, out_channels = 1, kernel_size = 5, stride=2)
    self.lin2 = nn.Linear(in_features = 5, out_features = 5)
  
  def forward(self, x):
    '''

    '''
    
    x = x.unsqueeze(1)
    y = self.lin1(x)
    y = self.conv1(y)
    y = self.conv2(y)
    y = self.conv3(y)
    y = self.conv4(y)
    y = self.conv5(y)
    y= self.lin2(y)
    y = y.abs()
    y = y.squeeze()
    y = self.transformer_encoder(y)
    return y

class StockPredictor(pl.LightningModule):
  def __init__(self, learning_rate, batch_size):
    super().__init__()
    self.save_hyperparameters()
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.model = StockPred()

  def training_step(self, batch, batch_idx):
    '''
    
    '''
    input, output = batch
    pred = self.model(input)
    loss = nn.functional.mse_loss(pred, output)
    self.log("train_loss", loss, prog_bar=True)
    return loss



  def validation_step(self, batch, batch_idx):
    input, output = batch
    pred = self.model(input)
    loss = nn.functional.mse_loss(pred, output)
    self.log("val_loss", loss, prog_bar=True)

    if (batch_idx % 4) == 0:
      print('REF: ', output[0])
      print('PRED: ', pred[0])
    return loss

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    return optimizer

  def train_dataloader(self):
    dataset = StockDataset('database.csv', inp_len = 128, out_len = 2)
    dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True, num_workers = 4)
    return dataloader

  def val_dataloader(self):
    dataset = StockDataset('database.csv', inp_len = 128, out_len = 2)
    dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = False, num_workers = 4)
    return dataloader
  
  def forward(self, x):
    return self.model(x)


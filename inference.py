from StockCore import StockPredictor
from StockCore import StockDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from get_data import get_data
import torch

def inference(x,
              checkpoint = 'checkpoints/lin-conv-lin-out1/epoch=294-step=1770.ckpt',
              device = 'cpu', # cpu or gpu
              batch_size = 1, 
              learning_rate = 1e-3,
              ):
    
    if device == 'cuda':
        model = StockPredictor.load_from_checkpoint(checkpoint_path=checkpoint, learning_rate=learning_rate, batch_size=batch_size).eval().cuda(device=0)
    elif device == 'cpu':
        model = StockPredictor.load_from_checkpoint(checkpoint_path=checkpoint, learning_rate=learning_rate, batch_size=batch_size).eval()
    pred = model(x).detach().numpy()
    return pred

if __name__ == '__main__':
    
    organization_name = 'OLYMPIC'
    database = {}
    with open('database.csv', 'r') as f:
      lines = f.readlines()
    
    for line in tqdm(lines):
        database[line.split(',')[0]] = line.split(',')[1]
    
    URL = database[organization_name]
    data = get_data(URL)
    input = torch.FloatTensor(data[-129:-1]).unsqueeze(0)
    pred = inference(input)
    print('last day price (BDT):',input[0, -1].numpy())
    print('Next 2 days Predicted LTP (BDT):',pred)



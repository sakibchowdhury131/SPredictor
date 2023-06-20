from StockCore import StockPredictor
from tqdm import tqdm
from get_data import get_data
import torch
from plot_candle_chart import plot_data
import numpy as np

def inference(x,
              checkpoint = 'checkpoints/4D_data_mmul/epoch=3-step=1928.ckpt',
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
    
    organization_name = 'SONALILIFE'
    
    data = torch.FloatTensor(get_data(organization_name))

    final_gt = data[:, -128:]
    plot_data(final_gt)
    # predict the next 7 days
    predicted_days = 8
    input = data[:, -128-predicted_days:-predicted_days]
    input = input.unsqueeze(0)
    for i in tqdm(range(predicted_days)):
        pred = inference(input).squeeze()
        next_day = pred[:, 0]
        print(next_day)
        input[:, :, :-1] = input[:, :, 1:]
        input[:, :, -1] = torch.tensor(next_day)
    plot_data(input.squeeze())
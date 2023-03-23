from StockCore import StockPredictor
from StockCore import StockDataset
from torch.utils.data import DataLoader

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
    pred = model(x)
    return pred

if __name__ == '__main__':
    
    stockDataset = StockDataset('database.csv',  inp_len = 128, out_len = 2)
    dataloader = DataLoader(stockDataset, batch_size = 1, shuffle = True, num_workers = 2)
    dataiter = iter(dataloader)
    input, output = next(dataiter)
    pred = inference(input).detach().numpy()

    print(output.numpy(), pred)



from get_data import get_data
from tqdm import tqdm
from StockCore import StockPredictor
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import pytorch_lightning as pl

def check_database(csv_file):
    with open(csv_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        unpacked = line.split(',')
        URL = unpacked[-1]
        if len(get_data(URL)) <400:
            print(f'problem identified in {unpacked[0]}: data size --> {len(get_data(URL))}')

def train():
    # check_database('database.csv')
    model = StockPredictor(learning_rate = 1e-3, batch_size = 16)
    ModelSummary(model)
    EXP_NAME = '4D_data_activation'
    logger = TensorBoardLogger("tensorboard", name=EXP_NAME)

    checkpoint_callback = ModelCheckpoint(dirpath=f"./checkpoints/{EXP_NAME}", 
                                        save_top_k=2, 
                                        monitor="val_loss",
                                        mode="min",)

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        devices=1,
        limit_train_batches = 100000,
        max_epochs=2000,
        accelerator="gpu",
        logger = logger,
        log_every_n_steps=2,    # since we have only 6 batches
    ) 
    trainer.fit(model=model, 
                # ckpt_path = "checkpoints/4D_data_activation/epoch=12-step=78.ckpt",
                )
    

if __name__ == '__main__':
    train()
    

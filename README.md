# StockPredictor
trained on 128 days of history to predict next 2 days

## Dependencies
```
PyTorch
Pytorch Lightning
Tensorboard
```

## Setup Environment

- Create a virtual environment
    ```bash
    python -m venv env
    source env/bin/activate
    ```
- Install Dependencies
    ```bash
    pip install torch torchvision torchaudio
    pip install pytorch-lightning
    pip install tensorboard
    pip install pandas
    ```
    or you can simply install everything by running 

    ```bash
    pip install -r requirements.txt
    ```
## Inference 
The trained model is stored in ```checkpoints``` directory. To run inference, simply use:
```bash
python inference.py
```

## Training
The training script is provided in the ```StockCore.py``` file. Simply modify the ```StockDataset``` class to adapt to your custom dataset. To execute training, simply run:
```bash
python train.py
```
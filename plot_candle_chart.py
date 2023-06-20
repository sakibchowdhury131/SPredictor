import plotly.graph_objects as go
from get_data import get_data
import pandas as pd
from datetime import datetime

def plot_data(data):
    fig = go.Figure(data=[go.Candlestick(
                    open=data[0, :],
                    high=data[1, :],
                    low=data[2, :],
                    close=data[3, :])])

    fig.show()


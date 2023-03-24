import requests as r
from pandas.io.json import json_normalize

def get_data(URL = 'https://stocknow.com.bd/api/v1/instruments/JAMUNAOIL/history?resolution=D&additionalField=trade'):
    action_getURL = URL
    res = r.get(action_getURL)
    return res.json()[:4]

#print(res.json()[3])
#headers = {"user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}


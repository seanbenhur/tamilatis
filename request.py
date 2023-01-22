import requests
import json

url = "http://localhost:7000/"

payload={"data":"ஹைதராபாத்தில் இருந்து உதய்பூர் செல்லும் விமானங்களைக் காட்டு"}
payload = json.dumps(payload)
headers = {'Content-Type': 'application/json'}
response = requests.request("POST", url, headers=headers, json=payload)

print(response.text)

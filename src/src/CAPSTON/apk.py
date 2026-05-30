import requests

url = "http://10.0.0.35:5000/predict"  # Update with your server IP
data = [
    {
        "L4_SRC_PORT": 0.5,
        "L4_DST_PORT": -0.2,
        "PROTOCOL": 1,
        "L7_PROTO": 0,  
        "IN_BYTES": 50,
        "OUT_BYTES": 30,
        "IN_PKTS": 2,
        "OUT_PKTS": 1,
        "TCP_FLAGS": 0,
        "FLOW_DURATION_MILLISECONDS": 1000,
        "Label": 0
    }
]

response = requests.post(url, json=data)
print("API Response:", response.json())  # Expected: {"prediction": [1]} or {"prediction": [0]}

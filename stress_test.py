import requests
import random
import time

url = "http://127.0.0.1:8000/predict"

for i in range(200):

    data = {
        "age": random.randint(30, 80),
        "sex": random.randint(0, 1),
        "cp": random.randint(0, 3),
        "trestbps": random.randint(90, 180),
        "chol": random.randint(120, 350),
        "fbs": random.randint(0, 1),
        "restecg": random.randint(0, 2),
        "thalach": random.randint(70, 200),
        "exang": random.randint(0, 1),
        "oldpeak": round(random.uniform(0, 5), 1),
        "slope": random.randint(0, 2),
        "ca": random.randint(0, 4),
        "thal": random.randint(0, 3)
    }

    response = requests.post(url, json=data)

    print(f"{i+1} -> {response.status_code}")

    time.sleep(0.2)

print("Selesai")
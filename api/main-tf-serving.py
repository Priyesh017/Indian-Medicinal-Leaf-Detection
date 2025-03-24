from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import requests

app = FastAPI()

#MODEL = tf.keras.models.load_model("saved_models/1.keras")

endpoint = "https://localhost:8501/v1/models/MedicinalLeafDataset/predict"

CLASS_NAMES = ["Alovera", "Bamboo", "Chilly", "Eucalyptus", "Guava", "Hibiscus", "Jackfruit"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    json_data = {
        "instances": img_batch.tolist()
    }
    response = requests.post(endpoint, json = json_data)
    prediction = response.json()["predictions"][0]

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return{
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)

from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
app= FastAPI()
origins={
    "https://plantdisease.vercel.app/",
    "http://localhost:3000"
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL =tf.keras.models.load_model("saved_models/4")
CLASS_NAMES=["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
@app.get("/ping")
async def ping():
    return "Hello, I am alive"
def read_file_as_image(data) -> np.ndarray:
        print(data)
        image=Image.open(BytesIO(data))
        return image
@app.post("/predict")
async def predict(
        file: UploadFile
):
    bytes=await file.read()

    image=read_file_as_image(bytes)
    img_batch=np.expand_dims(image,0)
    prediction=MODEL.predict(img_batch)
    predicted_class=CLASS_NAMES[np.argmax(prediction[0])]
    confidence=np.max(prediction[0])
    return {
        'class':predicted_class,
        'confidence':float(confidence)
    }
if __name__ =="__main__":
     uvicorn.run(app,host='localhost',port=8000)

import base64
from urllib import response
from flask import Flask, render_template
import requests ,io
import urllib.request
from PIL import Image

app = Flask(__name__)

@app.get("/")
def receive():
    # image_url = "http://127.0.0.1:5001/"
    image_url = "https://mlp-web-service.herokuapp.com/"
    response = requests.get(image_url)
    img = response.content
    
    buffered = io.BytesIO(img)
    img_url = base64.b64encode(buffered.getvalue()).decode()
    
    return render_template("home.html",images={ 'image': img_url })

if __name__ == "__main__":
    app.run(debug=True)
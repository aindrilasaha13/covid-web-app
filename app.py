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
    image_url_1 = "https://mlp-web-service.herokuapp.com/"
    response_1 = requests.get(image_url_1)
    img_1 = response_1.content
    buffered_1 = io.BytesIO(img_1)
    img_url_1 = base64.b64encode(buffered_1.getvalue()).decode()
    
    image_url_2 = "https://lr-web-service.herokuapp.com/"
    response_2 = requests.get(image_url_2)
    img_2 = response_2.content
    buffered_2 = io.BytesIO(img_2)
    img_url_2 = base64.b64encode(buffered_2.getvalue()).decode()
    
    return render_template("home.html",images={ 'image_1': img_url_1,'image_2': img_url_2 })

if __name__ == "__main__":
    app.run(debug=True)
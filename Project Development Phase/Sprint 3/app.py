from flask import Flask,render_template,request
from PIL import Image, ImageOps
import os
import random
import string
from pathlib import Path
import numpy as np
from CNN import recognize

def random_name_generator(n):
  return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))

app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/submit',methods=['POST'])
def submit():
    if request.method=='POST':
      image = request.files['image']
      img_name = image.filename
      file_path = os.path.join('static/data/', img_name)
      image.save(file_path)
      img = Image.open(image).convert("L")
      img = img.resize((255, 255))
      img.save(os.path.join('static/thumb/', "255X255_"+img_name))
      # best = (9, 73.19)
      # others = [(0, 9.15),(1, 0.35000000000000003), (2, 0.4), (3, 0.0), (4, 109.9), (5, 4.1499999999999995), (6, 3.5), (7, 3.4000000000000004), (8, 3.15), (9, 365.95)]
      # others = [(0, 1.83), (1, 0.07), (2, 0.08), (3, 0.0), (4, 21.98), (5, 0.83), (6, 0.7), (7, 0.68), (8, 0.63),(9, 73.19)]
      best, others = recognize(image)
      
      return render_template("submit.html", best=best, others=others, img_name=img_name)


if __name__=="__main__":
    app.run()


if __name__ == '__main__':
    app.run(debug=True)

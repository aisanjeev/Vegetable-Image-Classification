from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np


app = Flask(__name__)

dic = {0 : 'Bean', 1 : 'Bitter_Gourd', 2 :"Bottle_Gourd", 3:'Brinjal', 4: 'Broccoli', 5: 'Cabbage',
	   6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9:'Cucumber', 10: 'Papaya', 11: 'Potato',
	   12: 'Pumpkin', 13: 'Radish', 14: 'Tomata'}


model = load_model('Vegetables-Image-Classification.hdf5')
model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i)
	i = i.reshape(1, 224, 224, 3)
	p = np.argmax(model.predict(i), axis=1)

	return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
	return render_template("home.html")

@app.route("/about")
def about_page():
	return "About You..!!!"



@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)


	return render_template("home.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
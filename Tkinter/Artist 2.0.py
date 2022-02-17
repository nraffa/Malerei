import silence_tensorflow.auto
import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
from tkinter import *
import tkinter as tk
from PIL import ImageTk , Image 
from tkinter import filedialog
import os
import os.path

#PARTE 1: TKINTER
#Magenta
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


#Abro cuadro para ver si quiero multiples estilos
def printentry():
	global e
	global entry
	entry = int(e.get() )
	print (entry)
	return entry

from tkinter import *
root2 = Tk()

root2.title('Name')

e = Entry(root2)
e.pack()
e.focus_set()

b = Button(root2,text='Enter',command=printentry)
b.pack(side='bottom')
root2.mainloop()

booleano = False

if entry > 1:
	booleano = True #True si quiero muchas imagenes
else:
	booleano = False

#defino funcion para el boton del GUI
def open_content():
 global my_imageC
 global root_filenameC
 root_filenameC = filedialog.askopenfilename(title = "Select content image")
 #my_labelC = Label(root , text = root_filenameC).pack() #creo que es para visualizar la direc de la imagen
 my_imageC = Image.open(root_filenameC)
 return root_filenameC

def open_style():
 global my_imageS
 global root_filenameS
 root_filenameS = filedialog.askopenfilename(title = "Select style image")
 #my_labelS = Label(root , text = root_filenameS).pack() 
 my_imageS = Image.open(root_filenameS)
 return root_filenameS

def open_styles():
	global files
	files = filedialog.askopenfilenames(parent=root,title='Select multiple style images')
	files = root.tk.splitlist(files)
	global my_imageS
	global root_filenameS
	for imgs in files:
		my_imageS = Image.open(imgs)
	return files

'''
root = tkinter.Tk()

files = filedialog.askopenfilenames(parent=root,title='Choose a file')

	#files = raw_input("which files do you want processed?")

files = root.tk.splitlist(files)

print ("list of filez =",files)

'''
#genero ventana
root = Tk()
#root.iconbitmap("c:/Users/nicor/OneDrive/Documentos/Python/Neural Style Transfer/Executable/paint_brush.png")
root.title("DobleFalta - Gallery")

#defino dimensiones ventana
width = 400
height = 300
size = str(width) + "x" + str(height)
root.geometry(size)

#defino botones --> hago modificacion aca para que agarre varias imagenes de estilo
my_buttonC = Button(root, text = "Open Content Image" , command = open_content).pack()

if booleano == True:
	my_buttonS = Button(root, text = "Open Style Image" , command = open_styles).pack()
else:
	my_buttonS = Button(root, text = "Open Style Image" , command = open_style).pack()

#defino la salida y donde se va a guardar la imagen
def save_as():
	global new_image_dir
	new_image_dir = filedialog.askdirectory() 
	return new_image_dir

Save_button = Button(root , text = "Save Location" , command = save_as).pack() 
Pintar = Button(root , text = "Paint Work of Art" , command = lambda:root.quit()).pack()
Salir = Button(root , text = "Exit" , command = lambda:exit()).pack()

root.mainloop()

#PARTE 2: NEURAL STYLE TRANSFER

#defino funcion de preprocessing de imagen
def load_image(img_path):
	img = tf.io.read_file(img_path) #leo archivo
	img = tf.image.decode_image(img , channels = 3) #me aseguro que tiene 3 canales (rgb)
	img = tf.image.convert_image_dtype(img , tf.float32) #me aseguro que tenga el formato correcto: float 32 bit
	img = img[tf.newaxis , :] #me aseguro que la imagen este en un array
	return img

#cargo imagenes
content_image = load_image(root_filenameC)
#print(content_image.shape) #(1, 720, 1200 , 3) --> una imagen, height and width , channels
if booleano == False:
	style_image = load_image(root_filenameS)

	#visualizo imagenes
	plt.imshow(np.squeeze(content_image)) #np.squeeze hace que se pueda visualizar el tensor
	plt.show()

	plt.imshow(np.squeeze(style_image))
	plt.show()

	#aplico Neural Style Transfer
	stylized_image = model(tf.constant(content_image) , tf.constant(style_image))[0]

	plt.imshow(np.squeeze(stylized_image))
	plt.show()

	cv2.imwrite(new_image_dir + "\Generated_Image.jpg", cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2RGB)) #exporto imagen
	#se multiplica por 255 porque el algoritmo realizo una normalizacion para poder trabajar
else:
	i = 0
	for imgs in files:
		i = i + 1
		style_image = load_image(imgs)
		#visualizo imagenes
		plt.imshow(np.squeeze(content_image)) #np.squeeze hace que se pueda visualizar el tensor
		plt.show()

		plt.imshow(np.squeeze(style_image))
		plt.show()

		#aplico Neural Style Transfer
		stylized_image = model(tf.constant(content_image) , tf.constant(style_image))[0]

		plt.imshow(np.squeeze(stylized_image))
		plt.show()

		cv2.imwrite(new_image_dir + "\Generated_Image " + str(i) + ".jpg", cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2RGB)) #exporto imagen
		#se multiplica por 255 porque el algoritmo realizo una normalizacion para poder trabajar





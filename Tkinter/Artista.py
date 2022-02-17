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

#genero ventana
root = Tk()
#root.iconbitmap("c:/Users/nicor/OneDrive/Documentos/Python/Neural Style Transfer/Executable/paint_brush.png")
root.title("DobleFalta - Gallery")

#defino dimensiones ventana
width = 400
height = 300
size = str(width) + "x" + str(height)
root.geometry(size)

#defino botones
my_buttonC = Button(root, text = "Open Content Image" , command = open_content).pack()
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
print(content_image.shape) #(1, 720, 1200 , 3) --> una imagen, height and width , channels
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





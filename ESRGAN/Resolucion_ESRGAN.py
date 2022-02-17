#TUTORIAL DE TENSORFLOW: SUPER RESOLUCION DE IMAGEN USANDO ESRGAN
#https://www.tensorflow.org/hub/tutorials/image_enhancing

'''
Esta colab demuestra el uso del módulo de concentrador de TensorFlow para la red de adversarios
generativos de súper resolución mejorada ( por Xintao Wang et.al. ) [ Documento ] [ Código ]
para mejorar la imagen. (Preferiblemente imágenes bicúbicamente submuestreadas).
Modelo entrenado en el conjunto de datos DIV2K (en imágenes con muestreo reducido bicúbicamente)
en parches de imagen de tamaño 128 x 128.

Preparando el entorno
'''
import silence_tensorflow.auto
import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"


folder_path = 'C:/Users/nicor/OneDrive/Documentos/Python/Neural Style Transfer/inspiraciones/'
image_name = 'Scuba Caribe Hat.jpg'
image_path = folder_path + image_name

#Definicion de funciones auxiliares
def preprocess_image(image_path):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
  """
    Saves unscaled Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save to.
  """
  if not isinstance(image, Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  image.save("%s.jpg" % filename)
  print("Saved as %s.jpg" % filename)

#%matplotlib inline
def plot_image(image, title=""):
  """
    Plots images from image tensors.
    Args:
      image: 3D image tensor. [height, width, channels].
      title: Title to display in the plot.
  """
  image = np.asarray(image)
  image = tf.clip_by_value(image, 0, 255)
  image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  plt.imshow(image)
  plt.axis("off")
  plt.title(title)
  plt.show()

#Realización de súper resolución de imágenes cargadas desde la ruta
hr_image = preprocess_image(image_path)

# Plotting Original Resolution image
plot_image(tf.squeeze(hr_image), title="Original Image")
save_image(tf.squeeze(hr_image), filename="Original Image")

model = hub.load(SAVED_MODEL_PATH)

start = time.time()
fake_image = model(hr_image)
fake_image = tf.squeeze(fake_image)
print("Time Taken: %f" % (time.time() - start))

# Plotting Super Resolution Image
plot_image(tf.squeeze(fake_image), title="Super Resolution")
save_image(tf.squeeze(fake_image), filename="Super Resolution")

print("Resolucion Original Image: " + hr_image.shape)
print("Resolucion HR Image: " + fake_image.shape)

'''
#Mido el performance del modelo con la metrica PSNR (Peak Signal to Noise Ratio)
plot_image(tf.squeeze(fake_image), title="Super Resolution")
# Calculating PSNR wrt Original Image
psnr = tf.image.psnr(
    tf.clip_by_value(fake_image, 0, 255),
    tf.clip_by_value(hr_image, 0, 255), max_val=255)
print("PSNR Achieved: %f" % psnr)

'''


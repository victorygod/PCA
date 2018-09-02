import tensorflow as tf
import numpy as np
from scipy import misc
import PCA
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
shape = np.shape(x_train)
x_train = np.reshape(x_train, (shape[0], shape[1]*shape[2]))
y_train = np.reshape(y_train, (shape[0], 1))
shape = np.shape(x_test)
x_test = np.reshape(x_test, (shape[0], shape[1]*shape[2]))
y_test = np.reshape(y_test, (shape[0], 1))


if __name__ == "__main__":
	p, u = PCA.PCA(x_train)

	pic1 = x_train[0]
	pic1 = np.reshape(pic1, (28,28))
	misc.imsave("origin.jpg", pic1)

	pic_r = PCA.reconstruct(p[0], u)
	pic_r = np.reshape(pic_r, (28,28))
	misc.imsave("output.jpg", pic_r)

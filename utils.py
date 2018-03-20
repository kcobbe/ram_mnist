import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class Plotter(object):
    def __init__(self, rows=4, columns=4):
        fig=plt.figure(figsize=(8, 8))
        idx = 0

        plots = []

        for i in range(1, columns*rows +1):
            img = np.zeros((10, 10, 3))
            ax = fig.add_subplot(rows, columns, i)
            ax.axis('off')

            new_plot = plt.imshow(img, extent=[0, 1, 0, 1])
            plots.append(new_plot)

            idx += 1

        plt.show(block=False)
        plt.pause(.1)
        plt.ion()

        self.fig = fig
        self.plots = plots

    def plot(self, images):
        k = min(len(images), len(self.plots))

        for i in range(k):
            self.plots[i].set_data(images[i])

        self.fig.canvas.draw()
        plt.pause(.1)


def load_MNIST():
    mnist_data = tf.contrib.learn.datasets.mnist.load_mnist(train_dir='MNIST-data')
    train_data = mnist_data.train.images
    test_data = mnist_data.test.images
    train_data = np.reshape(train_data, (np.shape(train_data)[0], 28, 28, 1))
    test_data = np.reshape(test_data, (np.shape(test_data)[0], 28, 28, 1))
    train_labels = mnist_data.train.labels
    test_labels = mnist_data.test.labels

    return train_data, train_labels, test_data, test_labels

def create_translated_mnist(output_dim):
    print('Creating translate MNIST set with output dim', output_dim)

    train_data, train_labels, test_data, test_labels = load_MNIST()
    
    def translate(batch):
        (n, input_dim, _, c) = np.shape(batch)
        data = np.zeros(shape=(n,output_dim,output_dim,c), dtype=np.float32)

        for k in range(n):
            if output_dim == input_dim:
                i, j = 0, 0
            else:
                i, j = np.random.randint(0, output_dim-input_dim, size=2)

            data[k, i:i+input_dim, j:j+input_dim, :] += batch[k]

        return data

    translated_train = translate(train_data)
    translated_test = translate(test_data)

    return translated_train, train_labels, translated_test, test_labels
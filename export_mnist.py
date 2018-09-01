import os
import struct
import numpy as np
import scipy.misc

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

images, labels = load_mnist("")

js_arrary = ''
label_array = ''
js_arrary += 'var images = ['
label_array += 'var labels = ['
for j in range(len(images)):
    if labels[j] == 0:
        scipy.misc.toimage(images[j].reshape(28, 28), cmin=0.0,cmax=1.0).save('mnist/train_{0}_{1}.jpg'.format(labels[j], j))
        label_array += '{0},'.format(labels[j])
        js_arrary += '['
        for i in range(len(images[j])):
            js_arrary += '{0},'.format(images[j][i])
        js_arrary += '],'
    elif labels[j] == 1:
        scipy.misc.toimage(images[j].reshape(28, 28), cmin=0.0,cmax=1.0).save('mnist/train_{0}_{1}.jpg'.format(labels[j], j))
        label_array += '{0},'.format(labels[j])
        js_arrary += '['
        for i in range(len(images[j])):
            js_arrary += '{0},'.format(images[j][i])
        js_arrary += '],'
js_arrary += '];\n'
label_array += '];\n'
# print(js_arrary)

f = open("mnist.js", 'a+')
f.write(js_arrary)
f.write(label_array)
f.close()

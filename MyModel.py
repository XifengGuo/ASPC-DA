from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from tensorflow.keras.initializers import VarianceScaling
import numpy as np


def autoencoder(dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')

    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


class MyIterator(NumpyArrayIterator):
    """
    The only difference with NumpyArrayIterator is this.next() returns (samples, index) while NumpyArrayIterator
    returns samples
    """
    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array), index_array


class MyImageGenerator(ImageDataGenerator):
    """
    The only difference with ImageDataGenerator is this.flow().next() returns (samples, index) while ImageDataGenerator
    returns samples
    """
    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return MyIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


def generator(image_generator, x, y=None, sample_weight=None, batch_size=32, shuffle=True):
    """
    Data generator that supplies training batches for Model().fit_generator.
    :param image_generator: MyImageGenerator, defines and applies transformations for the input images
    :param x: input image data, supports shape=[n_samples, width, height, channels] and [n_samples, n_features]
    :param y: the target of the network's output
    :param sample_weight: weight for x, shape=[n_samples]
    :param batch_size: batch size
    :param shuffle: whether to shuffle the data
    :return: An iterator, outputs [batch_x, batch_x] or [batch_x, batch_y] or
            [batch_x, batch_y, batch_sample_weight] each time
    """
    if len(x.shape) > 2:  # image
        gen0, idx = image_generator.flow(x, shuffle=shuffle, batch_size=batch_size)
        while True:
            batch_x = gen0.next()
            result = [batch_x] + \
                     [batch_x if y is None else y[idx]] + \
                     ([] if sample_weight is None else [sample_weight[idx]])
            yield tuple(result)
    else:  # if the sample is represented by vector, need to reshape to matrix and then flatten back
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen0 = image_generator.flow(np.reshape(x, im_shape), shuffle=shuffle, batch_size=batch_size)
        while True:
            batch_x, idx = gen0.next()
            batch_x = np.reshape(batch_x, [batch_x.shape[0], x.shape[-1]])
            result = [batch_x] + \
                     [batch_x if y is None else y[idx]] + \
                     ([] if sample_weight is None else [sample_weight[idx]])
            yield tuple(result)


def random_transform(x, datagen):
    if len(x.shape) > 2:  # image
        return datagen.flow(x, shuffle=False, batch_size=x.shape[0]).next()

    # if input a flattened vector, reshape to image before transform
    width = int(np.sqrt(x.shape[-1]))
    if width * width == x.shape[-1]:  # gray
        im_shape = [-1, width, width, 1]
    else:  # RGB
        width = int(np.sqrt(x.shape[-1] / 3.0))
        im_shape = [-1, width, width, 3]
    gen = datagen.flow(np.reshape(x, im_shape), shuffle=False, batch_size=x.shape[0])
    return np.reshape(gen.next(), x.shape)

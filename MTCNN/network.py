import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, PReLU, Flatten, Softmax
from tensorflow.python.keras.models import Model
import numpy as np

class NetworkFactory():

    def pnet(self, input_shape=None):

        if input_shape is None:
            input_shape = (None, None, 3)

        net = Input(input_shape)

        pnet_layer = Conv2D(10, (3, 3), strides=(1, 1), padding='valid')(net)
        pnet_layer = PReLU(shared_axes=[1, 2])(pnet_layer)
        pnet_layer = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(pnet_layer)

        pnet_layer = Conv2D(16, (3, 3), strides=(1, 1), padding='valid')(pnet_layer)
        pnet_layer = PReLU(shared_axes=[1, 2])(pnet_layer)

        pnet_layer = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(pnet_layer)
        pnet_layer = PReLU(shared_axes=[1, 2])(pnet_layer)

        pnet_out1 = Conv2D(2, (1, 1), strides=(1, 1))(pnet_layer)
        pnet_out1 = Softmax(axis=3)(pnet_out1)

        pnet_out2 = Conv2D(4, (1, 1), strides=(1, 1))(pnet_layer)

        p_net = Model(net, [pnet_out2, pnet_out1])

        return p_net
    

    def rnet(self, input_shape=None):

        if input_shape is None:
            input_shape = (24, 24, 3)

        net = Input(input_shape)

        rnet_layer = Conv2D(28, (3, 3), strides=(1, 1), padding='valid')(net)
        rnet_layer = PReLU(shared_axes=[1, 2])(rnet_layer)
        rnet_layer = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(rnet_layer)

        rnet_layer = Conv2D(48, (3, 3), strides=(1, 1), padding='valid')(rnet_layer)
        rnet_layer = PReLU(shared_axes=[1, 2])(rnet_layer)
        rnet_layer = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(rnet_layer)

        rnet_layer = Conv2D(64, (2, 2), strides=(1, 1), padding='valid')(rnet_layer)
        rnet_layer = PReLU(shared_axes=[1, 2])(rnet_layer)
        rnet_layer = Flatten()(rnet_layer)

        rnet_layer = Dense(128)(rnet_layer)
        rnet_layer = PReLU()(rnet_layer)

        rnet_out1 = Dense(2)(rnet_layer)
        rnet_out1 = Softmax(axis=1)(rnet_out1)

        rnet_out2 = Dense(4)(rnet_layer)

        rnet = Model(net, [rnet_out2, rnet_out1])

        return rnet

    
    def onet(self, input_shape=None):

        if input_shape is None:
            input_shape = (48, 48, 3)

        net = Input(input_shape)

        onet_layer = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(net)
        onet_layer = PReLU(shared_axes=[1, 2])(onet_layer)
        onet_layer = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(onet_layer)

        onet_layer = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(onet_layer)
        onet_layer = PReLU(shared_axes=[1, 2])(onet_layer)
        onet_layer = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(onet_layer)

        onet_layer = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(onet_layer)
        onet_layer = PReLU(shared_axes=[1, 2])(onet_layer)
        onet_layer = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(onet_layer)

        onet_layer = Conv2D(128, (2, 2), strides=(1, 1), padding='valid')(onet_layer)
        onet_layer = PReLU(shared_axes=[1, 2])(onet_layer)

        onet_layer = Flatten()(onet_layer)
        onet_layer = Dense(256)(onet_layer)
        onet_layer = PReLU()(onet_layer)

        onet_out1 = Dense(2)(onet_layer)
        onet_out1 = Softmax(axis=1)(onet_out1)

        onet_out2 = Dense(4)(onet_layer)
        onet_out3 = Dense(10)(onet_layer)

        onet = Model(net, [onet_out2, onet_out3, onet_out1])

        return onet
    
    def build_nets_from_file(self, weights_file):
        weights = np.load(weights_file, allow_pickle=True).tolist()

        p = self.pnet()
        r = self.rnet()
        o = self.onet()

        p.set_weights(weights['pnet'])
        r.set_weights(weights['rnet'])
        o.set_weights(weights['onet'])

        return p, r, o
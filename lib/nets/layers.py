#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tensorcv.models.layers import *


@add_arg_scope
def transpose_conv(x, filter_size, out_dim, data_dict, out_shape=None, use_bias=True, reuse=False, stride=2,
                   padding='SAME', trainable=False, nl=tf.identity, name='dconv'):
    """

    Args:
        x: [N, H,W, C]
        filter_size: [fh, fw, oc, ic]
        out_dim: Int, the feature channel number
        data_dict: dict, for example, vgg19 key is dict_keys(['conv5_4', 'conv5_1', 'fc6', 'conv5_3', 'conv5_2',
            'conv4_4', 'fc7', 'conv4_1', 'conv4_2', 'conv4_3', 'fc8', 'conv3_4', 'conv3_3', 'conv3_2', 'conv3_1',
            'conv1_1', 'conv1_2', 'conv2_2', 'conv2_1'])
        out_shape: 1D-Tensor,default the shape is (4,).
        use_bias: Boolean,
        reuse: Boolean,
        stride: Int or list with length 2, default=2
        padding: Str, either `'VALID'` or `'SAME'`.
        trainable: Boolean, If `True` also add the variable to the graph collection `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).\n
        nl: a function, tf.identity return a tensor with the same shape and contents as input
        name: Str, the layer name

    Returns:

    """
    # print('transpose_conv x:', x, 'out_dim:', out_dim, 'stride:', stride)
    stride = get_shape4D(stride)  # default return [1, 2, 2, 1]
    # print('data_dict keys list', data_dict.keys())
    in_dim = x.get_shape().as_list()[-1]  # int, the number of the feature channel
    # print('in_dim', in_dim)
    # TODO other ways to determine the output shape
    x_shape = tf.shape(x)
    # assume output shape is input_shape*stride
    if out_shape is None:
        out_shape = tf.stack([x_shape[0],
                              tf.multiply(x_shape[1], stride[1]),
                              tf.multiply(x_shape[2], stride[2]),
                              out_dim])  # Tensor, Dim is (4,) the value represents the (N, H, W, C)
    filter_shape = get_shape2D(filter_size) + [out_dim, in_dim]  # List, length is 4, represent the [fh, fw, oc, ic]
    with tf.variable_scope(name) as scope:
        if reuse is True:
            scope.reuse_variables()
            init_w = None
            init_b = None
        else:
            try:
                load_data = data_dict[name][0]
            except KeyError:
                load_data = data_dict[name]['weights']
            print('transpose_conv Load {} weights!'.format(name))
            # load_data = np.reshape(load_data, shape)
            # load_data = tf.nn.l2_normalize(
            #     tf.transpose(load_data, perm=[1, 0, 2, 3]))
            # load_data = tf.transpose(load_data, perm=[1, 0, 2, 3])
            init_w = tf.constant_initializer(load_data)

            if use_bias:
                try:
                    load_data = data_dict[name][1]
                except KeyError:
                    load_data = data_dict[name]['biases']
                print('transpose_conv Load {} biases!'.format(name))
                init_b = tf.constant_initializer(load_data)

        weights = tf.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w,
                                  trainable=trainable)
        if use_bias:
            biases = tf.get_variable('biases',
                                     [in_dim],
                                     initializer=init_b,
                                     trainable=trainable)
            x = tf.nn.bias_add(x, -biases)
        output = tf.nn.conv2d_transpose(x,
                                        weights,
                                        output_shape=out_shape,
                                        strides=stride,
                                        padding=padding,
                                        name=scope.name)

        # if use_bias:
        #     output = tf.nn.bias_add(output, biases)
        # TODO need test
        output.set_shape([None, None, None, out_dim])

        output = nl(output, name='output')
        return output


# https://github.com/tensorflow/tensorflow/pull/16885
def unpool_2d(pool,
              ind,
              stride=[1, 2, 2, 1],
              scope='unpool_2d'):
    """Adds a 2D unpooling op.
    https://arxiv.org/abs/1505.04366
    Unpooling layer after max_pool_with_argmax.
         Args:
             pool:        max pooled output tensor
             ind:         argmax indices
             stride:      stride is the same as for the pool
         Return:
             unpool:    unpooling tensor
    """

    with tf.variable_scope(scope):
        ind_shape = tf.shape(ind)
        # pool = pool[:, :ind_shape[1], :ind_shape[2], :]
        input_shape = tf.shape(pool)
        print('input_shape', input_shape)
        output_shape = [input_shape[0],
                        input_shape[1] * stride[1],
                        input_shape[2] * stride[2],
                        input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0],
                             output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(
            tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
            shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0],
                            set_input_shape[1] * stride[1],
                            set_input_shape[2] * stride[2],
                            set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret


def max_pool(x,
             name='max_pool',
             filter_size=2,
             stride=None,
             padding='VALID',
             switch=False):
    """
    Max pooling layer
    Args:
        x: tf.tensor
        name: Str, name scope of the layer
        filter_size: Int or list with length 2, size of filter
        stride: Int or list with length 2, Default to be the same as shape
        padding: Str, 'VALID' or 'SAME'. Use 'SAME' for FCN.
        switch: Boolean, If True, the max pooling will use tf.nn.max_pool_with_argmax, else use tf.nn.max_pool
    Returns:
        tf.tensor with name 'name'
    """

    padding = padding.upper()
    filter_shape = get_shape4D(filter_size)
    if stride is None:
        stride = filter_shape
    else:
        stride = get_shape4D(stride)
    if switch is True:
        return tf.nn.max_pool_with_argmax(
            x,
            ksize=filter_shape,
            strides=stride,
            padding=padding,
            Targmax=tf.int64,
            name=name)

    else:
        return tf.nn.max_pool(
            x,
            ksize=filter_shape,
            strides=stride,
            padding=padding,
            name=name), None

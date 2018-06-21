#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: deconv.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import scipy.misc
import argparse
import numpy as np
import tensorflow as tf
from tensorcv.dataflow.image import ImageFromFile

import config_path as config

import sys

sys.path.append('../')
from lib.nets.vgg import DeconvBaseVGG19, BaseVGG19
import lib.utils.viz as viz
import lib.utils.normalize as normlize
import lib.utils.image as uim

IM_SIZE = 224


def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imtype', type=str, default='.jpg',
                        help='Image type')
    parser.add_argument('--feat', type=str, required=True,
                        help='Choose of feature map layer')
    parser.add_argument('--id', type=int, default=None,
                        help='feature map id')

    return parser.parse_args()


def im_scale(im):
    return uim.im_rescale(im, [IM_SIZE, IM_SIZE])


if __name__ == '__main__':
    FLAGS = get_parse()

    input_im = ImageFromFile(FLAGS.imtype,
                             data_dir=config.im_path,
                             num_channel=3,
                             shuffle=False,
                             pf=im_scale,
                             )
    input_im.set_batch_size(1)
    print('size', input_im.size())
    vizmodel = DeconvBaseVGG19(config.vgg_path,
                               feat_key=FLAGS.feat,
                               pick_feat=FLAGS.id)

    vizmap = vizmodel.layers['deconvim']  #
    print('vizmap', vizmap)
    feat_op = vizmodel.feats  # 4D Tensor, Dim is [N, H, W, C], depending on FLAGS.feat
    max_act_op = vizmodel.max_act  # 1D Tensor, the cur_feats_pick max value, depending on FLAGS.feat

    act_size = vizmodel.receptive_size[FLAGS.feat]
    act_scale = vizmodel.stride[FLAGS.feat]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        max_act_list = []
        print('input_im.epochs_completed', input_im.epochs_completed)
        while input_im.epochs_completed < 1:
            im = input_im.next_batch()[0]
            max_act = sess.run(max_act_op, feed_dict={vizmodel.im: im})
            max_act_list.append(max_act)
        print('max_act_list', len(max_act_list), max_act_list)
        max_list = np.argsort(max_act_list)[::-1]
        print('max_list', len(max_list), max_list)
        im_file_list = input_im.get_data_list()[0]
        print('im_file_list', len(im_file_list), im_file_list)

        feat_list = []
        im_list = []
        for i in range(0, 10):
            im = input_im.next_batch()[0]
            file_path = os.path.join(config.im_path, im_file_list[max_list[i]])
            print('file_path', file_path)
            misc_imread = scipy.misc.imread(file_path, mode='RGB')
            print('misc_imread', type(misc_imread), misc_imread.shape)
            im = np.array([im_scale(misc_imread)])
            print('im', type(im), im.shape)
            cur_vizmap, feat_map, max_act = sess.run(
                [vizmap, feat_op, max_act_op], feed_dict={vizmodel.im: im})
            print('cur_vizmap', cur_vizmap.shape, 'feat_map', feat_map.shape, 'max_act', max_act.shape)
            act_ind = np.nonzero(feat_map)
            print('Location of max activation {}'.format(act_ind))
            # get only the first nonzero element
            print('act_ind', act_ind)
            # act_first_hw_tuple is a tuple(h, w), record the first max value h, w position in the feat_map
            act_first_hw_tuple = (act_ind[1][0], act_ind[2][0])
            print('act_first_hw_tuple', act_first_hw_tuple)
            min_x = max(0, int(act_first_hw_tuple[0] * act_scale - act_size / 2))
            max_x = min(IM_SIZE, int(act_first_hw_tuple[0] * act_scale + act_size / 2))
            min_y = max(0, int(act_first_hw_tuple[1] * act_scale - act_size / 2))
            max_y = min(IM_SIZE, int(act_first_hw_tuple[1] * act_scale + act_size / 2))
            print('min_x', min_x, 'max_x', max_x, 'min_y', min_y, 'max_y', max_y, 'act_scale', act_scale, 'act_size',
                  act_size)
            im_crop = im[0, min_x:max_x, min_y:max_y, :]
            print('im_crop', im_crop.shape)
            feat_crop = cur_vizmap[0, min_x:max_x, min_y:max_y, :]
            print('feat_crop', feat_crop.shape)
            pad_size = (act_size - im_crop.shape[0], act_size - im_crop.shape[1])
            print('pad_size', pad_size)
            im_crop = np.pad(im_crop,
                             ((0, pad_size[0]), (0, pad_size[1]), (0, 0)),
                             'constant',
                             constant_values=0)
            feat_crop = np.pad(feat_crop,
                              ((0, pad_size[0]), (0, pad_size[1]), (0, 0)),
                              'constant',
                              constant_values=0)

            feat_list.append(feat_crop)
            im_list.append(im_crop)

        viz.viz_filters(np.transpose(feat_list, (1, 2, 3, 0)),
                        [3, 3],
                        os.path.join(config.save_path, '{}_feat.png'.format(FLAGS.feat)),
                        gap=2,
                        gap_color=0,
                        nf=normlize.indentity,
                        shuffle=False)
        viz.viz_filters(np.transpose(im_list, (1, 2, 3, 0)),
                        [3, 3],
                        os.path.join(config.save_path, '{}_im.png'.format(FLAGS.feat)),
                        gap=2,
                        gap_color=0,
                        nf=normlize.indentity,
                        shuffle=False)

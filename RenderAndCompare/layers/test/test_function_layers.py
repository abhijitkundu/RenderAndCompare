#!/usr/bin/env python

import _init_paths
import numpy as np
from test_gradient_for_python_layer import test_gradient_for_python_layer


def test_AngularExpectation_layer():
    # set the inputs
    input_names_and_values = [('expected_cs', np.random.randn(10, 24))]
    output_names = ['normalized_cs']
    py_module = 'RenderAndCompare.layers'
    py_layer = 'AngularExpectation'
    param_str = '-b 24'
    propagate_down = [True]
    test_gradient_for_python_layer(input_names_and_values, output_names, py_module, py_layer, param_str, propagate_down)


def test_L2Normalization_layer():
    # set the inputs
    input_names_and_values = [('expected_cs', np.random.randn(10, 24))]
    output_names = ['normalized_cs']
    py_module = 'RenderAndCompare.layers'
    py_layer = 'L2Normalization'
    param_str = ''
    propagate_down = [True]
    test_gradient_for_python_layer(input_names_and_values, output_names, py_module, py_layer, param_str, propagate_down)


def test_ConstantMultiply_layer():
    input_names_and_values = [('bottom', np.random.randn(10, 24))]
    output_names = ['10_times_bottom']
    py_module = 'RenderAndCompare.layers'
    py_layer = 'ConstantMultiply'
    param_str = '-m 10'
    propagate_down = [True]
    test_gradient_for_python_layer(input_names_and_values, output_names, py_module, py_layer, param_str, propagate_down)


if __name__ == '__main__':
    test_AngularExpectation_layer()
    test_L2Normalization_layer()
    # test_ConstantMultiply_layer()

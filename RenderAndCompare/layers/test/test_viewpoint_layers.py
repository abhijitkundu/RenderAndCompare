#!/usr/bin/env python

import _init_paths
import numpy as np
from test_gradient_for_python_layer import test_gradient_for_python_layer


def test_ViewpointExpectation_layer():
    # set the inputs
    data = np.random.rand(10, 24)
    data = data / data.sum(axis=1).reshape(-1, 1)
    input_names_and_values = [('view_prob', data)]
    output_names = ['veiw_normalized_cs']
    py_module = 'RenderAndCompare.layers'
    py_layer = 'ViewpointExpectation'
    param_str = '-b 24'
    propagate_down = [True]
    test_gradient_for_python_layer(input_names_and_values, output_names, py_module, py_layer, param_str, propagate_down)


if __name__ == '__main__':
    test_ViewpointExpectation_layer()

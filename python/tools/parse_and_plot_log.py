#!/usr/bin/env python

"""
Parse caffe training log and plot them

Evolved from caffe's parse_log.py
"""

import os
import re
import argparse
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def parse_log(path_to_log):
    """Parse log file
    Returns (train_dict_list, test_dict_list)

    train_dict_list and test_dict_list are lists of dicts that define the table
    rows
    """

    regex_iteration = re.compile('Iteration (\d+)')
    regex_iteration_loss = re.compile('loss = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')
    regex_train_output = re.compile(
        'Train net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_test_output = re.compile(
        'Test net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_learning_rate = re.compile(
        'lr = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')

    # Pick out lines of interest
    iteration = -1
    learning_rate = float('NaN')
    total_loss = float('NaN')
    train_dict_list = []
    test_dict_list = []
    train_row = None
    test_row = None

    with open(path_to_log) as f:

        for line in f:
            iteration_match = regex_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
                iteration_loss_match = regex_iteration_loss.search(line)
                if iteration_loss_match:
                    total_loss = float(iteration_loss_match.group(1))
            if iteration == -1:
                # Only start parsing for other stuff if we've found the first
                # iteration
                continue

            learning_rate_match = regex_learning_rate.search(line)
            if learning_rate_match:
                learning_rate = float(learning_rate_match.group(1))

            train_dict_list, train_row = parse_line_for_net_output(
                regex_train_output, train_row, train_dict_list,
                line, iteration, total_loss, learning_rate
            )
            test_dict_list, test_row = parse_line_for_net_output(
                regex_test_output, test_row, test_dict_list,
                line, iteration, total_loss, learning_rate
            )

    fix_initial_nan_learning_rate(train_dict_list)
    fix_initial_nan_learning_rate(test_dict_list)

    return pd.DataFrame.from_dict(train_dict_list), pd.DataFrame.from_dict(test_dict_list)


def parse_line_for_net_output(regex_obj, row, row_dict_list,
                              line, iteration, total_loss, learning_rate):
    """Parse a single line for training or test output

    Returns a a tuple with (row_dict_list, row)
    row: may be either a new row or an augmented version of the current row
    row_dict_list: may be either the current row_dict_list or an augmented
    version of the current row_dict_list
    """

    output_match = regex_obj.search(line)
    if output_match:
        if not row or row['NumIters'] != iteration:
            # Push the last row and start a new one
            if row:
                # If we're on a new iteration, push the last row
                # This will probably only happen for the first row; otherwise
                # the full row checking logic below will push and clear full
                # rows
                row_dict_list.append(row)

            row = OrderedDict([
                ('NumIters', iteration),
                ('TotalLoss', total_loss),
                ('LearningRate', learning_rate)
            ])

        # output_num is not used; may be used in the future
        # output_num = output_match.group(1)
        output_name = output_match.group(2)
        output_val = output_match.group(3)
        row[output_name] = float(output_val)

    if row and len(row_dict_list) >= 1 and len(row) == len(row_dict_list[0]):
        # The row is full, based on the fact that it has the same number of
        # columns as the first row; append it to the list
        row_dict_list.append(row)
        row = None

    return row_dict_list, row


def fix_initial_nan_learning_rate(dict_list):
    """Correct initial value of learning rate

    Learning rate is normally not printed until after the initial test and
    training step, which means the initial testing and training rows have
    LearningRate = NaN. Fix this by copying over the LearningRate from the
    second row, if it exists.
    """

    if len(dict_list) > 1:
        dict_list[0]['LearningRate'] = dict_list[1]['LearningRate']


def parse_args():
    description = ('Parse a Caffe training log and plot them')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_path', help='Path to log file')
    parser.add_argument('-i', '--update_interval', default=2000, type=int, help='Update interval in ms. Use -ve values for Static graph')

    args = parser.parse_args()
    return args


def init_plots(logfile_path):
    train_df, test_df = parse_log(logfile_path)

    num_of_axes = 2 - train_df.empty - test_df.empty
    assert num_of_axes != 0 , "Both train_df and test_df are empty"

    fig, axes = plt.subplots(num_of_axes, 3, squeeze=False, figsize=(30, 7 * num_of_axes))
    row = 0
    if not train_df.empty:
        axes[row, 0].set_title('Training Loss')
        axes[row, 1].set_title('Training Accuracy')
        axes[row, 2].set_title('Training Error Metrics')
        row += 1
    
    if not test_df.empty:
        axes[row, 0].set_title('Testing Loss')
        axes[row, 1].set_title('Testing Accuracy')
        axes[row, 2].set_title('Testing Error Metrics')

    return fig, axes
    

def draw_plot(data_frame, fig, axes, row):
    all_column_names = list(data_frame.columns.values)
    outputs = [e for e in all_column_names if e not in ('NumIters', 'TotalLoss', 'LearningRate')]
    loss_outputs = [output for output in outputs if "loss" in output]
    acc_outputs = [output for output in outputs if any(x in output for x in ["acc", "iou"])]
    error_outputs = [output for output in outputs if any(x in output for x in ["error"])]

    # Plot loss outputs
    axes[row, 0].clear()
    axes[row, 0].plot(data_frame['NumIters'], data_frame['TotalLoss'], color="red", alpha=0.6, label="TotalLoss")
    # axes[0].plot(data_frame['NumIters'], data_frame['LearningRate'], color="blue", alpha=0.5, label="LearningRate")
    for output in loss_outputs:
        axes[row, 0].plot(data_frame['NumIters'], data_frame[output], alpha=0.5, label=output)
    axes[row, 0].legend(loc='upper right')

    # Plot accuracy outputs
    if acc_outputs:
        axes[row, 1].clear()
        for output in acc_outputs:
            axes[row, 1].plot(data_frame['NumIters'], data_frame[output], alpha=0.5, label=output)
        axes[row, 1].axhline(1.0, color='b', linestyle='dashed', linewidth=2)
        axes[row, 1].legend(loc='lower right')
    
    # Plot error metrics outputs
    if error_outputs:
        axes[row, 2].clear()
        for output in error_outputs:
            axes[row, 2].plot(data_frame['NumIters'], data_frame[output], alpha=0.5, label=output)
        axes[row, 2].legend(loc='upper right')

    current_iters = int(data_frame['NumIters'].iloc[-1])
    fig.suptitle('Stats after Iteration# {}'.format(current_iters), fontsize=14)
    

def update_plots(frame, logfile_path, fig, axes):
    train_df, test_df = parse_log(logfile_path)

    row = 0
    if not train_df.empty:
        draw_plot(train_df, fig, axes, row)
        axes[row, 0].set_title('Training Loss')
        axes[row, 1].set_title('Training Accuracy')
        axes[row, 2].set_title('Training Error Metrics')
        row += 1
        
    if not test_df.empty:
        draw_plot(test_df, fig, axes, row)
        axes[row, 0].set_title('Testing Loss')
        axes[row, 1].set_title('Testing Accuracy')
        axes[row, 2].set_title('Testing Error Metrics')

def main():
    args = parse_args()

    fig, axes = init_plots(args.logfile_path)
    update_plots(0, args.logfile_path, fig, axes)

    if args.update_interval > 0:
         ani = FuncAnimation(fig, update_plots, fargs=(args.logfile_path, fig, axes), interval = args.update_interval)
    
    plt.show()


if __name__ == '__main__':
    main()

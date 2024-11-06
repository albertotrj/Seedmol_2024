#!/usr/bin/env python

import argparse
import numpy as np
from matplotlib import pyplot as plt
#from lmfit.models import GaussianModel


def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--last',    type=int, default=100)
    parser.add_argument('--predict', type=int, default=0)
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--virial', action='store_true')
    parser.add_argument('--lr',     action='store_true')
    parser.add_argument('--no_energy', action='store_false')
    parser.add_argument('--no_force',  action='store_false')
    parser.add_argument('--no_angle',  action='store_false')
    parser.add_argument('--no_norm',  action='store_false')
    parser.add_argument('--no_drift',  action='store_false')
    parser.add_argument('--epoch',  type=int)
    parser.add_argument('--nepoch',  type=int)
    parser.add_argument('--auto_epoch', action='store_true')
    parser.add_argument('--average',  type=int)
    parser.add_argument('--file', '-f',  type=str, default='lcurve.out')
    parser.add_argument('--logx', action='store_true')
    parser.add_argument('--e_target', type=float, default=1e-3)
    parser.add_argument('--f_target', type=float, default=4e-2)
    parser.add_argument('--v_target', type=float, default=1e-1)
    parser.add_argument('--d_target', type=float, default=1e-3)
    parser.add_argument('--a_target', type=float, default=1e0)
    parser.add_argument('--n_target', type=float, default=4e-2)
    args = parser.parse_args()
    return args


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def main():
    
    labels = ['batch', 'loss', 'energy', 'force', 'angle', 'norm', 'drift', 'virial', 'lr']
    labels_1D = ['batch', 'lr']
    label_tags = {'batch' : 'step',
                  'loss'  : 'rmse_val',
                  'energy': 'rmse_e_val',
                  'force' : 'rmse_f_val',
                  'virial': 'rmse_v_val',
                  'angle' : 'l2_a_tst', 
                  'norm'  : 'l2_n_tst', 
                  'drift' : 'rmse_fd_val', 
                  'lr'    : 'lr'}
    
    # defaults
    plot = {'batch' : True, 
            'loss'  : False,
            'energy': True,
            'force' : True,
            'angle' : False,
            'norm'  : False,
            'drift' : False,
            'lr'    : False}
    
    args = read_arguments()
    percent_steps_to_plot, predict, linear_plot = (args.last,
                                                   args.predict,
                                                   args.linear)
    plot['energy'], \
    plot['force'],  \
    plot['virial'], \
    plot['angle'],  \
    plot['norm'],   \
    plot['drift'],  \
    plot['lr'] = (args.no_energy,
                  args.no_force,
                  args.virial, 
                  args.no_angle,
                  args.no_norm,
                  args.no_drift,
                  args.lr)
    epoch = args.epoch
    n_epoch = args.nepoch
    auto_epoch = args.auto_epoch
    n_average  = args.average
    log_x = args.logx
    e_target = args.e_target
    f_target = args.f_target
    v_target = args.v_target
    d_target = args.d_target
    a_target = args.a_target
    n_target = args.n_target
    
    # read from file
    file_data = args.file
    with open(file_data, 'r') as f:
        tags = f.readline().strip('#').split()
    data = {}
    for label in labels:
        print(label, plot[label])
        if plot[label] is True:
            if label_tags[label] in tags:
                label_idx = tags.index(label_tags[label])
                if label in labels_1D:
                    data[label] = {'idx': [label_idx], 'vals': None}
                else:
                    data[label] = {'idx': [label_idx+1,label_idx], 'train': None, 'test': None}
                print('data[%s][idx] = ' % label, data[label]['idx'])
            else:
                plot[label] == False

    columns = []
    for col in data.values():
        columns.extend(col['idx'])
    print('Reading columns', columns)
    
    data_values = np.loadtxt(file_data, usecols=tuple(columns)).T
    

    for key in data.keys():
        if key in ['batch', 'lr']:
            data[key]['vals'] = data_values[columns.index(data[key]['idx'][0])]
            if key == 'batch':
                if epoch is not None:
                    data[key]['vals'] /= epoch
                elif auto_epoch:
                    data[key]['vals'] = np.arange(len(data[key]['vals']))
                elif n_epoch is not None:
                    data[key]['vals'] = np.arange(len(data[key]['vals'])) / n_epoch
        else:
            data[key]['train'] = data_values[columns.index(data[key]['idx'][0])]
            data[key]['test']  = data_values[columns.index(data[key]['idx'][1])]
    
    if percent_steps_to_plot == 100:
        b_min = 0
    else:
        b_min = int((100-percent_steps_to_plot)/100.0 * len(data['batch']['vals']))

    # make figure
    n_figs = len(data.keys()) - 1
    print(data.keys())
    plt.style.use('seaborn-deep')
    rc_params = {'font.size': 12,}
    plt.rcParams.update(rc_params)
    fig, axes = plt.subplots(n_figs, 1, figsize=(16,8), sharex=True)
    fig.subplots_adjust( left=0.07, bottom=0.07, right=0.99, top=0.99, hspace=0.1 )
    
    plot_labels = {'energy': 'Energy error (eV)',
                   'force' : 'Force error (eV/$\AA$)',
                   'drift' : 'Force drift (eV/$\AA$)',
                   'angle' : 'Force angle error ($\degree$)',
                   'norm'  : '$||F||$ error (eV/$\AA$)',
                   'virial': 'Virial error',
                   'lr'    : 'Learning rate'}
    
    # graph descriptor
    graph = []
    for key in data.keys():
        if key != 'batch':
            if key == 'lr':
                graph.append({'name' : key,
                              'label': plot_labels[key],
                              'vals' : data[key]['vals']})
            else:
                graph.append({'name' : key,
                              'label': plot_labels[key],
                              'train': data[key]['train'],
                              'test' : data[key]['test']})
    
    # linestyles
    line = {'type' : {'train': 'r-', 'test' : 'b-', 'vals': 'r-'},
            'alpha': {'train': 1.0,  'test' : 0.6,  'vals': 1.0},
            'color': {'train': 'b',  'test' : 'r',  'vals': 'k'}}
    
    
    # targets
    chem_accu = 43.348e-3 # eV/atom = 1 kcal/mol
    accuracy_target = {'energy': e_target,
                       'force' : f_target,
                       'virial': v_target,
                       'drift' : d_target,
                       'angle' : a_target,
                       'norm'  : n_target,
                       'lr'    : None}
    
    batch = data['batch']['vals']
    
    # plot
    for ax, g in zip(axes, graph):
        datasets = ['vals'] if g['name'] == 'lr' else ['train', 'test']
        for t in datasets:
            x = batch[b_min:]
            #if log_x and b_min == 0:
                #batch[0] = 1e-1    # shift 0 so it is plot
            y = g[t][b_min:]
            if n_average is not None:
                y_rolling = y[:n_average*(len(y)//n_average)].reshape((-1, n_average))
                y_std = np.std(y_rolling, axis=1)
                y = np.mean(y_rolling, axis=1)
                y_up = y + y_std
                y_low = y - y_std
                x = x[:n_average*(len(x)//n_average)].reshape((-1, n_average)).mean(axis=1)
                ax.fill_between(x, y_up,  y_low, color=line['color'][t], alpha=0.3*line['alpha'][t], lw=0)
            ax.plot(x, y, line['type'][t], label=t, color=line['color'][t], alpha=line['alpha'][t], linewidth=1.5)
        ax.set_ylabel(g['label'])
        if not linear_plot:
            ax.set_yscale('log')
        if log_x:
            ax.set_xscale('log')
        if accuracy_target[g['name']] is not None:
            ax.axhline(y=accuracy_target[g['name']], linewidth=1, linestyle='--', color='k', alpha=0.5, label='target')
        ax.grid(which='major')
        ax.grid(which='minor', color='0.9')
        ax.legend()
    if epoch is None and auto_epoch is False and n_epoch is None:
        ax.set_xlabel('# batch')
    else:
        ax.set_xlabel('# epoch')
    
    if predict > 0:
        b_min = int((100-predict)/100.0 * len(batch))
        x_fit = batch[b_min:]
        n_fit = len(batch[b_min:])
        x_plot = np.linspace(x_fit[0], min(x_fit[-1] + (x_fit[-1] - x_fit[0]), 1e6))
        for ax, g in zip(axes, graph):
            fit = np.polyfit(x_fit, g['test'][b_min:], 1)
            ax.plot(x_plot, np.poly1d(fit)(x_plot), 'k--', linewidth=1.5, label='$\delta y_{%i} = %.1e$' % (n_fit, fit[0] * n_fit))
            ax.legend()
            #fit, components = fit_errors(y, x_fit, x)
    #axis.plot(x, fit, 'k-', linewidth=2, label='fit', alpha=0.8)
    
    #plt.tight_layout()
    plt.show()
    fig.savefig('training.png', dpi=300)


if __name__ == '__main__':
    main()

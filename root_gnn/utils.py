import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os

def read_log(file_name):
    time_format = '%d %b %Y %H:%M:%S'
    get2nd = lambda x: x.split()[1]

    time_info = []
    data_info = []
    itime = -1
    with open(file_name) as f:
        for line in f:
            if line[0] != '#':
                tt = time.strptime(line[:-1], time_format)
                time_info.append(tt)
                data_info.append([])
                itime += 1
            else:
                items = line.split(',')
                try:
                    iteration = int(get2nd(items[0]))
                except ValueError:
                    continue
                time_consumption = float(get2nd(items[1]))
                loss_train = float(get2nd(items[2]))
                loss_test  = float(get2nd(items[3]))
                precision  = float(get2nd(items[4]))
                recall     = float(get2nd(items[5]))
                data_info[itime].append([iteration, time_consumption, loss_train,
                                      loss_test, precision, recall])
    return data_info, time_info


def plot_log(info, name, axs=None):
    fontsize = 16
    minor_size = 14
    if type(info) is not 'numpy.ndarray':
        info = np.array(info)
    df = pd.DataFrame(info, columns=['iteration', 'time', 'loss_train', 'loss_test', 'precision', 'recall'])

    # make plots
    if axs is None:
        _, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        axs = axs.flatten()

    y_labels = ['Time [s]', 'Training Loss', 'Precision', 'Recall']
    y_data   = ['time', 'loss_train', 'precision', 'recall']
    x_label = 'Iterations'
    x_data = 'iteration'
    for ib, values in enumerate(zip(y_data, y_labels)):
        ax = axs[ib]

        if 'loss_train' == values[0]:
            df.plot(x=x_data, y=values[0], ax=ax, label='Training')
            df.plot(x=x_data, y='loss_test', ax=ax, label='Testing')
            ax.set_ylabel("Losses", fontsize=fontsize)
            ax.legend(fontsize=fontsize)
        else:
            df.plot(x=x_data, y=values[0], ax=ax)
            ax.set_ylabel(values[1], fontsize=fontsize)

        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)

    return axs


class IndexMgr:
    def __init__(self, n_total, training_frac=0.8):
        self.max_tr = int(n_total*training_frac)
        self.total = n_total
        self.n_test = n_total - self.max_tr
        self.tr_idx = 0
        self.te_idx = self.max_tr

    def next(self, is_training=False):
        if is_training:
            self.tr_idx += 1
            if self.tr_idx > self.max_tr:
                self.tr_idx = 0
            return self.tr_idx
        else:
            self.te_idx += 1
            if self.te_idx > self.total:
                self.te_idx = self.max_tr
            return self.te_idx


def load_yaml(file_name):
    find_file = False
    if not os.path.exists(file_name):
        import pkg_resources
        try:
            file_name = pkg_resources.resource_filename('root_gnn', os.path.join('configs', file_name))
        except:
            pass
        finally:
            find_file = True
    else:
        find_file = True

    if not find_file:
        raise FileNotFoundError(file_name,"missing")
    
    with open(file_name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    if dphi > np.pi:
        dphi -= 2*np.pi
    if dphi < -np.pi:
        dphi += 2*np.pi
    return dphi
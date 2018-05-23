import numpy as np
import matplotlib.pyplot as plt
import csv

DPI = 600
NUM_LAYERS = 31

def read_data(fname):
    data = list(csv.reader(open(fname)))
    data = data[1:]  # ignore header

    median = [float(row[1]) for row in data]
    perc5 = [float(row[2]) for row in data]
    perc95 = [float(row[3]) for row in data]

    return median, perc5, perc95

def makeplot(plt, median, perc5, perc95, label=None, mark='-', color=None):
    x = range(len(median))
    
    plt.plot(x, median, mark, label=label, c=color)
    plt.fill_between(x, perc5, perc95, facecolor=color, alpha=0.1)

def pre_plot(plt):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(14, adjustable='box-forced')
    return ax

def setup_plot(plt, xlabel, ylabel):
    axes = plt.gca()
    axes.set_xlim(0, NUM_LAYERS)
    axes.set_ylim(0, 1)
    plt.xticks(range(0, NUM_LAYERS + 1), 
               range(0, NUM_LAYERS + 1),
               # [(str(i) if i%2 == 1 else '') for i in range(0, NUM_LAYERS + 1)],
               rotation=90)
    plt.xlabel(xlabel, fontsize=14, labelpad = 10)
    plt.ylabel(ylabel, fontsize=14, labelpad = 10)

def cosine_fgsm():
    median, perc5, perc95 = read_data('cosine_fgsm.csv')
    fig = pre_plot(plt)
    makeplot(fig, median, perc5, perc95, 
            label='Adversarial Samples', mark='-', color='r')
    
    median, perc5, perc95 = read_data('cosine_norm.csv')

    makeplot(fig, median, perc5, perc95, 
             label='Normal Samples', mark='-', color='b')

    setup_plot(plt, 'Activation Layer', 'Cosine Distance')
    plt.savefig('fgsm_squeezed.png', dpi=DPI)

def cosine_cw2ll():
    median, perc5, perc95 = read_data('cosine_cw2ll.csv')
    fig = pre_plot(plt)
    makeplot(fig, median, perc5, perc95, 
            label='Adversarial Samples', mark='-', color='r')
    
    median, perc5, perc95 = read_data('cosine_norm.csv')
    makeplot(fig, median, perc5, perc95, 
            label='Normal Samples', mark='-', color='b')

    # plt.legend(loc='upper left')
    setup_plot(plt, 'Activation Layer', 'Cosine Distance')
    plt.savefig('cw2ll_squeezed.png', dpi=DPI)

def cifar_layer():
    median, perc5, perc95 = read_data('cifar_layer_dist_cwinfll.csv')
    fig = pre_plot(plt)
    makeplot(fig, median, perc5, perc95, 
            label='Adversarial Samples', mark='-', color='r')
    
    median, perc5, perc95 = read_data('cifar_layer_dist_rand_cwinfll.csv')
    makeplot(plt, median, perc5, perc95, 
            label='Randomly perturbed Samples', mark='-', color='b')

    # plt.legend(loc='upper left')
    setup_plot(plt, 'Activation Layer', 'Cosine Distance')
    plt.savefig('adv_rand.png', dpi=DPI)

def adv_layer():
    median, perc5, perc95 = read_data('cifar_layer_dist_cwinfll.csv')
    fig = pre_plot(plt)
    makeplot(fig, median, perc5, perc95, 
            label='Adversarial Samples', mark='-', color='r')
    
    # median, perc5, perc95 = read_data('cifar_layer_dist_rand.csv')
    #makeplot(plt, median, perc5, perc95, 
    #         label='Randomly perturbed Samples', mark='-', color='b')

    # plt.legend(loc='upper left')
    setup_plot(plt, 'Activation Layer', 'Cosine Distance')
    plt.savefig('adv_layers.png', dpi=DPI)

if __name__ == '__main__':
    # plt.rc('font',family='DejaVu Sans')
    # plt.rc('font',family='Fira Sans')
    plt.rc('font', family = 'Trebuchet MS')
    adv_layer()
    plt.clf()
    cosine_fgsm()
    plt.clf()
    cosine_cw2ll()
    plt.clf()
    cifar_layer()

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
    plt.rc('font',family='DejaVu Sans')
    #plt.axes.Axes.set_aspect(16/9)
    x = range(len(median))
    
    plt.plot(x, median, mark, label=label, c=color)
    plt.fill_between(x, perc5, perc95, facecolor=color, alpha=0.1)

def cosine_fgsm():
    median, perc5, perc95 = read_data('cosine_fgsm.csv')
    makeplot(plt, median, perc5, perc95, 
            label='Adversarial Samples', mark='-', color='r')
    
    median, perc5, perc95 = read_data('cosine_norm.csv')
    makeplot(plt, median, perc5, perc95, 
            label='Normal Samples', mark='-', color='b')

    # plt.legend(loc='upper left') - No legend
    axes = plt.gca()
    axes.set_xlim(0, NUM_LAYERS)
    axes.set_ylim(0, 1)
    plt.xlabel('Activation Layer')
    plt.ylabel('Cosine Distance', # between squeezed and non-squeezed',
                fontsize=10)
    plt.savefig('fgsm_squeezed.png', dpi=DPI)

def cosine_cw2ll():
    median, perc5, perc95 = read_data('cosine_cw2ll.csv')
    makeplot(plt, median, perc5, perc95, 
            label='Adversarial Samples', mark='-', color='r')
    
    median, perc5, perc95 = read_data('cosine_norm.csv')
    makeplot(plt, median, perc5, perc95, 
            label='Normal Samples', mark='-', color='b')

    # plt.legend(loc='upper left')
    axes = plt.gca()
    axes.set_xlim(0)
    axes.set_ylim(0,1)
    axes.set_xlim(0, NUM_LAYERS)
    plt.xlabel('Activation Layer')
    plt.ylabel('Cosine Distance', # between squeezed and non-squeezed',
                fontsize=10)
    plt.savefig('cw2ll_squeezed.png', dpi=DPI)

def cifar_layer():
    median, perc5, perc95 = read_data('cifar_layer_dist_adv.csv')
    makeplot(plt, median, perc5, perc95, 
            label='Adversarial Samples', mark='-', color='r')
    
    median, perc5, perc95 = read_data('cifar_layer_dist_rand.csv')
    makeplot(plt, median, perc5, perc95, 
            label='Randomly perturbed Samples', mark='-', color='b')

    # plt.legend(loc='upper left')
    axes = plt.gca()
    axes.set_xlim(0)
    axes.set_ylim(0,1)
    axes.set_xlim(0, NUM_LAYERS)
    plt.xlabel('Activation Layer')
    plt.ylabel('Cosine Distance', # between original and perturbed',
                fontsize=10)
    plt.savefig('adv_rand.png', dpi=DPI)

if __name__ == '__main__':
    cosine_fgsm()
    plt.clf()
    cosine_cw2ll()
    plt.clf()
    cifar_layer()

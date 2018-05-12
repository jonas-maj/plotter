import numpy as np
import matplotlib.pyplot as plt
import csv

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

def main():
    median, perc5, perc95 = read_data('cosine_fgsm.csv')
    makeplot(plt, median, perc5, perc95, 
            label='Adversarial Samples', mark='-', color='r')
    
    median, perc5, perc95 = read_data('cosine_norm.csv')
    makeplot(plt, median, perc5, perc95, 
            label='Normal Samples', mark='-', color='b')

    plt.legend(loc='upper left')
    axes = plt.gca()
    axes.set_xlim(0)
    axes.set_ylim(0,1)
    plt.xlabel('Layer #')
    plt.ylabel('Cosine distance between squeezed and non-squeezed',
                fontsize=10)
    plt.savefig('fgsm_squeezed.png', dpi=288)

if __name__ == '__main__':
    main()

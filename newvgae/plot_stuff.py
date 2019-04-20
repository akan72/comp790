import pickle as pkl
from utils import plot_results
import numpy as np

# plot_results(pkl.load(open('CORA_RESULTS.p', 'rb')), path='../figures/geometric/CORA_RESULTS.png')
my_data = np.genfromtxt('test.edgelist', delimiter=' ')
my_data = my_data[:,:2]


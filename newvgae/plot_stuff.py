import pickle as pkl
from utils import plot_results

plot_results(pkl.load(open('CORA_RESULTS.p', 'rb')), path='../figures/geometric/CORA_RESULTS.png')

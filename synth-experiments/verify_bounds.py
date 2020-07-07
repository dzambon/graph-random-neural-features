import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
colors = cm.get_cmap("Set1").colors

from config import SEED, NUM_HID_FEAT
from grnf import GRNFLayer

import torch
from torch_geometric.data import Batch
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    
    M_list = [2**i for i in range(5, 17)]  # embedding dimensions
    M_star = max(M_list+[1e6])
    eps_ratio = .25
    repeat = 500  # number of repetitions
    no_M = len(M_list)
    
    batch = torch.load("./data_synth/sbm_2_graphs.pt")

    nf_dim = batch.num_node_features
    ef_dim = batch.num_edge_features
    grnf = GRNFLayer(in_node_features=nf_dim, in_edge_features=ef_dim,
                     hidden_features=NUM_HID_FEAT, out_features=int(M_star))
    grnf.to('cpu')
    z = torch.tanh(grnf(batch, verbose=True)).data.numpy()

    # Show features correlation
    fig = plt.figure()
    #plt.scatter(z[0], z[1])
    plt.hexbin(z[0], z[1], gridsize=50, bins='log')
    cb = plt.colorbar()
    cb.set_label('log frequency')
    plt.xlabel('$\psi(g_1,w)$')
    plt.ylabel('$\psi(g_2,w)$')
    fig.savefig('correlation.pdf')

    # Precompute some operations
    z_dist_sq = (z[0] - z[1]) ** 2
    dsq_target = np.mean(z_dist_sq)
    dsq_std = np.std(z_dist_sq)
    eps = np.abs(eps_ratio * dsq_target)

    C_G = (z**4).mean(axis=1).max()  # max_g E_w[psi(g,w)^4]
    C_var = z_dist_sq.var()
    print('dist 16C:', 16*C_G, 'C_var:', C_var)

    delta_M = - np.ones(no_M)
    delta_M_hat = - np.ones(no_M)
    delta_star = - np.ones(no_M)
    delta_star_hat = - np.ones(no_M)
    delta_clt = - np.ones(no_M)

    for i in tqdm(range(no_M), desc="testing M"):
        dsq_M = get_randomize_estimate(M=M_list[i], z_paired=z_dist_sq, repeat=repeat)
        
        # let X = |z(g_1)-z(g_2)|^2 and mu = d(g_1,g_2)^2 \approx |z_*(g_1)- z_*(g_2)|^2
        # we use C_d_var is a tighter condition wrt 16 C_g
        # delta_M : P(|X-X'|>=eps) <= 8 * C_d_var / eps**2 / M <= 128 * C_G / eps**2 / M
        delta_M[i] = C_var * 8 / eps**2 / M_list[i]
        delta_M_hat[i] = (np.abs(dsq_M.reshape(-1, 1) - dsq_M.reshape(1, -1)) > eps).mean()

        # delta_star : P(|X-mu|>=eps) <= C_d_var / eps**2 / M <= 16 * C_G / eps**2 / M
        delta_star[i] = C_var / (eps**2) / M_list[i]
        delta_star_hat[i] = (np.abs(dsq_M - dsq_target) > eps).mean()

        # delta_clt : P(|X-mu|>=eps) = P(|X-\mu|/sigma > eps/sigma) \approx 2*Phi(-eps/sigma) = delta_clt
        delta_clt[i] = 2*norm.cdf(- eps / dsq_std * np.sqrt(M_list[i]))


    fig = plt.figure(figsize=(5, 3))
    plt.subplots_adjust(left=.15, bottom=.22, right=.98)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    up_bound = 1
    low_power = 10
    low_bound = 10**(-low_power)

    plt.loglog(M_list, np.clip(delta_star, low_bound, up_bound), color=colors[0], marker='o', label=r'$\delta_*$')
    plt.loglog(M_list, np.clip(delta_M, low_bound, up_bound),    color=colors[0], marker='+', label=r'$\delta_{M}$')
    plt.loglog(M_list, np.clip(delta_clt, low_bound, up_bound),  color='k', label=r'$\delta_{clt}$')
    plt.loglog(M_list, np.clip(delta_star_hat, low_bound, up_bound), color=colors[1], marker='o', linestyle="dashed", label=r'$\hat\delta_*$')
    plt.loglog(M_list, np.clip(delta_M_hat, low_bound, up_bound),    color=colors[1], marker='+', linestyle="dashed", label=r'$\hat\delta_{M}$')

    pows = list(range(0,low_power,3))+[low_power]
    ticks = [10**(-p) for p in pows]
    labs = ['$10^{{{}}}$'.format(-p) for p in pows]
    labs[-1] = '$\le10^{{{}}}$'.format(-low_power)
    plt.yticks(ticks, labs)
    plt.title('$\\varepsilon$={:.2f}\\%'.format(100*eps_ratio))
    plt.legend()
    plt.xlabel('Embedding dimension $M$')
    
    fig.savefig('bounds.pdf')
    
def get_randomize_estimate(M, z_paired, repeat):
    subsets = np.random.choice(z_paired.shape[0], size=(repeat, M))
    val = np.empty(repeat)
    for i, s in enumerate(subsets):
        val[i] = np.sum(z_paired[s])
    return val / M # z_paired needs to be non normalized so that I can normalize here


if __name__ == "__main__":
    main()

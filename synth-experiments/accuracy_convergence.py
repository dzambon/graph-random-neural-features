import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
colors = cm.get_cmap("Set1").colors
colors2 = cm.get_cmap("Dark2").colors

from config import SEED, N_ANNEAL, NUM_HID_FEAT
from grnf import GRNFLayer

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors, manifold
import torch
from torch_geometric.data import Batch

torch.manual_seed(SEED)
np.random.seed(SEED)

def main(dataset='del'):

    # Parameters
    test_ratio = .5
    M_list = np.logspace(1, np.log2(1e3), num=20, endpoint=True, base=2)
    M_list = sorted(list(set(M_list.astype(int))))
    M_star = max(M_list + [int(1e4)])
    M_max = M_star * 2
    no_annealing = N_ANNEAL
    
    # Load data setdel
    if dataset == 'sbm':
        batch = torch.load("./data_synth/sbm_graphs.pt")
        y = np.load("./data_synth/sbm_labels.npy")
        name = "SBM"
    else:
        batch = torch.load("./data_synth/delaunay_graphs.pt")
        y = np.load("./data_synth/delaunay_labels.npy")
        name = 'Del'
    
    # Embed data set

    nf_dim = batch.num_node_features
    ef_dim = batch.num_edge_features
    grnf = GRNFLayer(in_node_features=nf_dim, in_edge_features=ef_dim,
                     hidden_features=NUM_HID_FEAT, out_features=M_max)
    grnf.to('cpu')
    z = torch.tanh(grnf(batch, verbose=True)) - torch.tanh(grnf.get_zerograph_representation())

    # Visualize dataset
    print("creating t-SNE")
    z_red = manifold.TSNE(n_components=2).fit_transform(z)
    fig = plt.figure(0, figsize=(3.5, 3.5))
    plt.subplots_adjust(left=.14, bottom=.15, right=.98, top=.88)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    styles = {"fillstyle":'none', "mew":8, "ms":1,  "linestyle":'none'}
    plt.plot(z_red[np.where(y == 0), 0][0], z_red[np.where(y == 0), 1][0], color=colors2[4], marker='o', label="Class 0", **styles)
    plt.plot(z_red[np.where(y != 0), 0][0], z_red[np.where(y != 0), 1][0], color=colors2[5], marker='+', label="Class 1", **styles)
    
    plt.title('t-SNE {}'.format(name))
    plt.legend()
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    fig.savefig('tSNE_{}.pdf'.format(name))
    plt.show()
    
    # Classification
    print('training M_star')
    svm_M = np.zeros((len(M_list) + 1, no_annealing))
    knn_M = np.zeros((len(M_list) + 1, no_annealing))
    
    for ann in range(no_annealing):
        tr, te, y_train, y_test = train_test_split(np.arange(y.shape[0]), y, test_size=test_ratio)
        
        for i, M in tqdm(enumerate(M_list + [M_star]), desc=f"{ann}/{no_annealing}"):
            print('training M{}'.format(M))
            perm = np.random.permutation(M_star)[:M]
            
            z_tr = z[tr][:, perm]
            z_te = z[te][:, perm]
            
            # SVM
            svm_M[i, ann] = get_accuracy(msg='SVM M{}:'.format(M),
                                         clf=svm.SVC(kernel='linear'),
                                         z_train=z_tr, y_train=y_train,
                                         z_test=z_te, y_test=y_test)
            # k-NN
            knn_M[i, ann] = get_accuracy(msg='k-NN M{}:'.format(M),
                                         clf=neighbors.KNeighborsClassifier(),
                                         z_train=z_tr, y_train=y_train,
                                         z_test=z_te, y_test=y_test)
            
    svm_res = arrange_acc(svm_M)
    knn_res = arrange_acc(knn_M)
    
    fig = plt.figure(1, figsize=(5, 3))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.subplots_adjust(left=.11, bottom=.15, right=.98, top=.9)
    ax = fig.gca()
    ax.set_xscale('log')
    
    def draw_entry(acc, col, lstyle, msg):
        plt.fill_between(M_list, acc['mean'] - acc['std'], acc['mean'] + acc['std'], color=col, edgecolor=(1., 1., 1.),
                         alpha=.2)
        ax.plot(M_list, acc['mean'],
                linestyle=lstyle, color=col, marker='+', label=msg[1])
        ax.plot(M_list, [acc['star']] * len(M_list),
                linestyle=lstyle, color=col, label=msg[0])
    
    lstyle = "dashed"
    col = colors[0]
    draw_entry(acc=svm_res, col=col, lstyle=lstyle,
               msg=['SVM with $k_P(g_1,g_2)$',
                    'SVM with $\\tilde{k}_P(g_1,g_2)$'])

    lstyle = "dotted"
    col = colors[1]
    draw_entry(acc=knn_res, col=col, lstyle=lstyle,
               msg=['k-NN with $d_P(g_1,g_2)$',
                    'k-NN with $|z(g_1) -z(g_2)|$'])
    
    plt.ylim(None, 1)
    plt.legend(loc="lower right")
    plt.xlabel('Embedding dimension $M$')
    plt.ylabel('Accuracy')
    
    plt.title('{} ($M_*={:d}$)'.format(name, int(M_star)))
    fig.savefig('classification_{}.pdf'.format(name))


def enc_labels(y):
    if y.max()<1e-10:
        return y
    return (y / y.max()).ravel()

def get_accuracy(clf, z_train, y_train, z_test, y_test, msg='acc'):
    clf.fit(z_train, y_train)
    y_pred = clf.predict(z_test)
    acc = np.mean(enc_labels(y_pred) == enc_labels(y_test))
    print(msg, acc)
    return acc

def arrange_acc(acc_M):
    acc_mean = np.mean(acc_M, axis=1)
    acc_std = np.std(acc_M, axis=1)
    acc_star = acc_mean[-1]
    acc_star_std = acc_std[-1]
    acc_mean = acc_mean[:-1]
    acc_std = acc_std[:-1]
    acc = {'mean': acc_mean,
           'std': acc_std,
           'star': acc_star,
           'star_std': acc_star_std}
    return acc


if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        main()
    else:
        main(sys.argv[1])

import argparse
import os
import math
import numpy as np
import NTK
import tools
from itertools import combinations, product

parser = argparse.ArgumentParser()
parser.add_argument('-dir', default = "Experiment_UCI\data", type = str, help = "data directory")
parser.add_argument('-file', default = "Experiment_UCI\uci_result.log", type = str, help = "Output File")
parser.add_argument('-max_tot', default = 5000, type = int, help = "Maximum number of data samples")
parser.add_argument('-max_dep', default = 5, type = int, help = "Maximum number of depth")

args = parser.parse_args()
MAX_N_TOT = args.max_tot
MAX_DEP = args.max_dep

DEP_LIST = [MAX_DEP - 1]
C_LIST = [0.1]
a = [0, 0.125, 0.25, 0.5, 1] # WNTK Case
# a = [1] # NTK Case

datadir = args.dir
alg = tools.ridge_regression
outf = open(args.file, "w")
print ("Dataset\tValidation Acc\tTest Acc", file = outf)

for idx, dataset in enumerate(sorted(os.listdir(datadir))):
    if not os.path.isdir(datadir + "/" + dataset):
        continue
    if not os.path.isfile(datadir + "/" + dataset + "/" + dataset + ".txt"):
        continue
    dic = dict()
    for k, v in map(lambda x : x.split(), open(datadir + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
        dic[k] = v
    
    c = int(dic["n_clases="])
    d = int(dic["n_entradas="])
    n_train = int(dic["n_patrons_entrena="])
    n_val = int(dic["n_patrons_valida="])
    n_train_val = int(dic["n_patrons1="])
    n_test = 0
    if "n_patrons2=" in dic:
        n_test = int(dic["n_patrons2="])
    n_tot = n_train_val + n_test
    
    if n_tot > MAX_N_TOT or n_test > 0:
        print (str(dataset) + '\t0\t0', file = outf)
        continue
    
    print (idx, dataset, "\tN:", n_tot, "\td:", d, "\tc:", c)
    
    # load data
    f = open("Experiment_UCI/data/" + dataset + "/" + dic["fich1="], "r").readlines()[1:]
    X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
    y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))
    
    # load training and validation set
    fold = list(map(lambda x: list(map(int, x.split())), open(datadir + "/" + dataset + "/" + "conxuntos.dat", "r").readlines()))
    train_fold, val_fold = fold[0], fold[1]
    best_acc = 0.0
    best_value = 0
    best_dep = 0
    best_ker = 0

    # calculate NTK
    weights = list(product(a, repeat = MAX_DEP))
    count = 0

    # Prepare Work
    S_array = []
    Sn_array = []
    S = np.matmul(X, X.T)
    S_array.append(S)
    for dep in range(MAX_DEP):
        L = np.diag(S)
        P = np.clip(np.sqrt(np.outer(L, L)), a_min = 1e-9, a_max = None)
        Sn = np.clip(S / P, a_min = -1, a_max = 1)
        Sn_array.append(Sn)
        S = (Sn * (math.pi - np.arccos(Sn)) + np.sqrt(1.0 - Sn * Sn)) * P / 2.0 / math.pi
        S_array.append(S)

    
    for weight in weights:
        print(count, len(weights), weight)
        count += 1
        Ks = NTK.fast_kernel_value_batch(S_array, Sn_array, X, MAX_DEP, weight)
        # enumerate kenerls and cost values to find the best hyperparameters
        for dep in DEP_LIST:
            K = Ks[dep][0]
            for value in C_LIST:
                acc = alg(K[train_fold][:, train_fold], K[val_fold][:, train_fold], y[train_fold], y[val_fold], value, c)
                print(acc)
                if acc > best_acc:
                    best_acc = acc
                    best_value = value
                    best_dep = dep
                    best_weight = weight
                    best_ker = K

    K = best_ker
    print ("best acc (to find the best hyperparameter):", best_acc, "\tC:", best_value, "\tdep:", best_dep + 1, "\tweight:", best_weight)
    
    # 4-fold cross-validating
    avg_acc = 0.0
    fold = list(map(lambda x: list(map(int, x.split())), open("Experiment_UCI/data/" + dataset + "/" + "conxuntos_kfold.dat", "r").readlines()))
    for repeat in range(4):
        train_fold, test_fold = fold[repeat * 2], fold[repeat * 2 + 1]
        acc = alg(K[train_fold][:, train_fold], K[test_fold][:, train_fold], y[train_fold], y[test_fold], best_value, c)
        avg_acc += 0.25 * acc 
    print ("acc (cross validation):", avg_acc, "\n")
    print (str(dataset) + '\t' + str(best_acc * 100) + '\t' + str(avg_acc * 100) + "\tSize:" + str(X.shape[0]) + "\tClass:" + str(c) + "\tC:" + str(best_value) + "\tdep:" + str(best_dep + 1) + "\tweight:" + str(best_weight), file = outf)

outf.close()
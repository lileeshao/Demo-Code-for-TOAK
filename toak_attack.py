import json
import os
import argparse
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import random
import numpy as np
from vgae import VGAE
import numba
import scipy.sparse as sp
import time
import ot
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Attack UIL model")
    parser.add_argument('--dataset', default="douban")
    parser.add_argument('--ratio', default=0.1, type=float, help="attack ratio")

    #Rand Walk and EMD parameters
    parser.add_argument('--walks_per_node', default=1000, type=int, help="random walk numbers per node")
    parser.add_argument('--walk_len', default=5, type=int, help="random walk length")
    parser.add_argument('--lamda', type=float, default=1, help='lamda parameter')

    #VGAE parameters
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate used in VGAE.')
    parser.add_argument('--epochs', '-e', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--hidden1', '-h1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', '-h2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU id to use.')

    return parser.parse_args()

def load_id2idx(base_dir):
    id2idx_file = os.path.join(base_dir, 'id2idx.json')
    id2idx = {}
    id2idx = json.load(open(id2idx_file)) 
    for k, v in id2idx.items():
        id2idx[str(k)] = v
    return id2idx

def load_graph(base_dir):
    G_data = json.load(open(os.path.join(base_dir, "G.json")))
    G = json_graph.node_link_graph(G_data)
    id2idx = load_id2idx(base_dir)
    mapping = {k: int(id2idx[k]) for k in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return G

def load_H(dataset, id2idx):
    h_file = './dataset/'+str(dataset)+'/train'
    h = set()
    with open(h_file) as f:
        for line in f:
            _,anchor = line.strip().split()
            h.add(id2idx[str(anchor)])
    return h


def cal_EMD(D, weight, index):    
#Calculate accurate EMD
    p = weight/weight.sum()
    weight1 = weight.copy()
    weight1[index] = 0
    q = weight1/weight1.sum()   
    return ot.emd2(p,q,D)

def cal_approximate_EMD(emb, weight, index):    
#Calculate approximate EMD
    if weight.shape[0]==1:
        return 1
    p_g = weight/weight.sum()
    weight1 = weight.copy()
    weight1[index] = 0
    p_g_star = weight1/weight1.sum() 
    d_norm = np.linalg.norm(np.average(emb,axis=0, weights=p_g)-np.average(emb,axis=0, weights=p_g_star))
    return np.square(d_norm)


def construct_adjacency(G, id2idx):
    adjacency = np.zeros((len(G.nodes()), len(G.nodes())))
    for src_id, trg_id in G.edges():
        adjacency[src_id, trg_id] = 1
        adjacency[trg_id, src_id] = 1
    return adjacency

@numba.jit(nopython=True)
def random_walk(indptr, indices, walk_length, walks_per_node, seed=333):
    np.random.seed(seed)
    N = len(indptr) - 1
    walks = []

    for _ in range(walks_per_node):
        for n in range(N):
            if indptr[n]==indptr[n + 1]:
                continue
            one_walk = []
            for _ in range(walk_length+1):
                one_walk.append(n)
                n = np.random.choice(indices[indptr[n]:indptr[n + 1]])
            walks.append(one_walk)

    return walks

def calculate_score(graph, id2idx, z, anchor_set, args):
    edge_num = graph.number_of_edges()
    node_num = graph.number_of_nodes()
    candidate_num = int(edge_num*args.ratio*3)

    adj_matrix = sp.csr_matrix(construct_adjacency(graph,id2idx))
    walks_on_clean_graph = random_walk(adj_matrix.indptr,adj_matrix.indices,args.walk_len,args.walks_per_node)
    print('*'*20)
    print('Random walk process on clean ego networks has done!!!')
    print('*'*20)

    deg = graph.degree()
    endpoints1 = sorted([(k,deg[k]) for k in graph.nodes()], key=lambda x:x[1], reverse=True)[:30]
    endpoints2 = sorted([(k,deg[k]) for k in list(anchor_set)], key=lambda x:x[1], reverse=True)
    add_edge = []

    num=0
    for i in endpoints1:
        for j in endpoints2:
            if i!=j and not graph.has_edge(i[0],j[0]):
                graph.add_edge(i[0],j[0])
                add_edge.append((i[0],j[0]))
                num+=1
                if num==candidate_num:break
        if num==candidate_num:break

    adj_matrix = sp.csr_matrix(construct_adjacency(graph,id2idx))
    walks_on_poisoned = random_walk(adj_matrix.indptr,adj_matrix.indices,args.walk_len,args.walks_per_node)
    print('*'*20)
    print('Random walk process on candidate poisoned ego networks has done!!!')
    print('*'*20)

    adj = sp.triu(adj_matrix).tocoo()
    edge_num = graph.number_of_edges()
    edge_emb = np.zeros((edge_num, z.shape[1]*2),dtype=np.float32)
    e2eid = {}
    #edge embedding
    for i in range(edge_num):
        if adj.row[i] in anchor_set or adj.col[i] in anchor_set:
            factor = np.exp(args.lamda*1)
        else:
            factor = 1

        e2eid[(adj.row[i], adj.col[i])] = [i, factor]
        e_vector = np.hstack((z[adj.row[i]],z[adj.col[i]]))
        e_vector = e_vector/np.linalg.norm(e_vector)
        edge_emb[i] = e_vector

    print('*'*20)
    print('Calculating edge embedding has done!!!')
    print('*'*20)

    edge_score = np.zeros((edge_num,))
    
    rw_on_clean_ego = {i:{} for i in range(node_num)}
    total_rw = {v[0]:0 for v in e2eid.values()}
    for wk in walks_on_clean_graph:
        start_node = wk[0]
        for i in range(len(wk)-1):
            if wk[i]>wk[i+1]:
                e = e2eid[(wk[i+1], wk[i])]
            else:
                e = e2eid[(wk[i], wk[i+1])]

            total_rw[e[0]] += e[1]

            if e[0] in rw_on_clean_ego[start_node].keys():
                rw_on_clean_ego[start_node][e[0]] += e[1]
            else:
                rw_on_clean_ego[start_node][e[0]] = e[1]

    cnt = 0
    total_rw_list = [[k,v] for k,v in total_rw.items()]
    candidate_for_del = sorted(total_rw_list, key=lambda x:x[1],reverse=True)[:candidate_num]
    candidate_for_del = set([item[0] for item in candidate_for_del])
    candidate_for_add = set()
    for e in add_edge:
        if e[0]>e[1]:
            candidate_for_add.add(e2eid[(e[1],e[0])][0])
        else:
            candidate_for_add.add(e2eid[(e[0],e[1])][0])
    print(len(candidate_for_del), len(candidate_for_add))

    rw_on_poisoned_ego = {i:{} for i in range(node_num)}
    for wk in walks_on_poisoned:
        start_node = wk[0]
        for i in range(len(wk)-1):
            if wk[i]>wk[i+1]:
                e = e2eid[(wk[i+1], wk[i])]
            else:
                e = e2eid[(wk[i], wk[i+1])]

            if e[0] in rw_on_poisoned_ego[start_node].keys():
                rw_on_poisoned_ego[start_node][e[0]] += e[1]
            else:
                rw_on_poisoned_ego[start_node][e[0]] = e[1]

    all_num = 0
    for node in graph.nodes():
        e1 = rw_on_clean_ego[node]
        e2 = rw_on_poisoned_ego[node]
        num = len(e1)
        edge = []
        weight = []
        for e,w in e1.items():
            edge.append(e)
            weight.append(w)
        
        emb = edge_emb[edge]
        weight = np.asarray(weight)
        for index in range(num):
            if edge[index] in candidate_for_del:
                temp = np.exp(-np.abs(cal_approximate_EMD(emb, weight, index)))
                edge_score[edge[index]] += (temp-1)
                all_num+=1

        for ed,wed in e2.items():
            if ed in candidate_for_add:
                emb1 = np.vstack((emb,edge_emb[ed]))
                weight1 = np.append(weight,wed)
                temp = np.exp(-np.abs(cal_approximate_EMD(emb1, weight1, num)))
                edge_score[edge[index]] += (temp-1)
                all_num+=1

        cnt+=1
        if cnt%500==0:
            print(cnt, len(e2))
    print('*'*20)
    print('Calculating candidate edge score has done!!!')
    print('*'*20)
    print(all_num)
    print(np.max(edge_score),np.min(edge_score))
    return edge_score,e2eid

def generate_attack(edge_score, e2eid, id2idx, args, G, flip_num):
    print(G.number_of_edges())
    if not os.path.exists('./attack_graph/'):
        os.mkdir('./attack_graph/')
    if not os.path.exists('./attack_graph/'+args.dataset):
        os.mkdir('./attack_graph/'+args.dataset)
    if not os.path.exists('./attack_graph/'+args.dataset+'/toak'):
        os.mkdir('./attack_graph/'+args.dataset+'/toak')

    fname = '_'.join(['toak', args.dataset, str(args.ratio),'l',str(args.walk_len),'n',str(args.walks_per_node),'lmd',str(args.lamda),'1'])
    save = './attack_graph/'+args.dataset+'/toak/'+fname

    edge = []
    idx2id = {v:k for k,v in id2idx.items()}
    for k in e2eid.keys():
        if edge_score[e2eid[k][0]]!=0:
            edge.append([[k[0], k[1]], edge_score[e2eid[k][0]]])

    edge = sorted(edge, key=lambda x:x[1])
    try:
        np.save(args.dataset+str(args.ratio)+'_edge_score', edge)
    except:
        pass

    file = open(save, 'w')
    del_n = 0
    add_n = 0
    for e in edge[:flip_num]:
        if not G.has_edge(e[0][0], e[0][1]):
            add_n+=1
        else:
            del_n+=1
        file.write("{0} {1}\n".format(idx2id[e[0][0]], idx2id[e[0][1]]))
    file.close()
    print('del   ',del_n,'add   ', add_n)
    print('Done. Edges are saving at ', save)
    return 0


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    args = parse_args()
    print(args)
    target_dir = './dataset/'+args.dataset+'/target'

    print("START ATTACK THE GRAPH!!!")
    print("Removing {:3.0%} edges, dataset={}, walks per nodes={:5d}, walk length={:3d}".format(args.ratio, args.dataset, args.walks_per_node, args.walk_len))
    graph  = load_graph(target_dir)
    flip_num = int(graph.number_of_edges()*args.ratio)
    print("Total Edges:", len(list(graph.edges())))
    id2idx = load_id2idx(target_dir)

    #generate VGAE embedding
    t = time.time()
    if os.path.exists('./emb/'+args.dataset+'_vgae_emb.npy'):
        print("VGAE embedding already exist! Load it!")
        z = np.load('./emb/'+args.dataset+'_vgae_emb.npy', allow_pickle=True)
    else:
        print('Train VGAE!')
        z = VGAE(graph, id2idx, target_dir, args)

    H = load_H(args.dataset, id2idx)
    edge_score,e2eid = calculate_score(graph, id2idx, z, H, args)
    graph  = load_graph(target_dir)
    generate_attack(edge_score,e2eid,id2idx,args,graph,flip_num)
    print("ATTACK FINISHED!!! Spend Time={:.3f}".format(time.time() - t))

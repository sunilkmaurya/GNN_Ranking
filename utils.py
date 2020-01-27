from networkit import *
import networkx as nx
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau
import pickle
import scipy.sparse as sp
import copy
import random
import numpy as np
import torch


def get_out_edges(g_nkit,node_sequence):
    global all_out_dict
    all_out_dict = dict()
    for all_n in node_sequence:
        all_out_dict[all_n]=set()
        
    for all_n in node_sequence:
            _ = g_nkit.forEdgesOf(all_n,nkit_outedges)
            
    return all_out_dict

def get_in_edges(g_nkit,node_sequence):
    global all_in_dict
    all_in_dict = dict()
    for all_n in node_sequence:
        all_in_dict[all_n]=set()
        
    for all_n in node_sequence:
            _ = g_nkit.forInEdgesOf(all_n,nkit_inedges)
            
    return all_in_dict


def nkit_inedges(u,v,weight,edgeid):
    all_in_dict[u].add(v)


def nkit_outedges(u,v,weight,edgeid):
    all_out_dict[u].add(v)

    

def nx2nkit(g_nx):
    
    node_num = g_nx.number_of_nodes()
    g_nkit = Graph(directed=True)
    
    for i in range(node_num):
        g_nkit.addNode()
    
    for e1,e2 in g_nx.edges():
        g_nkit.addEdge(e1,e2)
        
    assert g_nx.number_of_nodes()==g_nkit.numberOfNodes(),"Number of nodes not matching"
    assert g_nx.number_of_edges()==g_nkit.numberOfEdges(),"Number of edges not matching"
        
    return g_nkit




    
def clique_check(index,node_sequence,all_out_dict,all_in_dict):
    node = node_sequence[index]
    in_nodes = all_in_dict[node]
    out_nodes = all_out_dict[node]

    for in_n in in_nodes:
        tmp_out_nodes = set(out_nodes)
        tmp_out_nodes.discard(in_n)
        if tmp_out_nodes.issubset(all_out_dict[in_n]) == False:
            return False
    

    return True

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




def graph_to_adj_bet(list_graph,list_n_sequence,list_node_num,model_size):
    

    list_adjacency = list()
    list_adjacency_t = list()
    list_degree = list()
    max_nodes = model_size
    zero_list = list()
    list_rand_pos = list()
    list_sparse_diag = list()
    
    for i in range(len(list_graph)):
        print(f"Processing graphs: {i+1}/{len(list_graph)}",end='\r')
        graph = list_graph[i]
        edges = list(graph.edges())
        graph = nx.MultiDiGraph()
        graph.add_edges_from(edges)

        #self_loops = [i for i in graph.selfloop_edges()]
        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)
        node_sequence = list_n_sequence[i]

        adj_temp = nx.adjacency_matrix(graph,nodelist=node_sequence)

        node_num = list_node_num[i]
        
        adj_temp_t = adj_temp.transpose()
        
        arr_temp1 = np.sum(adj_temp,axis=1)
        arr_temp2 = np.sum(adj_temp_t,axis=1)
        

        arr_multi = np.multiply(arr_temp1,arr_temp2)
        
        arr_multi = np.where(arr_multi>0,1.0,0.0)
        
        degree_arr = arr_multi
        
        non_zero_ind = np.nonzero(degree_arr.flatten())
        non_zero_ind = non_zero_ind[0]
        
        g_nkit = nx2nkit(graph)
        

        in_n_seq = [node_sequence[nz_ind] for nz_ind in non_zero_ind]
        all_out_dict = get_out_edges(g_nkit,node_sequence)
        all_in_dict = get_in_edges(g_nkit,in_n_seq)


        
        for index in non_zero_ind:
           
            is_zero = clique_check(index,node_sequence,all_out_dict,all_in_dict)
            if is_zero == True:
              
                degree_arr[index,0]=0.0
                    
        adj_temp = adj_temp.multiply(csr_matrix(degree_arr))
        adj_temp_t = adj_temp_t.multiply(csr_matrix(degree_arr))
                

        rand_pos = 0
        top_mat = csr_matrix((rand_pos,rand_pos))
        remain_ind = max_nodes - rand_pos - node_num
        bottom_mat = csr_matrix((remain_ind,remain_ind))
        
        list_rand_pos.append(rand_pos)
        #remain_ind = max_nodes - node_num
        #small_arr = csr_matrix((remain_ind,remain_ind))
        
        #adding extra padding to adj mat,normalise and save as torch tensor

        adj_temp = csr_matrix(adj_temp)
        adj_mat = sp.block_diag((top_mat,adj_temp,bottom_mat))
        
        adj_temp_t = csr_matrix(adj_temp_t)
        adj_mat_t = sp.block_diag((top_mat,adj_temp_t,bottom_mat))
        
        adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
        list_adjacency.append(adj_mat)
        
        adj_mat_t = sparse_mx_to_torch_sparse_tensor(adj_mat_t)
        list_adjacency_t.append(adj_mat_t)
    print("")          
    return list_adjacency,list_adjacency_t

def graph_to_adj_close(list_graph,list_n_sequence,list_node_num,model_size,print_time=False):
    

    list_adjacency = list()
    list_adjacency_mod = list()
    list_degree = list()
    max_nodes = model_size
    zero_list = list()
    list_rand_pos = list()
    list_sparse_diag = list()
    
    for i in range(len(list_graph)):
        print(f"Processing graphs: {i+1}/{len(list_graph)}",end='\r')
        graph = list_graph[i]
        edges = list(graph.edges())
        graph = nx.MultiDiGraph()
        graph.add_edges_from(edges)

        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)
        node_sequence = list_n_sequence[i]

        adj_temp = nx.adjacency_matrix(graph,nodelist=node_sequence)

        node_num = list_node_num[i]
        
        adj_temp_t = adj_temp.transpose()
        
        arr_temp1 = np.sum(adj_temp,axis=1)
        arr_temp2 = np.sum(adj_temp_t,axis=1)
        

        arr_multi = np.multiply(arr_temp1,arr_temp2)
        
        arr_multi = np.where(arr_multi>0,1.0,0.0)

        
        degree_arr = arr_multi
        
        non_zero_ind = np.nonzero(degree_arr.flatten())
        non_zero_ind = non_zero_ind[0]
        
        g_nkit = nx2nkit(graph)
        

        in_n_seq = [node_sequence[nz_ind] for nz_ind in non_zero_ind]
        all_out_dict = get_out_edges(g_nkit,node_sequence)
        all_in_dict = get_in_edges(g_nkit,in_n_seq)

        
        for index in non_zero_ind:
           
            is_zero = clique_check(index,node_sequence,all_out_dict,all_in_dict)
            if is_zero == True:
              
                degree_arr[index,0]=0.0


        #modify the in-degree matrix for different layers

        degree_arr = degree_arr.reshape(1,node_num)
 

        #for out_degree
        adj_temp_mod = adj_temp.multiply(csr_matrix(degree_arr))


        rand_pos = 0
        top_mat = csr_matrix((rand_pos,rand_pos))
        remain_ind = max_nodes - rand_pos - node_num
        bottom_mat = csr_matrix((remain_ind,remain_ind))
        
        list_rand_pos.append(rand_pos)
        #remain_ind = max_nodes - node_num
        #small_arr = csr_matrix((remain_ind,remain_ind))
        
        #adding extra padding to adj mat,normalise and save as torch tensor
        
        adj_temp = csr_matrix(adj_temp)
        adj_mat = sp.block_diag((top_mat,adj_temp,bottom_mat))
        
        adj_temp_mod = csr_matrix(adj_temp_mod)
        adj_mat_mod = sp.block_diag((top_mat,adj_temp_mod,bottom_mat))

        
        adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
        list_adjacency.append(adj_mat)
        
        adj_mat_mod = sparse_mx_to_torch_sparse_tensor(adj_mat_mod)
        list_adjacency_mod.append(adj_mat_mod)

    print("")        
    return list_adjacency,list_adjacency_mod



def ranking_correlation(y_out,true_val,node_num,model_size):
    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))

    predict_arr = y_out.cpu().detach().numpy()
    true_arr = true_val.cpu().detach().numpy()


    kt,_ = kendalltau(predict_arr[:node_num],true_arr[:node_num])

    return kt


def loss_cal(y_out,true_val,num_nodes,device,model_size):

    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))
    
    _,order_y_true = torch.sort(-true_val[:num_nodes])

    sample_num = num_nodes*20

    ind_1 = torch.randint(0,num_nodes,(sample_num,)).long().to(device)
    ind_2 = torch.randint(0,num_nodes,(sample_num,)).long().to(device)
    

    rank_measure=torch.sign(-1*(ind_1-ind_2)).float()
        
    input_arr1 = y_out[:num_nodes][order_y_true[ind_1]].to(device)
    input_arr2 = y_out[:num_nodes][order_y_true[ind_2]].to(device)
        

    loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1,input_arr2,rank_measure)
 
    return loss_rank


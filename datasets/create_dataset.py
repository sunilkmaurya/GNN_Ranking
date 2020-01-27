import networkx as nx
import pickle
import numpy as np
import time
import glob
import random
random.seed(10)

def reorder_list(input_list,serial_list):
    new_list_tmp = [input_list[j] for j in serial_list]
    return new_list_tmp

def create_dataset(list_data,num_copies):

    adj_size = 10000
    num_data = len(list_data)
    total_num = num_data*num_copies
    cent_mat = np.zeros((adj_size,total_num),dtype=np.float)
    list_graph = list()
    list_node_num = list()
    list_n_sequence = list()
    mat_index = 0
    for g_data in list_data:

        graph, cent_dict = g_data
        nodelist = [i for i in graph.nodes()]
        assert len(nodelist)==len(cent_dict),"Number of nodes are not equal"
        node_num = len(nodelist)

        for i in range(num_copies):
            tmp_nodelist = list(nodelist)
            random.shuffle(tmp_nodelist)
            list_graph.append(graph)
            list_node_num.append(node_num)
            list_n_sequence.append(tmp_nodelist)

            for ind,node in enumerate(tmp_nodelist):
                cent_mat[ind,mat_index] = cent_dict[node]
            mat_index +=  1


    serial_list = [i for i in range(total_num)]
    random.shuffle(serial_list)

    list_graph = reorder_list(list_graph,serial_list)
    list_n_sequence = reorder_list(list_n_sequence,serial_list)
    list_node_num = reorder_list(list_node_num,serial_list)
    cent_mat_tmp = cent_mat[:,np.array(serial_list)]
    cent_mat = cent_mat_tmp

    return list_graph, list_n_sequence, list_node_num, cent_mat


def get_split(source_file,num_train,num_test,num_copies,adj_size,save_path):

    with open(source_file,"rb") as fopen:
        list_data = pickle.load(fopen)

    num_graph = len(list_data)
    assert num_train+num_test == num_graph,"Required split size doesn't match number of graphs in pickle file."
    
    #For training split
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[:num_train],num_copies = num_copies)

    with open(save_path+"training.pickle","wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)

    #For test split
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[num_train:num_train+num_test],num_copies = 1)

    with open(save_path+"test.pickle","wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)



#creating training/test dataset split for the model

adj_size = 10000
graph_types = ["ER","SF","GRP"]
num_train = 40
num_test = 10
#Number of permutations for node sequence
#Can be raised higher to get more training graphs
num_copies = 6

#Total number of training graphs = 40*6 = 240

for g_type in graph_types:
    print("Loading graphs from pickle files...")
    bet_source_file = "./graphs/"+ g_type + "_data_bet.pickle"
    close_source_file = "./graphs/"+ g_type + "_data_close.pickle"

    #paths for saving splits
    save_path_bet = "./data_splits/"+g_type+"/betweenness/"
    save_path_close = "./data_splits/"+g_type+"/closeness/"

    #save betweenness split
    get_split(bet_source_file,num_train,num_test,num_copies,adj_size,save_path_bet)

    #save closeness split
    get_split(close_source_file,num_train,num_test,num_copies,adj_size,save_path_close)
    print(" Data split saved.")






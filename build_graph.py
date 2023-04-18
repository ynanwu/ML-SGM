import numpy as np
import torch
import pickle
import heapq

from torch_geometric.data import Data,Dataset,DataLoader
from torch_geometric.nn import MetaLayer


class Graph_Generator(torch.nn.Module):
	def __init__(self):
		super(Graph_Generator, self).__init__()


	def cal_center(self,bbox_coords):  # (40,4)
		centers_x=(0.5*(bbox_coords[:,0]+bbox_coords[:,2])).unsqueeze(1)  #(40,1) unsqueeze()增加一个维度
		centers_y =(0.5 * (bbox_coords[:, 1] + bbox_coords[:, 3])).unsqueeze(1)
		centers=torch.cat((centers_x,centers_y),dim=1)

		return centers

	def get_adjacent_cells(self, num_ins):
		sq = int(np.sqrt(num_ins))
		input = np.arange(num_ins).reshape((sq, sq))
		s_idx = []
		t_idx = []
		for i in range(sq):
			for j in range(sq):
				s_ins = input[i, j]
				w = np.asarray(list({(i, j)}))
				adj_idxs = []
				for col in range(w.shape[1]):
					w_ = w.copy()
					w_[:, col] += 1
					adj_idxs.extend(list(w_))
					w_ = w.copy()
					w_[:, col] -= 1
					adj_idxs.extend(list(w_))

				adj_idxs = np.array(adj_idxs)

				# remove out of bounds coordinates
				for col, dim_size in enumerate(input.shape):
					adj_idxs = adj_idxs[(adj_idxs[:, col] >= 0) & (adj_idxs[:, col] < dim_size)]

				t_ins = [input[idx] for idx in list(map(tuple, adj_idxs))]

				s_idx.append(list(np.array(s_ins).repeat(len(t_ins))))
				t_idx.append(t_ins)
		s_idx = torch.from_numpy(np.array([i for p in s_idx for i in p]).reshape(-1, 1))
		t_idx = torch.from_numpy(np.array([i for p in t_idx for i in p]).reshape(-1, 1))

		return s_idx, t_idx

	def build_graph(self,bbox_info,label_feats):
		num_ins,num_label=bbox_info.shape[0],label_feats.shape[0]

		"""build instance graph"""
		s_idx, t_idx = self.get_adjacent_cells(num_ins)
		edge_idx_ins = torch.cat((s_idx, t_idx), dim=1).transpose(1, 0).cuda().type(torch.long)
		edge_attr_ins = torch.cat((bbox_info[edge_idx_ins[0, :]], bbox_info[edge_idx_ins[1, :]]), dim=1).cuda()
		kwargs = [num_ins, num_label]

		"""build cross edge"""
		bbox_idx_C = torch.from_numpy(np.array(list(range(num_ins))).repeat(num_label).reshape(-1, 1)).type(torch.long)
		label_idx_C = torch.from_numpy(np.array(list(range(num_label))).reshape(-1, 1)).repeat(num_ins, 1).type(torch.long)
		edge_index_C = torch.cat((bbox_idx_C, label_idx_C), dim=1).cuda()
		edge_attr_C = torch.cat((bbox_info[edge_index_C[:,0]], label_feats[edge_index_C[:,1]]), dim=1)
		cross_edge=torch.cat((edge_index_C.type(torch.float),edge_attr_C.type(torch.float)),dim=1)

		"""build label graph"""
		label_idx=torch.from_numpy(np.array(list(range(num_label))).repeat(num_label).reshape(-1,num_label)).type(torch.long)
		label_idx1=label_idx.reshape(-1,1)
		label_idx2=label_idx.transpose(1,0).reshape(-1,1)
		edge_idx_label=torch.cat((label_idx1,label_idx2),dim=1).transpose(1,0).cuda()
		edge_attr_label=torch.cat((label_feats[edge_idx_label[0,:]],label_feats[edge_idx_label[1,:]]),dim=1)

		return edge_idx_ins, edge_attr_ins, kwargs, edge_idx_label, edge_attr_label, cross_edge,

	def forward(self,bbox_info,label_feats):
		x_ins_batch, edge_idx_ins_batch, edge_attr_ins_batch, x_label_batch, edge_idx_label_batch, edge_attr_label_batch, cross_edge_batch= [], [], [], [], [], [],[]
		node_sum = 0
		labe_sum = 0
		kwargs_ins = []
		for i in range(bbox_info.shape[0]):
			edge_idx_ins, edge_attr_ins, kwargs, edge_idx_label, edge_attr_label, cross_edge = self.build_graph(bbox_info[i], label_feats[i])

			x_ins_batch.append(bbox_info[i])
			edge_idx_ins_batch.append(edge_idx_ins+node_sum)
			edge_attr_ins_batch.append(edge_attr_ins)
			kwargs_ins.append(kwargs)

			x_label_batch.append(label_feats[i])
			edge_idx_label_batch.append(edge_idx_label+labe_sum)
			edge_attr_label_batch.append(edge_attr_label)
			cross_edge_batch.append(cross_edge)

			node_sum += bbox_info[i].shape[0]
			labe_sum += label_feats[i].shape[0]

		x_ins_batch = torch.cat(x_ins_batch)
		edge_idx_ins_batch = torch.cat(edge_idx_ins_batch, dim=1)
		edge_attr_ins_batch = torch.cat(edge_attr_ins_batch)

		x_label_batch = torch.cat(x_label_batch)
		edge_idx_label_batch = torch.cat(edge_idx_label_batch, dim=1)
		edge_attr_label_batch = torch.cat(edge_attr_label_batch)
		cross_edge_batch = torch.cat(cross_edge_batch)

		graph_ins = Data(x=x_ins_batch, edge_index=edge_idx_ins_batch,
						 edge_attr=edge_attr_ins_batch, kwargs=kwargs_ins)
		graph_label = Data(x=x_label_batch, edge_index=edge_idx_label_batch,
						   edge_attr=edge_attr_label_batch, kwargs=cross_edge_batch)

		return graph_ins, graph_label




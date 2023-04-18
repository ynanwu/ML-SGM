import numpy as np
import torch
import csv
from sklearn.metrics import average_precision_score
from torch_geometric.data import Dataset,Data
import pickle
import os
from build_graph import Graph_Generator
import shutil
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import json

class GraphData(Dataset):
	def __init__(self,data_path, split_file, word2vecFile):
		super(GraphData, self).__init__()

		self.graph_generator=Graph_Generator()
		self.data_path = data_path

		voc_label = pickle.load(open(word2vecFile, 'rb'))
		self.labels = torch.from_numpy(voc_label)

		self.image = read_object_labels_csv(split_file)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.image)
	# return 1600

	def __getitem__(self, index):
		'Generates one sample of data'
		img_name, gt = self.image[index]
		proposals = np.load(os.path.join(self.data_path, img_name + '.npy'), allow_pickle=True)
		proposals = torch.from_numpy(proposals)
		graph_ins,graph_label=self.graph_generator(proposals, self.labels)

		return graph_ins,graph_label,gt

def generate_cross_graph(graph_label,counts):
	cross_edges = graph_label["kwargs"]
	edge_index_cross = cross_edges[:, :2]
	edge_attr_cross = cross_edges[:, 2:]

	start_index, count_ins, count_label = 0, 0, 0
	for count in counts:
		count_i, count_l = count
		num_assignment = int(count_i * count_l)
		edge_index_cross[start_index:start_index + num_assignment, 0] += count_ins
		edge_index_cross[start_index:start_index + num_assignment, 1] += count_label

		start_index += num_assignment
		count_ins += count_i
		count_label += count_l
	edge_index_cross = edge_index_cross.type(torch.long).transpose(1, 0)
	if torch.cuda.is_available():
		graph_cross = Data(x=None, edge_index=edge_index_cross.type(torch.long).cuda(), edge_attr=edge_attr_cross.cuda(),
						   y=None)
	else:
		graph_cross = Data(x=None, edge_index=edge_index_cross.type(torch.long), edge_attr=edge_attr_cross,
						   y=None)
	return graph_cross

class ImageFilelist(data.Dataset):
	def __init__(self, flist, inp_name, transform=None, dataType=None, imgPath=None):
		self.transform = transform
		self.dataType = dataType
		self.imgPath = imgPath
		self.flist = flist
		self.dataType = dataType

		with open(inp_name, 'rb') as f:
			self.inp = pickle.load(f)
		self.inp_name = inp_name

		if self.dataType == 'voc':
			self.images = read_object_labels_csv(self.flist)
		if self.dataType == 'coco':
			self.images = read_object_labels_json(self.flist)

		print('[dataset] VOC 2007 classification set=%s number of images=%d' % (
			set, len(self.images)))

	def __getitem__(self, index):
		path, target = self.images[index]
		img = Image.open(os.path.join(self.imgPath, path + '.jpg')).convert('RGB')
		img = self.transform(img) if self.transform else img

		return (img, self.inp), target

	def __len__(self):
		return len(self.images)

def data_transform(new_size, train=True):
	# transform_list = [transforms.Resize((new_size, new_size)),
	# 				  transforms.ToTensor(),
	# 				  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
	# transform_list = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()] + transform_list if train else transform_list
	# transform = transforms.Compose(transform_list)
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	scale_size = 640
	crop_size = new_size
	if train:
		transform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
										transforms.RandomChoice([transforms.RandomCrop(640),
																 transforms.RandomCrop(576),
																 transforms.RandomCrop(512),
																 transforms.RandomCrop(384),
																 transforms.RandomCrop(320)]),
										transforms.RandomHorizontalFlip(),
										transforms.RandomVerticalFlip(),
										transforms.Resize((crop_size, crop_size)),
										transforms.ToTensor(),
										normalize])
	else:
		transform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
										transforms.CenterCrop(crop_size),
										transforms.ToTensor(),
										normalize])
	return transform

def read_object_labels_csv(file, header=True):
	images = []
	num_categories = 0
	print('[dataset] read', file)
	with open(file, 'r') as f:
		reader = csv.reader(f)
		rownum = 0
		for row in reader:
			# if rownum < 200:
			if header and rownum == 0:
				header = row
			else:
				if num_categories == 0:
					num_categories = len(row) - 1
				name = row[0]
				labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
				labels = torch.from_numpy(labels)
				item = (name, labels)
				images.append(item)
			rownum += 1
	return images

def read_object_labels_json(file):
	images = []
	with open(file, encoding='utf-8') as f:
		line = f.readline()
		d = json.loads(line)
		for i in range(len(d)):
			name = d[i]['file_name'].split('.')[0]
			labels = d[i]['labels']
			labels = torch.tensor(labels, dtype=torch.long)
			labels = torch.zeros(80).scatter_(0, labels, 1)
			# item = (name, labels)
			item = (name, labels)
			images.append(item)

	return images

def mean_average_precision(y_true, y_pred):
	ap = 0
	for true, pred in zip(y_true, y_pred):
		ap += average_precision_score(np.array(true), np.array(pred))
	return float(ap/len(y_true))

def save_checkpoint(state, is_best, filename='', modelbest = ''):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, modelbest)

def cal_metric(gt,prediction):
	prediction=torch.tensor(prediction)
	gt=torch.tensor(gt)

	count_pos = float(torch.sum(gt))
	count_all=float((gt.shape[0]*gt.shape[1]))
	count_neg=count_all-count_pos

	prediction[prediction>=0.5]=1
	prediction[prediction<0.5]=0

	gt_1_pos, pre_1_pos, gt_2_pos, pre_2_pos, gt_3_pos, pre_3_pos, gt_4_pos, pre_4_pos = [], [], [], [], [], [], [], []

	for i, gt_label in enumerate(gt):
		if torch.sum(gt_label) == 1:
			gt_1_pos.append(gt_label)
			pre_1_pos.append(prediction[i])
		if torch.sum(gt_label) == 2:
			gt_2_pos.append(gt_label)
			pre_2_pos.append(prediction[i])
		if torch.sum(gt_label) == 3:
			gt_3_pos.append(gt_label)
			pre_3_pos.append(prediction[i])
		if torch.sum(gt_label) == 4:
			gt_4_pos.append(gt_label)
			pre_4_pos.append(prediction[i])
	gt_1_pos, pre_1_pos, gt_2_pos, pre_2_pos, gt_3_pos, pre_3_pos, gt_4_pos, pre_4_pos = \
		torch.stack(gt_1_pos), torch.stack(pre_1_pos), torch.stack(gt_2_pos), torch.stack(pre_2_pos), \
		torch.stack(gt_3_pos), torch.stack(pre_3_pos), torch.stack(gt_4_pos), torch.stack(pre_4_pos)

	tp_1 = torch.sum((pre_1_pos==gt_1_pos)[gt_1_pos==1]).float()
	tn_1 = torch.sum((pre_1_pos == gt_1_pos)[gt_1_pos == 0]).float()
	fn_1 = torch.sum((pre_1_pos != gt_1_pos)[gt_1_pos == 1]).float()
	fp_1 = torch.sum((pre_1_pos != gt_1_pos)[gt_1_pos == 0]).float()
	count_pos_1 = float(torch.sum(gt_1_pos))
	count_all_1 = float((gt_1_pos.shape[0] * gt_1_pos.shape[1]))
	count_neg_1 = count_all_1 - count_pos_1
	print("count_pos_1:", count_pos_1, "count_neg_1", count_neg_1, "tp_1:",tp_1/count_pos_1,"\ttn_1:",tn_1/count_neg_1,"\tfn_1:",fn_1/count_pos_1,"\tfp_1:",fp_1/count_neg_1)

	tp_2 = torch.sum((pre_2_pos == gt_2_pos)[gt_2_pos == 1]).float()
	tn_2 = torch.sum((pre_2_pos == gt_2_pos)[gt_2_pos == 0]).float()
	fn_2 = torch.sum((pre_2_pos != gt_2_pos)[gt_2_pos == 1]).float()
	fp_2 = torch.sum((pre_2_pos != gt_2_pos)[gt_2_pos == 0]).float()
	count_pos_2 = float(torch.sum(gt_2_pos))
	count_all_2 = float((gt_2_pos.shape[0] * gt_2_pos.shape[1]))
	count_neg_2 = count_all_2 - count_pos_2
	print("count_pos_2:", count_pos_2, "count_neg_2", count_neg_2, "tp_2:", tp_2/count_pos_2, "\ttn_2:", tn_2/count_neg_2, "\tfn_2:", fn_2/count_pos_2, "\tfp_2:", fp_2/count_neg_2)

	tp_3 = torch.sum((pre_3_pos == gt_3_pos)[gt_3_pos == 1]).float()
	tn_3 = torch.sum((pre_3_pos == gt_3_pos)[gt_3_pos == 0]).float()
	fn_3 = torch.sum((pre_3_pos != gt_3_pos)[gt_3_pos == 1]).float()
	fp_3 = torch.sum((pre_3_pos != gt_3_pos)[gt_3_pos == 0]).float()
	count_pos_3 = float(torch.sum(gt_3_pos))
	count_all_3 = float((gt_3_pos.shape[0] * gt_3_pos.shape[1]))
	count_neg_3 = count_all_3 - count_pos_3
	print("count_pos_3:", count_pos_3, "count_neg_3", count_neg_3, "tp_3:", tp_3/count_pos_3, "\ttn_3:", tn_3/count_neg_3, "\tfn_3:", fn_3/count_pos_3, "\tfp_3:", fp_3/count_neg_3)

	tp_4 = torch.sum((pre_4_pos == gt_4_pos)[gt_4_pos == 1]).float()
	tn_4 = torch.sum((pre_4_pos == gt_4_pos)[gt_4_pos == 0]).float()
	fn_4 = torch.sum((pre_4_pos != gt_4_pos)[gt_4_pos == 1]).float()
	fp_4 = torch.sum((pre_4_pos != gt_4_pos)[gt_4_pos == 0]).float()
	count_pos_4 = float(torch.sum(gt_4_pos))
	count_all_4 = float((gt_4_pos.shape[0] * gt_4_pos.shape[1]))
	count_neg_4 = count_all_4 - count_pos_4
	print("count_pos_4:", count_pos_4, "count_neg_4", count_neg_4,"tp_4:", tp_4/count_pos_4, "\ttn_4:", tn_4/count_neg_4, "\tfn_4:", fn_4/count_pos_4, "\tfp_4:", fp_4/count_neg_4)

	tp_=torch.sum((prediction==gt)[gt==1]).float()
	tn_=torch.sum((prediction==gt)[gt==0]).float()
	fn_=torch.sum((prediction!=gt)[gt==1]).float()
	fp_=torch.sum((prediction!=gt)[gt==0]).float()

	rate_tp=tp_/count_pos
	rate_tn=tn_/count_neg
	print("count_pos:", count_pos, "count_neg", count_neg, "tp:",rate_tp,"\ttn:",rate_tn,"\tfn:",fn_/count_pos,"\tfp:",fp_/count_neg)

	return rate_tp,rate_tn

def weighted_cross_entropy(predition,gt,weight):

	loss=weight*gt*torch.log(predition)+(1-weight)*(1-gt)*(torch.log(1-predition))

	return torch.mean(-loss)


def iccv2019_mAP(gt, prediction):
	gt = np.array(gt)
	prediction = np.array(prediction)
	num_target = np.sum(gt, axis=1, keepdims=True)
	threshold = 1 / (num_target + 1e-6)

	sample_num = len(gt)
	class_num = gt.shape[1]
	tp = np.zeros(sample_num)
	fp = np.zeros(sample_num)
	aps = []

	for class_id in range(class_num):
		confidence = prediction[:, class_id]
		sorted_ind = np.argsort(-confidence)
		sorted_label = [gt[x][class_id] for x in sorted_ind]

		for i in range(sample_num):
			tp[i] = (sorted_label[i] > 0)
			fp[i] = (sorted_label[i] <= 0)
		true_num = sum(tp)
		fp = np.cumsum(fp)
		tp = np.cumsum(tp)
		rec = tp / true_num
		prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
		ap = voc_ap(rec, prec)
		aps += [ap]

	np.set_printoptions(precision=3, suppress=True)
	mAP = np.mean(aps)
	return mAP

def voc_ap(rec, prec):
	mrec = np.concatenate(([0.], rec, [1.]))
	mpre = np.concatenate(([0.], prec, [0.]))
	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
	i = np.where(mrec[1:] != mrec[:-1])[0]
	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
	return ap



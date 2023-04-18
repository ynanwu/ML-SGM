import torch
import os
import numpy as np

import math
from torch.nn import init

import argparse
from GNBlock_model import _Model
from util import iccv2019_mAP,cal_metric,weighted_cross_entropy, data_transform, ImageFilelist
from test import test

#/newnfs/zzwu/04_centerNet/wuyanan/wyn/00_Multi-Label/ML-GCN-master_2/data/
#D:/01_code/datasets/voc/

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='voc', type=str, metavar='N', help='run_data')
parser.add_argument('--proposal_data', default='D:/01_code/datasets/voc/', type=str, metavar='PATH',
					help='root dir')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('--end_epochs', default=100, type=int, metavar='H-P',
					help='number of total epochs to run')
parser.add_argument('--dir', dest='dir', default='models/', type=str, metavar='PATH',
					help='model dir')
parser.add_argument('--image_size', default=448, type=int,
					metavar='N', help='image size (default: 227)')
parser.add_argument('--batch_size', default=4, type=int,
					help='number of batch size')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr', default=0.01, type=float,
					metavar='H-P', help='initial learning rate')
parser.add_argument('--k', default=0, type=int, help='KNN')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

args = parser.parse_args()
best_prec = 0
category_num = 0

#5011
transform_train = data_transform(new_size=args.image_size, train=True)
transform_val = data_transform(new_size=args.image_size, train=False)
# DATAPATH = args.proposal_data + args.dataset
if args.dataset == 'voc':
	category_num = 20
	inp_name = args.proposal_data + 'VOC2007/voc_glove_word2vec.pkl'
	imgPath = args.proposal_data + 'VOC2007/VOCdevkit/VOC2007/JPEGImages/'
	trainset = ImageFilelist(flist=args.proposal_data + "VOC2007/classification_trainval.csv", inp_name=inp_name,
							 transform=transform_train, imgPath=imgPath, dataType=args.dataset)
	valset = ImageFilelist(flist=args.proposal_data + "VOC2007/classification_test.csv", inp_name=inp_name,
						   transform=transform_val, imgPath=imgPath, dataType=args.dataset)

if args.dataset == 'coco':
	category_num = 80
	inp_name = args.proposal_data + 'coco2014/coco_glove_word2vec.pkl'
	flist = '/newnfs/zzwu/04_centerNet/wuyanan/wyn/00_Multi-Label/multilabel-DeViSE/data/coco/'
	valset = ImageFilelist(flist=flist + "val_anno.json", inp_name=inp_name,
						   transform=transform_val, imgPath='/newnfs/COCO/val2014/', dataType=args.dataset)
	trainset = ImageFilelist(flist=flist+ "train_anno.json", inp_name=inp_name,
							 transform=transform_train, imgPath='/newnfs/COCO/train2014/', dataType=args.dataset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True)


save_path = "models/1129_test_ap/"
if not os.path.exists(save_path):
	os.makedirs(save_path)

model=_Model()
for layer in model.modules():
	if isinstance(layer, torch.nn.Linear):
		init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
		if layer.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
			bound = 1/math.sqrt(fan_in)
			init.uniform_(layer.bias, -bound, bound)

if args.resume is not None:
	if os.path.isfile(args.resume):
		print("=> loading checkpoint '{}'".format(args.resume))
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']
		best_score = checkpoint['best_score']
		model.load_state_dict(checkpoint['state_dict'])
		print("=> loaded checkpoint best_score '{}' (epoch {})"
			  .format(best_score, checkpoint['epoch']))
	else:
		print("=> no checkpoint found at '{}'".format(args.resume))

criterion = torch.nn.BCELoss()
optimizer=torch.optim.SGD(params=model.parameters(),lr=args.lr, momentum=0.9, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9333)

if torch.cuda.is_available():
	model = model.cuda()
	model = torch.nn.DataParallel(model)

	criterion.cuda()

for epoch in range(args.start_epoch, args.end_epochs):
	running_loss = []
	results = {'predict': [], 'label': []}
	for idx, (inputs, gt) in enumerate(trainloader):
		optimizer.zero_grad()
		img, inp = inputs[0], inputs[1]
		gt = gt.cuda()
		cross_edges, counts = model(img, inp)
		gt[gt== -1] = 0
		count_cur,loss=0,0
		prediction_score = list()
		for j, count in enumerate(counts):
			prediction = cross_edges[count_cur: count_cur+(count[0]*count[1])]  #(6,64)
			prediction = prediction.view(-1,category_num)

			prediction = torch.max(prediction, dim=0)[0]
			prediction_score.append(prediction.unsqueeze(0))

			count_cur += (count[0]*count[1])

		prediction= torch.cat(prediction_score, dim=0)
		results['predict'].extend([x.tolist() for x in prediction])
		results['label'].extend([y.tolist() for y in gt])

		loss =weighted_cross_entropy(prediction, gt, 0.80)

		loss.backward()

		optimizer.step()

		running_loss.append(loss.item())

		if idx % 40 == 39:
			print("[epoch: %d, %5d samples] loss: %.3f" % (
				epoch + 1, (idx + 1) * args.batch_size, np.mean(np.array(running_loss)).item()))
			running_loss = []
	print("*" * 66)
	rate_pt,rate_pn = cal_metric(results['label'], results['predict'])
	ICCV_train_acc = iccv2019_mAP(results['label'], results['predict'])
	print("Epoch:",epoch+1,"\ttrain_pt_rate:",round(rate_pt.item(),4),"\ttrain_pn_rate:",round(rate_pn.item(),4),
		  'ICCV_train_acc:',round(ICCV_train_acc,4))
	print("*" * 66)
	model.eval()
	torch.set_grad_enabled(False)
	test_pt, test_pn, ICCV_test_acc = test(valloader, model, category_num)
	print("Epoch:",epoch+1,"\ttest_pt_rate:",round(test_pt.item(),4),"\ttest_pn_rate:",round(test_pn.item(),4),
		  'ICCV_test_acc:',round(ICCV_test_acc,4))

	scheduler.step()

	is_best = ICCV_test_acc > best_prec
	best_prec = max(ICCV_test_acc, best_prec)
	print(f"best_prec: {best_prec:0.4f}\n")
	print("*" * 66)

	if (epoch) % 10 == 0:
		model_name = "model_" + str(epoch) + "_map_" + str(round(ICCV_test_acc.item(), 2)) + ".pth"
		torch.save({
			'epoch': epoch,
			'state_dict': model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
			'best_score':best_prec
		}, save_path+model_name)
	if is_best:
		print('--------------------------model_best-------------')
		model_best = 'model_best.pth'
		torch.save({
			'epoch': epoch,
			'state_dict': model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
			'best_score': best_prec
		}, save_path+model_best)

	# Training mode
	model.train()
	torch.set_grad_enabled(True)












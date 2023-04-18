import torch
import os
import argparse

from util import weighted_cross_entropy, iccv2019_mAP,data_transform, ImageFilelist,cal_metric
from GNBlock_model import _Model

def test(valloader, model, category_num):
    model.eval()
    running_loss = []
    results = {'predict': [], 'label': []}
    for idx, (inputs, gt) in enumerate(valloader):
        img, inp = inputs[0].cuda(), inputs[1].cuda()
        cross_edges, counts = model(img, inp)
        gt[gt == -1] = 0
        count_cur, loss = 0, 0
        prediction_score = list()
        for j, count in enumerate(counts):
            prediction = cross_edges[count_cur: count_cur + (count[0] * count[1])]  # (6,64)

            prediction = prediction.view(-1, category_num)

            prediction = torch.max(prediction, dim=0)[0]
            prediction_score.append(prediction.unsqueeze(0))

            count_cur += (count[0] * count[1])
        # print('-----------------------prediction_score:', prediction_score)
        prediction = torch.cat(prediction_score, dim=0)
        results['predict'].extend([x.tolist() for x in prediction])
        results['label'].extend([y.tolist() for y in gt])

    # return acc
    rate_pt,rate_pn=cal_metric(results['label'], results['predict'])
    ICCV_test_acc = iccv2019_mAP(results['label'], results['predict'])
    return rate_ptrate_pn, ICCV_test_acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
    parser.add_argument('--dataset', default='voc_grid_vgg16/', type=str, metavar='N', help='run_data')

    parser.add_argument('--proposal_data',
                        default='/newnfs/zzwu/04_centerNet/wyn/graph-multi-lable/code/proposal_info/', type=str,
                        metavar='PATH',
                        help='root dir')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--end_epochs', default=100, type=int, metavar='H-P',
                        help='number of total epochs to run')
    parser.add_argument('--dir', dest='dir', default='models/', type=str, metavar='PATH',
                        help='model dir')
    parser.add_argument('--image_size', default=224, type=int,
                        metavar='N', help='image size (default: 227)')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='number of batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='H-P', help='initial learning rate')

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    args = parser.parse_args()

    inp_name = args.proposal_data + 'word_vec/voc_glove_word2vec.pkl'
    imgPath = '/newnfs/zzwu/04_centerNet/wyn/ML-GCN-master_2/data/voc/VOCdevkit/VOC2007/JPEGImages/'

    transform = data_transform(new_size=args.image_size, train=False)
    valset = ImageFilelist(root=args.proposal_data, flist="ground_truth/voc/classification_test.csv", inp_name=inp_name,
                           transform=transform, imgPath=imgPath)

    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True)

    # Load model
    model=_Model()
    if torch.cuda.is_available():
        model = model.cuda()

    # Test
    #model.load_state_dict(torch.load('../checkpoint/model_weights_best.pth.tar'))
    # optionally resume from a checkpoint
    modelbest_dir = './models/label_conv_dif/1110170303_model_100_pt_0.82_pn_0.97_.pth'

    model_dict = torch.load(modelbest_dir)
    model.load_state_dict(model_dict)

    model.eval()
    torch.set_grad_enabled(False)

    rate_pt,rate_pn,test_acc = test(valloader, model)
    print(f"Accuracy: {test_acc:0.4f}")

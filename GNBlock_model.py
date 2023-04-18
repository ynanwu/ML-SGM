import torch
from torch.nn import Sequential as Seq, Linear as Lin,LayerNorm, ReLU,Sigmoid
from torch_scatter import scatter_mean
from torchvision.models import resnet101
from conv_layer import ConvLayer
from build_graph import Graph_Generator
from util import generate_cross_graph
import torch.nn.functional as F
import torch.nn as nn

latent_dim = 512
final_latent_dim = 256

class GlobalNorm(torch.nn.Module):
    def __init__(self):
        super(GlobalNorm, self).__init__()
    def forward(self, data):
        [h,l]=data.shape
        mean_=torch.mean(data,dim=0).detach()
        var_=torch.var(data,dim=0).detach()

        return (data-mean_)/torch.sqrt(var_+0.00001)

class EdgeModel(torch.nn.Module):
    def __init__(self, insN_dim, insE_dim, labelN_dim, labelE_dim, crossE_dim):
        super(EdgeModel, self).__init__()
        self.latent_insN_dim = final_latent_dim
        self.latent_insE_dim = final_latent_dim
        self.latent_labelN_dim = final_latent_dim
        self.latent_labelE_dim = final_latent_dim
        self.latent_crossE_dim = final_latent_dim

        self.node_encoder_ins = Seq(Lin(insN_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_insN_dim), ReLU(), LayerNorm(self.latent_insN_dim))
        self.edge_encoder_ins = Seq(Lin(insE_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_insE_dim), ReLU(), LayerNorm(self.latent_insE_dim))
        self.node_encoder_label = Seq(Lin(labelN_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_labelN_dim), ReLU(), LayerNorm(self.latent_labelN_dim))
        self.edge_encoder_label = Seq(Lin(labelE_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_labelE_dim), ReLU(), LayerNorm(self.latent_labelE_dim))
        self.edge_encoder_cross = Seq(Lin(crossE_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_crossE_dim), ReLU(), LayerNorm(self.latent_crossE_dim))

        self.edge_mlp_ins = Seq(Lin(self.latent_insN_dim*2 + self.latent_insE_dim, latent_dim), ReLU(), LayerNorm(latent_dim),
                                  Lin(latent_dim, insE_dim), ReLU(), LayerNorm(insE_dim))
        self.edge_mlp_label = Seq(Lin(self.latent_labelN_dim *2+self.latent_insE_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, labelE_dim),
                                  ReLU(), LayerNorm(labelE_dim))
        self.edge_mlp_cross = Seq(Lin(self.latent_insN_dim+self.latent_labelN_dim+self.latent_crossE_dim, latent_dim), ReLU(), LayerNorm(latent_dim),
                                  Lin(latent_dim, crossE_dim),ReLU(), LayerNorm(crossE_dim))

    def forward(self, state_ins,state_label,state_cross):

        node_ins, edge_index_ins, edge_attr_ins=state_ins

        node_label, edge_index_label, edge_attr_label=state_label

        edge_index_cross, edge_attr_cross=state_cross

        """mapping the attributes into a latent space"""
        node_ins, edge_attr_ins =self.node_encoder_ins(node_ins),self.edge_encoder_ins(edge_attr_ins)
        node_label, edge_attr_label = self.node_encoder_label(node_label), self.edge_encoder_label(edge_attr_label)
        edge_attr_cross= self.edge_encoder_cross(edge_attr_cross)

        """instance edges convolution"""
        row_ins, col_ins = edge_index_ins
        src_ins,dest_ins=node_ins[row_ins],node_ins[col_ins]
        out_ins = torch.cat([src_ins, dest_ins, edge_attr_ins], 1)
        edge_attr_ins=self.edge_mlp_ins(out_ins)

        """label edges convolution"""
        row_label, col_label = edge_index_label

        src_label, dest_label = node_label[row_label], node_label[col_label]
        out_label= torch.cat([src_label, dest_label, edge_attr_label], 1)
        edge_attr_label = self.edge_mlp_label(out_label)

        """cross edges convotion"""
        row_cross, col_corss = edge_index_cross
        src_cross, dest_cross = node_ins[row_cross], node_label[col_corss]
        out_cross = torch.cat([src_cross, dest_cross, edge_attr_cross], 1)
        edge_attr_cross = self.edge_mlp_cross(out_cross)

        state_ins = node_ins, edge_index_ins, edge_attr_ins
        state_label = node_label, edge_index_label, edge_attr_label
        state_cross = edge_index_cross, edge_attr_cross

        return state_ins, state_label, state_cross

class NodeModel(torch.nn.Module):
    def __init__(self,insN_dim,insE_dim,labelN_dim,labelE_dim,crossE_dim):
        super(NodeModel, self).__init__()
        self.latent_insN_dim=final_latent_dim
        self.latent_insE_dim = final_latent_dim
        self.latent_labelN_dim = final_latent_dim
        self.latent_labelE_dim = final_latent_dim
        self.latent_crossE_dim = final_latent_dim

        self.node_encoder_ins = Seq(Lin(insN_dim,latent_dim),ReLU(),LayerNorm(latent_dim), Lin(latent_dim, self.latent_insN_dim),ReLU(),LayerNorm(self.latent_insN_dim))
        self.edge_encoder_ins = Seq(Lin(insE_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_insE_dim), ReLU(),LayerNorm(self.latent_insE_dim))
        self.node_encoder_label = Seq(Lin(labelN_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_labelN_dim), ReLU(),LayerNorm(self.latent_labelN_dim))
        self.edge_encoder_label = Seq(Lin(labelE_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_labelE_dim), ReLU(),LayerNorm(self.latent_labelE_dim))
        self.edge_encoder_cross = Seq(Lin(crossE_dim, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.latent_crossE_dim), ReLU(),LayerNorm(self.latent_crossE_dim))

        """############### modify #####################"""
        # self.node_mlp_ins_1 = Seq(Lin(self.latent_insN_dim+self.latent_insE_dim,16),ReLU(),GlobalNorm(),
        #                           Lin(16, self.latent_insN_dim),ReLU(),GlobalNorm())
        # self.node_mlp_ins_2 = Seq(Lin(self.latent_insN_dim+self.latent_insE_dim, 16),ReLU(),GlobalNorm(),
        #                           Lin(16,insN_dim),ReLU(),GlobalNorm())

        self.node_mlp_ins_inner = Seq(Lin(self.latent_insN_dim + self.latent_insE_dim, latent_dim), ReLU(), LayerNorm(latent_dim),
                                  Lin(latent_dim, self.latent_insN_dim), ReLU(), LayerNorm(self.latent_insN_dim))
        self.node_mlp_ins_inter = Seq(Lin(self.latent_insN_dim + self.latent_insE_dim, latent_dim), ReLU(), LayerNorm(latent_dim),
                                  Lin(latent_dim, self.latent_insN_dim), ReLU(), LayerNorm(self.latent_insN_dim))
        self.node_mlp_ins = Seq(Lin(self.latent_insN_dim*3,latent_dim),ReLU(),LayerNorm(latent_dim),Lin(latent_dim,insN_dim),ReLU(),LayerNorm(insN_dim))
        """############### modify done #####################"""
        self.node_mlp_label_inner = Seq(Lin(self.latent_labelN_dim+self.latent_labelE_dim,latent_dim), ReLU(), LayerNorm(latent_dim),
                                        Lin(latent_dim, self.latent_labelN_dim), ReLU(), LayerNorm(self.latent_labelN_dim))
        self.node_mlp_label_inter = Seq(Lin(self.latent_labelN_dim+self.latent_labelE_dim, latent_dim), ReLU(), LayerNorm(latent_dim),
                                        Lin(latent_dim, self.latent_labelN_dim), ReLU(), LayerNorm(self.latent_labelN_dim))
        self.node_mlp_label = Seq(Lin(self.latent_labelN_dim*3, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, labelN_dim), ReLU(), LayerNorm(labelN_dim))

    def forward(self, state_ins,state_label,state_cross):

        node_ins, edge_index_ins, edge_attr_ins=state_ins
        node_label, edge_index_label, edge_attr_label=state_label
        edge_index_cross, edge_attr_cross=state_cross

        """mapping the attributes into a latent space"""
        node_ins, edge_attr_ins =self.node_encoder_ins(node_ins),self.edge_encoder_ins(edge_attr_ins)
        node_label, edge_attr_label = self.node_encoder_label(node_label), self.edge_encoder_label(edge_attr_label)
        edge_attr_cross= self.edge_encoder_cross(edge_attr_cross)

        """cross edges are directed from instances to labels"""
        """instance node attribute update"""
        """############################# modify #############################"""
        row_ins, col_ins = edge_index_ins
        row_cross, col_cross = edge_index_cross
        out_ins_inner = torch.cat([node_ins[row_ins],edge_attr_ins],dim=1)
        out_ins_inner = self.node_mlp_ins_inner(out_ins_inner)
        out_ins_inner = scatter_mean(out_ins_inner,col_ins,dim=0,dim_size=node_ins.size(0))

        out_ins_inter =torch.cat([node_label[col_cross],edge_attr_cross],dim=1)
        out_ins_inter = self.node_mlp_ins_inter(out_ins_inter)
        out_ins_inter = scatter_mean(out_ins_inter,row_cross,dim=0,dim_size=node_ins.size(0))

        node_ins=torch.cat([node_ins,out_ins_inner,out_ins_inter],dim=1)
        node_ins=self.node_mlp_ins(node_ins)
        """############################# modify done#############################"""

        """label node attribute update"""
        row_label, col_label = edge_index_label
        row_cross, col_cross = edge_index_cross
        out_label_inner = torch.cat([node_label[row_label], edge_attr_label], dim=1)
        out_label_inner = self.node_mlp_label_inner(out_label_inner)
        out_label_inner = scatter_mean(out_label_inner, col_label, dim=0, dim_size=node_label.size(0))

        out_label_inter = torch.cat([node_ins[row_cross], edge_attr_cross], dim=1)
        out_label_inter = self.node_mlp_label_inter(out_label_inter)
        out_label_inter = scatter_mean(out_label_inter, col_cross, dim=0, dim_size=node_label.size(0))
        node_label = torch.cat([node_label, out_label_inner,out_label_inter], dim=1)
        node_label=self.node_mlp_label(node_label)

        state_ins=node_ins, edge_index_ins, edge_attr_ins
        state_label=node_label, edge_index_label, edge_attr_label
        state_cross=edge_index_cross, edge_attr_cross

        return state_ins,state_label,state_cross

class res_fea(torch.nn.Module):
    def __init__(self):
        super(res_fea, self).__init__()
        self.net = resnet101(pretrained=True)
        self.conv = nn.Conv2d(2048, 1024, kernel_size=3, stride=2, bias=False, padding=1)
        self.bn = nn.BatchNorm2d(1024, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)

        output = self.conv(output)
        output = self.bn(output)
        output = self.relu(output)

        return output

class _Model(torch.nn.Module):
    def __init__(self,dim_node_ins=1024,dim_node_label=300,dim_edge_ins=2048,dim_edge_label=600):
        super(_Model,self).__init__()

        # self.feature = vgg16(pretrained=True).features
        self.feature = res_fea()
        # self.feature = resnet101(pretrained=True)     # (1, 1000)
        self.graph_generator = Graph_Generator()

        self.red_dim_insN = final_latent_dim
        self.red_dim_insE = final_latent_dim
        self.red_dim_labelN = final_latent_dim
        self.red_dim_labelE = final_latent_dim
        self.red_dim_crossE = final_latent_dim

        "dimension reduction"
        self.red_node_ins=Seq(
            Lin(dim_node_ins,latent_dim),ReLU(),LayerNorm(latent_dim),Lin(latent_dim,self.red_dim_insN),ReLU(),LayerNorm(self.red_dim_insN)
        )
        self.red_edge_ins=Seq(
            Lin(dim_edge_ins, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.red_dim_insE), ReLU(), LayerNorm(self.red_dim_insE)
        )
        self.red_node_label = Seq(
            Lin(dim_node_label, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.red_dim_labelN), ReLU(), LayerNorm(self.red_dim_labelN)
        )
        self.red_edge_label= Seq(
            Lin(dim_edge_label, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.red_dim_labelE), ReLU(), LayerNorm(self.red_dim_labelE)
        )
        self.red_edge_cross=Seq(
            Lin(dim_node_label+dim_node_ins, latent_dim), ReLU(), LayerNorm(latent_dim), Lin(latent_dim, self.red_dim_crossE), ReLU(), LayerNorm(self.red_dim_crossE)
        )

        self.pro_layer1=ConvLayer(EdgeModel(self.red_dim_insN,self.red_dim_insE,self.red_dim_labelN,self.red_dim_labelE,self.red_dim_crossE),
                                  NodeModel(self.red_dim_insN,self.red_dim_insE,self.red_dim_labelN,self.red_dim_labelE,self.red_dim_crossE))
        self.pro_layer2=ConvLayer(EdgeModel(self.red_dim_insN,self.red_dim_insE,self.red_dim_labelN,self.red_dim_labelE,self.red_dim_crossE),
                                  NodeModel(self.red_dim_insN,self.red_dim_insE,self.red_dim_labelN,self.red_dim_labelE,self.red_dim_crossE))
        self.decoder_edge_cross= Seq(
            Lin(self.red_dim_crossE, 128), ReLU(),LayerNorm(128), Lin(128, 1), Sigmoid()
        )

    def forward(self, imgs, inp):
        img_feature = self.feature(imgs) #(BS,1024,7,7)
        b, c, w, h = img_feature.shape
        img_feature = img_feature.reshape(b, c, -1)
        img_feature = torch.transpose(img_feature, 1, 2)
        img_feature = F.normalize(img_feature, p=2, dim=2)
        graph_ins, graph_label = self.graph_generator(img_feature, inp)

        counts = torch.tensor(graph_ins["kwargs"]).cuda()
        graph_cross = generate_cross_graph(graph_label, counts)

        node_ins, edge_index_ins, edge_attr_ins, num_node_ins = graph_ins["x"], graph_ins["edge_index"], \
                                                                graph_ins["edge_attr"], graph_ins["kwargs"]

        node_label, edge_index_label, edge_attr_label, num_node_label = graph_label["x"], graph_label["edge_index"], \
                                                                graph_label["edge_attr"], graph_label["kwargs"]

        edge_index_cross, edge_attr_cross = graph_cross["edge_index"],graph_cross["edge_attr"]

        """dimension reduction"""
        node_ins, edge_attr_ins = self.red_node_ins(node_ins),self.red_edge_ins(edge_attr_ins)
        node_label, edge_attr_label = self.red_node_label(node_label), self.red_edge_label(edge_attr_label)
        edge_attr_cross = self.red_edge_cross(edge_attr_cross)

        """convolution layers"""
        state_ins, state_label, state_cross = [node_ins, edge_index_ins, edge_attr_ins],\
                                              [node_label, edge_index_label, edge_attr_label],\
                                              [edge_index_cross, edge_attr_cross]

        state_ins, state_label, state_cross = self.pro_layer1(state_ins, state_label, state_cross)
        state_ins, state_label, state_cross = self.pro_layer2(state_ins, state_label, state_cross)

        _,edge_attr_cross=state_cross

        prediction=self.decoder_edge_cross(edge_attr_cross)

        return prediction, counts

    def data_normlize(self, data):
        return -1+2*data
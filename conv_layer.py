import torch


class ConvLayer(torch.nn.Module):

    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super(ConvLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

    def forward(self, state_ins, state_label, state_cross):
        """"""
        state_ins, state_label, state_cross = self.edge_model(state_ins, state_label, state_cross)
        state_ins, state_label, state_cross = self.node_model(state_ins, state_label, state_cross)

        return state_ins, state_label, state_cross

    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)

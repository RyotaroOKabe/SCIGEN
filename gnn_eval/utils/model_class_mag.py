import torch
from torch_scatter import scatter_mean
from e3nn.o3 import Irreps, spherical_harmonics
from e3nn.nn import Gate
from e3nn.math import soft_one_hot_linspace
from utils.model_class import CustomCompose, GraphConvolution, tp_path_exists
torch.autograd.set_detect_anomaly(True)
class GraphNetworkClassifierMag(torch.nn.Module):
    """
    Graph Neural Network for magnetic property classification.
    Supports binary and multi-class classification using e3nn's equivariant features.
    """
    def __init__(self,
                 mul,
                 irreps_out,
                 lmax,
                 nlayers,
                 number_of_basis,
                 radial_layers,
                 radial_neurons,
                 node_dim,
                 node_embed_dim,
                 input_dim,
                 input_embed_dim,
                 num_classes=2):
        super().__init__()
        
        self.mul = mul
        self.irreps_in = Irreps(str(input_embed_dim)+'x0e')
        self.irreps_node_attr = Irreps(str(node_embed_dim)+'x0e')
        self.irreps_edge_attr = Irreps.spherical_harmonics(lmax)
        self.irreps_hidden = Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = Irreps(irreps_out)
        self.number_of_basis = number_of_basis

        act = {1: torch.nn.functional.silu,
               -1: torch.tanh}
        act_gates = {1: torch.sigmoid,
                     -1: torch.tanh}

        self.layers = torch.nn.ModuleList()
        irreps_in = self.irreps_in
        for _ in range(nlayers):
            irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps_in, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],
                        irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                        irreps_gated)
            conv = GraphConvolution(irreps_in,
                                    self.irreps_node_attr,
                                    self.irreps_edge_attr,
                                    gate.irreps_in,
                                    number_of_basis,
                                    radial_layers,
                                    radial_neurons)

            irreps_in = gate.irreps_out

            self.layers.append(CustomCompose(conv, gate))
        #last layer: conv
        self.layers.append(GraphConvolution(irreps_in,
                        self.irreps_node_attr,
                        self.irreps_edge_attr,
                        self.irreps_out,
                        number_of_basis,
                        radial_layers,
                        radial_neurons,)
                        )

        self.emx = torch.nn.Linear(input_dim, input_embed_dim, dtype = torch.float64)
        self.emz = torch.nn.Linear(node_dim, node_embed_dim, dtype = torch.float64)
        self.num_classes = num_classes
        if self.num_classes == 2:
            self.classifier = torch.nn.Linear(self.irreps_out.dim, 1, dtype=torch.float64)
        else:
            self.classifier = torch.nn.Linear(self.irreps_out.dim, self.num_classes, dtype=torch.float64)


    def forward(self, data):
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]
        edge_vec = data['edge_vec']
        edge_len = data['edge_len']
        edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'][0], self.number_of_basis, basis = 'gaussian', cutoff = False) 
        edge_sh = spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization = 'component')
        edge_attr = edge_sh
        numb = data['numb']
        x = torch.relu(self.emx(torch.relu(data['x'])))
        z = torch.relu(self.emz(torch.relu(data['z'])))
        node_deg = data['node_deg']
        n=None
        count = 0
        for layer in self.layers:
            x = layer(x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb, n)
            count += 1
        x = scatter_mean(x, data.batch, dim=0)
        if self.num_classes == 2:
            x = self.classifier(x)
            x = torch.sigmoid(x)
        else: 
            x = self.classifier(x)
            x = torch.softmax(x, dim=-1)
        return x


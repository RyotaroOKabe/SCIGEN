import os
import torch
from torch_scatter import scatter_mean, scatter
from e3nn.o3 import Irrep, Irreps, spherical_harmonics, TensorProduct, FullyConnectedTensorProduct
from e3nn.nn import Gate, FullyConnectedNet
from e3nn.math import soft_one_hot_linspace
import math
import time

torch.autograd.set_detect_anomaly(True)
palette = ['#43AA8B', '#F8961E', '#F94144', '#277DA1']
seedn = 42


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    """
    Check if a tensor product path exists between irreps_in1, irreps_in2 and the output ir_out.
    Args:
        irreps_in1 (Irreps): First input irreps.
        irreps_in2 (Irreps): Second input irreps.
        ir_out (Irrep): Output irrep.
    
    Returns:
        bool: True if path exists, False otherwise.
    """
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

class CustomCompose(torch.nn.Module):
    """
    Custom module to sequentially apply two modules, storing intermediate outputs.
    """
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x

class GraphConvolution(torch.nn.Module):
    """
    Graph convolution layer that processes node and edge features.
    """
    def __init__(self,
                 irreps_in,
                 irreps_node_attr,
                 irreps_edge_attr,
                 irreps_out,
                 number_of_basis,
                 radial_layers,
                 radial_neurons):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_node_attr = Irreps(irreps_node_attr)
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_out = Irreps(irreps_out)

        self.linear_input = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in)
        self.linear_mask = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)
        
        irreps_mid = []
        instructions = []
        for i, (mul, irrep_in) in enumerate(self.irreps_in):
            for j, (_, irrep_edge_attr) in enumerate(self.irreps_edge_attr):
                for irrep_mid in irrep_in * irrep_edge_attr:
                    if irrep_mid in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, irrep_mid))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_1, i_2, p[i_out], mode, train) for (i_1, i_2, i_out, mode, train) in instructions]

        self.tensor_edge = TensorProduct(self.irreps_in,
                                         self.irreps_edge_attr,
                                         irreps_mid,
                                         instructions,
                                         internal_weights=False,
                                         shared_weights=False)
        
        self.edge2weight = FullyConnectedNet([number_of_basis] + radial_layers * [radial_neurons] + [self.tensor_edge.weight_numel], torch.nn.functional.silu)
        self.linear_output = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out)

    def forward(self,
                node_input,
                node_attr,
                node_deg,
                edge_src,
                edge_dst,
                edge_attr,
                edge_length_embedded,
                numb, n):

        node_input_features = self.linear_input(node_input, node_attr)
        node_features = torch.div(node_input_features, torch.pow(node_deg, 0.5))

        node_mask = self.linear_mask(node_input, node_attr)

        edge_weight = self.edge2weight(edge_length_embedded)
        edge_features = self.tensor_edge(node_features[edge_src], edge_attr, edge_weight)

        node_features = scatter(edge_features, edge_dst, dim = 0, dim_size = node_features.shape[0])
        node_features = torch.div(node_features, torch.pow(node_deg, 0.5))

        node_output_features = self.linear_output(node_features, node_attr)

        node_output = node_output_features

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        mask = self.linear_mask.output_mask
        c_x = (1 - mask) + c_x * mask
        return c_s * node_mask + c_x * node_output


class BaseGraphNetwork(torch.nn.Module):
    """
    Base class for the graph network models with shared functionality.
    Subclasses should implement any specific functionality.
    """
    def __init__(self, mul, irreps_out, lmax, nlayers, number_of_basis, radial_layers, radial_neurons, 
                 node_dim, node_embed_dim, input_dim, input_embed_dim, **kwargs):
        super().__init__()
        self.mul = mul
        self.irreps_in = Irreps(str(input_embed_dim) + 'x0e')
        self.irreps_node_attr = Irreps(str(node_embed_dim) + 'x0e')
        self.irreps_edge_attr = Irreps.spherical_harmonics(lmax)
        self.irreps_hidden = Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = Irreps(irreps_out)
        self.number_of_basis = number_of_basis

        self.act = {1: torch.nn.functional.silu, -1: torch.tanh}
        self.act_gates = {1: torch.sigmoid, -1: torch.tanh}
        
        # Embedding layers
        self.emx = torch.nn.Linear(input_dim, input_embed_dim, dtype=torch.float64)
        self.emz = torch.nn.Linear(node_dim, node_embed_dim, dtype=torch.float64)
        
        self.layers = self._build_layers(nlayers, number_of_basis, radial_layers, radial_neurons)

    def _build_layers(self, nlayers, number_of_basis, radial_layers, radial_neurons):
        """
        Build layers for the network with gates and convolutions.
        """
        layers = torch.nn.ModuleList()
        irreps_in = self.irreps_in
        for _ in range(nlayers):
            irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            ir = '0e' if tp_path_exists(irreps_in, self.irreps_edge_attr, '0e') else '0o'
            irreps_gates = Irreps([(self.mul, ir) for self.mul, _ in irreps_gated])

            gate = Gate(irreps_scalars, [self.act[ir.p] for _, ir in irreps_scalars],
                        irreps_gates, [self.act_gates[ir.p] for _, ir in irreps_gates],
                        irreps_gated)
            conv = GraphConvolution(irreps_in, self.irreps_node_attr, self.irreps_edge_attr, gate.irreps_in, number_of_basis, radial_layers, radial_neurons)

            irreps_in = gate.irreps_out
            layers.append(CustomCompose(conv, gate))
        self.irreps_in_fin = irreps_in    
        return layers

    def _shared_forward(self, data):
        """
        Shared part of the forward pass common across different models.
        """
        edge_src, edge_dst = data['edge_index']
        edge_vec = data['edge_vec']
        edge_len = data['edge_len']
        edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'].item(), self.number_of_basis, basis='gaussian', cutoff=False)
        edge_sh = spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_attr = edge_sh

        numb = data['numb']
        x = torch.relu(self.emx(torch.relu(data['x'])))
        z = torch.relu(self.emz(torch.relu(data['z'])))
        node_deg = data['node_deg']
        return x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb


class GraphNetwork_Class(BaseGraphNetwork):
    """
    Graph Network model for VVN variant.
    """
    def __init__(self, mul, irreps_out, lmax, nlayers, number_of_basis, radial_layers, radial_neurons, 
                 node_dim, node_embed_dim, input_dim, input_embed_dim, **kwargs):
        super().__init__(mul, irreps_out, lmax, nlayers, number_of_basis, radial_layers, radial_neurons, 
                         node_dim, node_embed_dim, input_dim, input_embed_dim, **kwargs)
        
        self.layers.append(GraphConvolution(
                        self.irreps_in_fin,
                        self.irreps_node_attr,
                        self.irreps_edge_attr,
                        self.irreps_out,
                        number_of_basis,
                        radial_layers,
                        radial_neurons,)
                        )
        
        self.binary_classifier = torch.nn.Linear(self.irreps_out.dim, 1, dtype=torch.float64)
        
    def forward(self, data):
        x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb = self._shared_forward(data)

        for layer in self.layers:
            x = layer(x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb, None)
        
        x = scatter_mean(x, data.batch, dim=0)
        x = self.binary_classifier(x)
        x = torch.sigmoid(x)    #.unsqueeze(0)  # Apply sigmoid activation
        return x



import torch
import numpy as np
import math
import random
from scigen.pl_modules.diffusion_w_type import MAX_ATOMIC_NUM 
Pi = math.pi

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']


def lattice_params_to_matrix_xy_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree (alpha, beta, gamma)
    
    Returns:
    A torch.Tensor of shape (N, 3, 3) representing the lattice matrix.
    """
    # Convert angles from degrees to radians
    angles_r = torch.deg2rad(angles)
    # Extract the angles for clarity
    alpha = angles_r[:, 0]
    beta = angles_r[:, 1]
    gamma = angles_r[:, 2]
    # Calculate cosines and sines of the angles
    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)
    # Lattice vector a along x-axis
    vector_a = torch.stack([lengths[:, 0],  # a_x = a
                            torch.zeros_like(lengths[:, 0]),  # a_y = 0
                            torch.zeros_like(lengths[:, 0])], dim=1)  # a_z = 0
    # Lattice vector b in the xy-plane
    vector_b = torch.stack([lengths[:, 1] * cos_gamma,  # b_x = b * cos(gamma)
                            lengths[:, 1] * sin_gamma,  # b_y = b * sin(gamma)
                            torch.zeros_like(lengths[:, 1])], dim=1)  # b_z = 0
    # Lattice vector c in the general 3D direction
    vector_c_x = lengths[:, 2] * cos_beta
    vector_c_y = lengths[:, 2] * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    vector_c_z = lengths[:, 2] * torch.sqrt(1 - cos_beta**2 - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma)**2)
    vector_c = torch.stack([vector_c_x, vector_c_y, vector_c_z], dim=1)
    # Stack the vectors into a (N, 3, 3) matrix
    return torch.stack([vector_a, vector_b, vector_c], dim=1)

mask_l_reduced = torch.tensor([[1, 1, 0]])   #TODO: remove this line after making sure the mask_l is a (3,3) tensor
mask_l_reduced_full = torch.tensor([[1, 1, 1]])
mask_l_default = torch.tensor([[[1, 1, 1], 
                               [1, 1, 1], 
                               [0, 0, 0]]])   
mask_l_cvert = torch.tensor([[[1, 1, 1], 
                            [1, 1, 1], 
                            [1, 1, 0]]])
mask_l_full = torch.ones(3, 3, dtype=torch.int)
mask_l_zonly = torch.tensor([[[0, 0, 0], 
                            [0, 0, 0], 
                            [1, 1, 1]]])
mask_l_zeros = torch.zeros(3, 3, dtype=torch.int)

def cart2frac(cart_coords, lattice_matrix): 
    """
    Converts Cartesian coordinates to fractional coordinates.
    
    Parameters:
    - cart_coords: torch.tensor with shape (N, 2) or (N, 3)
    - lattice_vectors: 2x2 or 3x3 matrix of lattice vectors (torch.tensor) as columns

    Returns:
    - Fractional coordinates as ndarray
    """
    # Calculate the inverse of the lattice matrix
    lattice_inv = torch.inverse(lattice_matrix)
    # Calculate fractional coordinates
    fractional_coords = torch.einsum('ij,ki->kj', lattice_inv, cart_coords)
    return fractional_coords

def reflect_across_line(coords, line):  
    """
    Reflects multiple points across a line defined by `line = [a, b]` corresponding to `y = ax + b`.
    
    Parameters:
    - coords: torch.tensor, tensor of shape (n, 2) where n is the number of points, each represented by (x, y).
    - line: torch.tensor, tensor of shape (2,) representing the line coefficients [a, b] for the line y = ax + b.
    
    Returns:
    - A tensor of shape (n, 2) representing the reflected points.
    """
    a, b = line
    x1, y1 = coords[:, 0], coords[:, 1]

    # Calculate the projection of (x1, y1) onto the line y = ax + b
    x_proj = (x1 + a * (y1 - b)) / (1 + a**2)
    y_proj = a * x_proj + b
    
    # Calculate the reflection points
    reflected_x = x1 + 2 * (x_proj - x1)
    reflected_y = y1 + 2 * (y_proj - y1)
    
    return torch.stack([reflected_x, reflected_y], dim=1)


def vector_to_line_equation(vector, points):    
    vx, vy = vector[0], vector[1]
    if vx == 0:
        raise ValueError("The vector defines a vertical line, not representable as y = ax + b")
    
    x0, y0 = points[:, 0], points[:, 1]
    a = vy / vx
    b = y0 - a * x0
    
    return torch.stack([a.expand_as(b), b], dim=1)

class SC_Base():
    """
    Base class for structural constraints
    """
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        self.bond_len = bond_len
        self.num_atom = int(num_atom) if num_atom is not None else None
        self.type_known = type_known
        self.frac_z = frac_z
        self.c_vec_cons = c_vec_cons
        self.reduced_mask = reduced_mask
        self.device = device
        self.a_scale, self.b_scale, self.c_scale = None, None, None     # Initialize lattice scaling wrt self.bond_len
        self.mask_l = self.get_mask_l()
        self.frac_known = None  # Initialize known fractional coordinates
        self.num_known = None   # Initialize number of known atoms
        self.use_constraints = True

    def get_cell(self, alpha=90, beta=90, gamma=90):
        angles = [alpha, beta, gamma]
        self.c_scale = self.c_vec_cons['scale'] if self.c_vec_cons['scale'] is not None else np.mean([self.a_scale, self.b_scale])
        self.a_len, self.b_len, self.c_len = self.a_scale * self.bond_len, self.b_scale * self.bond_len, self.c_scale * self.bond_len
        self.cell_lengths = torch.tensor([self.a_len, self.b_len, self.c_len], dtype=torch.float, device=self.device)    # lttice lengths in Angstrom
        self.cell_angles_d = torch.tensor(angles, dtype=torch.float, device=self.device)   # lattice angles in degrees    #TODO: need to set
        return lattice_params_to_matrix_xy_torch(self.cell_lengths.unsqueeze(0), self.cell_angles_d.unsqueeze(0)).squeeze(0)
    

    def get_mask_l(self):
        c_scale, c_vert = self.c_vec_cons['scale'], self.c_vec_cons['vert']
        if c_vert:
            self.c_vec_cons['scale'] = None
            return mask_l_cvert
        
        if self.reduced_mask:
            if c_scale is None:
                self.c_vec_cons['vert'] = False
                return mask_l_reduced
            self.c_vec_cons['vert'] = True
            return mask_l_reduced_full
        
        else: 
            if c_scale is None:
                if c_vert:
                    return mask_l_cvert
                else:
                    return mask_l_default
            else:
                self.c_vec_cons['vert'] = True
                return mask_l_full

    def frac_coords_all(self):
        # self.num_atom, self.frac_known
        fcoords_zero = torch.zeros(self.num_atom, 3)
        # n_kn = self.frac_known.shape[0]
        fcoords, mask = fcoords_zero.clone(), fcoords_zero.clone()
        fcoords[:self.num_known, :] = self.frac_known
        fcoords = fcoords%1
        mask[:self.num_known, :] = torch.ones_like(self.frac_known) 
        if self.reduced_mask:
            mask = mask[:, 0].flatten()   #TODO: mask dimension must be (N, 3), which was transformed  from (N,) in the original code
        # return fcoords, mask
        self.frac_coords, self.mask_x = fcoords, mask
        if not self.use_constraints:
            self.mask_x = torch.zeros_like(self.mask_x)

    def atm_types_all(self):
        # self.num_atom, self.num_known, self.type_known
        types_idx_known = [chemical_symbols.index(self.type_known)] * self.num_known
        types_unk = random.choices(chemical_symbols[1:MAX_ATOMIC_NUM+1], k=int(self.num_atom-self.num_known))
        types_idx_unk = [chemical_symbols.index(elem) for elem in types_unk]    # list of unknown atom types (randomly chosen)
        types = torch.tensor(types_idx_known + types_idx_unk)
        mask = torch.zeros_like(types)
        mask[:self.num_known] = 1
        self.atom_types, self.mask_t = types, mask
        if not self.use_constraints:
            self.mask_t = torch.zeros_like(self.mask_t) 


class SC_Vanilla(SC_Base):
    """
    Vanilla case with no constraints
    """
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        self.use_constraints = False
        # Lattice 
        self.a_scale, self.b_scale = 1, 1 
        self.cell = self.get_cell(gamma=90)
        # coords
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z]]) 
        self.num_known = self.frac_known.shape[0]
        self.mask_l = torch.zeros_like(self.mask_l)
        
class SC_Triangular(SC_Base):
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 1, 1 
        self.cell = self.get_cell(gamma=120)
        # coords
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z]]) 
        self.num_known = self.frac_known.shape[0]


class SC_Honeycomb(SC_Base):
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = math.sqrt(3), math.sqrt(3) 
        self.cell = self.get_cell(gamma=120)
        # coords
        self.frac_known = torch.tensor([[1/3, 2/3, self.frac_z], 
                                        [2/3, 1/3, self.frac_z]]) 
        self.num_known = self.frac_known.shape[0]


class SC_Kagome(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 2, 2 
        self.cell = self.get_cell(gamma=120)
        # coords
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z], 
                                        [0.5, 0.0, self.frac_z], 
                                        [0.0, 0.5, self.frac_z]])  
        self.num_known = self.frac_known.shape[0]


class SC_Square(SC_Base):
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 1, 1  
        self.cell = self.get_cell(gamma=90)
        # coords
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z]]) 
        self.num_known = self.frac_known.shape[0]


class SC_ElongatedTriangular(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 1, math.sqrt((1/2)**2 + (1 + math.sqrt(3)/2)**2)  
        gamma_deg = math.degrees(math.atan((1 + math.sqrt(3)/2)/(1/2))) 
        self.cell = self.get_cell(gamma=gamma_deg)  
        # coords
        cart_known_xy = torch.tensor([[0., 0.], 
                            [self.bond_len/2, self.bond_len*math.sqrt(3)/2]]).to(self.device)
        self.num_known = cart_known_xy.shape[0]
        cell_xy = self.cell[:2, :2]
        frac_z_known = self.frac_z*torch.ones((self.num_known, 1))
        self.frac_known = torch.cat([cart2frac(cart_known_xy.to(device), cell_xy.to(device)), frac_z_known.to(device)], dim=-1)   


class SC_SnubSquare(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        cell_vert0_xy = torch.tensor([[math.cos(Pi/12), -math.sin(Pi/12)]])*self.bond_len/math.sqrt(2)
        cell_vert1_2_xy = torch.tensor([[math.sqrt(3)*math.cos(Pi/6), math.sqrt(3)*math.sin(Pi/6)], 
                                [math.cos(Pi*2/3), math.sin(Pi*2/3)]])*(1+math.sqrt(3))*self.bond_len/2
        cell_xy = cell_vert1_2_xy - cell_vert0_xy
        self.a_scale, self.b_scale = cell_xy.norm(dim=-1)/self.bond_len  
        self.cell = self.get_cell(gamma=90)
        # coords
        cart_known_xy = torch.tensor([[math.cos(Pi/6), math.sin(Pi/6)],
                                 [0, 1],
                                 [math.cos(Pi/3), 1 + math.sin(Pi/3)],
                                 [math.cos(Pi/6) + math.cos(Pi/3), math.sin(Pi/6) + math.sin(Pi/3)]])*self.bond_len - cell_vert0_xy
        cart_known_xy = cart_known_xy.to(self.device)
        self.num_known = cart_known_xy.shape[0]
        frac_z_known = self.frac_z*torch.ones((self.num_known, 1))
        self.frac_known = torch.cat([cart2frac(cart_known_xy.to(device), cell_xy.to(device)), frac_z_known.to(device)], dim=-1)  

class SC_TruncatedSquare(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 1+math.sqrt(2), 1+math.sqrt(2) 
        self.cell = self.get_cell(gamma=90)
        # coords
        x = 1/(2+math.sqrt(2))
        self.frac_known = torch.tensor([[x, 0.0, self.frac_z],
                                        [1-x, 0.0, self.frac_z],
                                        [0.0, x, self.frac_z],
                                        [0.0, 1-x, self.frac_z]])
        self.num_known = self.frac_known.shape[0]


class SC_SmallRhombotrihexagonal(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        cell_xy = torch.tensor([[math.cos(Pi/6), math.sin(Pi/6)], 
                                [math.cos(5*Pi/6), math.sin(5*Pi/6)]])*self.bond_len*(1+math.sqrt(3))
        self.a_scale, self.b_scale = cell_xy.norm(dim=-1)/self.bond_len  
        self.cell = self.get_cell(gamma=120)
        # coords
        cart_known_xy = torch.tensor([[1., math.sqrt(3)], 
                                [1+math.sqrt(3), 1+math.sqrt(3)],
                                [1, 2+math.sqrt(3)],
                                [-1., math.sqrt(3)], 
                                [-1-math.sqrt(3), 1+math.sqrt(3)],
                                [-1, 2+math.sqrt(3)]])*self.bond_len/2
        self.num_known = cart_known_xy.shape[0]
        frac_z_known = self.frac_z*torch.ones((self.num_known, 1))
        self.frac_known = torch.cat([cart2frac(cart_known_xy.to(device), cell_xy.to(device)), frac_z_known.to(device)], dim=-1)  


class SC_SnubHexagonal(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        cell_xy = torch.tensor([[4., 2*math.sqrt(3)], 
                                [-5., math.sqrt(3)]])*self.bond_len/2
        self.a_scale, self.b_scale = cell_xy.norm(dim=-1)/self.bond_len  
        self.cell = self.get_cell(gamma=120)
        # coords
        cart_known_xy = torch.tensor([[1., math.sqrt(3)], 
                                [2., 2*math.sqrt(3)],
                                [-1., math.sqrt(3)],
                                [0., 2*math.sqrt(3)],
                                [-3., math.sqrt(3)],
                                [-2, 2*math.sqrt(3)]])*self.bond_len/2
        self.num_known = cart_known_xy.shape[0]
        frac_z_known = self.frac_z*torch.ones((self.num_known, 1))
        self.frac_known = torch.cat([cart2frac(cart_known_xy.to(device), cell_xy.to(device)), frac_z_known.to(device)], dim=-1) 

class SC_TruncatedHexagonal(SC_Base):  
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        theta = Pi/12
        cos_th = math.cos(theta)
        cell_xy = torch.tensor([[math.cos(Pi/6), math.sin(Pi/6)], 
                                [math.cos(5*Pi/6), math.sin(5*Pi/6)]])*self.bond_len/math.tan(theta)
        self.a_scale, self.b_scale = 1/math.tan(theta), 1/math.tan(theta)  
        self.cell = self.get_cell(gamma=120)
        # coords
        cart_known_xy = torch.tensor([[self.a_len/(2*math.sqrt(2)*cos_th), self.a_len/(2*math.sqrt(2)*cos_th)], 
                                [self.a_len/(2*math.sqrt(2)*cos_th), self.a_len/(2*math.sqrt(2)*cos_th) + self.bond_len],
                                [math.cos(5*theta)*self.a_len/(2*cos_th), math.sin(5*theta)*self.a_len/(2*cos_th)],
                                [-self.a_len/(2*math.sqrt(2)*cos_th), self.a_len/(2*math.sqrt(2)*cos_th)], 
                                [-self.a_len/(2*math.sqrt(2)*cos_th), self.a_len/(2*math.sqrt(2)*cos_th) + self.bond_len],
                                [-math.cos(5*theta)*self.a_len/(2*cos_th), math.sin(5*theta)*self.a_len/(2*cos_th)]])
        self.num_known = cart_known_xy.shape[0]
        frac_z_known = self.frac_z*torch.ones((self.num_known, 1))
        self.frac_known = torch.cat([cart2frac(cart_known_xy.to(device), cell_xy.to(device)), frac_z_known.to(device)], dim=-1)  


class SC_GreatRhombotrihexagonal(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        cell_xy = torch.tensor([[1, 0.], [math.cos(2*Pi/3), math.sin(2*Pi/3)]]) * (3+math.sqrt(3))*self.bond_len
        self.a_scale, self.b_scale = cell_xy.norm(dim=-1)/self.bond_len  
        self.cell = self.get_cell(gamma=120)
        # coords
        vec_mirror = cell_xy.mean(dim=0)
        line_eq = vector_to_line_equation(vec_mirror, torch.tensor([[0., 0.]]))[0]
        six_angles = torch.linspace(0, 2 * torch.pi, steps=7)[:-1]  # [0, 60, 120, ..., 300] degrees
        # Coordinates calculation
        cart_x = self.bond_len * torch.cos(six_angles)
        cart_y = self.bond_len * torch.sin(six_angles)
        cart_known_xy_half = torch.stack((cart_x, cart_y), dim=1) + torch.tensor([[1, 1/math.sqrt(3)]]) * self.bond_len * (3+math.sqrt(3))/2
        cart_known_xy_another = reflect_across_line(cart_known_xy_half, line_eq)
        cart_known_xy = torch.cat([cart_known_xy_half, cart_known_xy_another], dim=0)
        self.num_known = cart_known_xy.shape[0]
        frac_z_known = self.frac_z*torch.ones((self.num_known, 1))
        self.frac_known = torch.cat([cart2frac(cart_known_xy.to(device), cell_xy.to(device)), frac_z_known.to(device)], dim=-1)  


class SC_Lieb(SC_Base):
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 2, 2 
        self.cell = self.get_cell(gamma=90)
        # coords
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z], 
                                        [0.5, 0.0, self.frac_z], 
                                        [0.0, 0.5, self.frac_z]]) 
        self.num_known = self.frac_known.shape[0]


class SC_Template(SC_Base):
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 1, 1   #TODO: set the lattice scaling wrt self.bond_len.
        self.cell = self.get_cell(gamma=90) #TODO: set the lattice matrix by giving the lattice angle \gamma.
        # coords
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z],
                                        [0.5, 0.0, self.frac_z]])   #TODO: set the fractional coordinates of the constrained atoms.
        self.num_known = self.frac_known.shape[0]
        

sc_dict = {'tri': SC_Triangular, 'hon': SC_Honeycomb, 'kag': SC_Kagome, 
           'sqr': SC_Square, 'elt': SC_ElongatedTriangular, 'sns': SC_SnubSquare, 
           'tsq': SC_TruncatedSquare, 'srt': SC_SmallRhombotrihexagonal, 'snh': SC_SnubHexagonal, 
           'trh': SC_TruncatedHexagonal,'grt': SC_GreatRhombotrihexagonal, 'lieb': SC_Lieb, 'van': SC_Vanilla}  

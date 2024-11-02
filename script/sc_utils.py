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

train_dist_sc = {
'van' : [0.0,
        0.0021742334905660377,
        0.021079009433962265,
        0.019826061320754717,
        0.15271226415094338,
        0.047132959905660375,
        0.08464770047169812,
        0.021079009433962265,
        0.07808814858490566,
        0.03434551886792453,
        0.0972877358490566,
        0.013303360849056603,
        0.09669811320754718,
        0.02155807783018868,
        0.06522700471698113,
        0.014372051886792452,
        0.06703272405660378,
        0.00972877358490566,
        0.053176591981132074,
        0.010576356132075472,
        0.08995430424528301],
'tri': [0,
  0,
  0.27976617542482696,
  0.18398577356544843,
  0.09961655807660827,
  0.09173682450408367,
  0.07830267550370268,
  0.046561762800609766,
  0.027236217962024422,
  0.01477959545221042,
  0.023658637557058975,
  0.014134828033249452,
  0.03773552163394306,
  0.012445304805906426,
  0.0034482758620689668,
  0.03505025808347467,
  0.009742602103203721,
  0.017811798278984048,
  0.012205581157194063,
  0.0,
  0.011781609195402302],
 'hon': [0,
  0,
  0.0,
  0.011901340996168581,
  0.013508953637303303,
  0.07316288731220491,
  0.14961327590541218,
  0.0744828246496879,
  0.07046304937734285,
  0.08768540945015275,
  0.0582871832627904,
  0.053438928055368096,
  0.07504594388561865,
  0.052706780598791546,
  0.05820011712456049,
  0.05324575100369833,
  0.05275083797159338,
  0.03345461179907757,
  0.03720681379498239,
  0.02399540954819115,
  0.02084988162705537],
 'kag': [0,
  0,
  0.0,
  0.0,
  0.0,
  0.004054535651107737,
  0.05023929916896083,
  0.08739338135846973,
  0.1472328631885327,
  0.04146417102943967,
  0.0472659766799984,
  0.05298803967628555,
  0.08094517809856433,
  0.059573355847999186,
  0.06564522755333242,
  0.07075417091906858,
  0.09049741414944491,
  0.05945299022842424,
  0.05482479147593927,
  0.046092709132356025,
  0.04157589584207628],
'sqr': [0,
  0,
  0.4197183539323242,
  0.08764353883132335,
  0.11969685822347349,
  0.09965747365922015,
  0.0763204982549282,
  0.04895452076345298,
  0.04641620820146202,
  0.023476958191950295,
  0.009586957948799582,
  0.013414114103900929,
  0.011234117507001751,
  0.007579585534621293,
  0.01224732839376951,
  0.004434013680003058,
  0.0106645613094886,
  0.003329590625852641,
  0.0014228967242567155,
  0.0015275141175260507,
  0.0026749099966450565],
'elt': [0,
  0,
  0,
  0.0188944388944389,
  0.061352806235574084,
  0.0511045740382378,
  0.211606937773541,
  0.11553449103105988,
  0.07378929146847563,
  0.057445591643075465,
  0.06563997382716409,
  0.0462598800360905,
  0.07304609674224623,
  0.05684855778355591,
  0.0436575452759212,
  0.042930726134119204,
  0.03133299760348942,
  0.02313885250598938,
  0.014988086636961976,
  0.007215957571466534,
  0.005213194798593199],
'sns': [0,
  0,
  0,
  0,
  0,
  0,
  0.07271154555531055,
  0.013686049996212757,
  0.05442344685705121,
  0.031244375975719857,
  0.055699860427078265,
  0.05005132315137785,
  0.0733361082208602,
  0.06539555220053461,
  0.08807012694705156,
  0.10712503988835606,
  0.11157868973997938,
  0.08285480426817048,
  0.07683580977490156,
  0.056802394583352525,
  0.06018487241404317],
'tsq': [0,
  0,
  0,
  0,
  0,
  0,
  0,
  0.000689655172413793,
  0.043561242547590265,
  0.0494201085084339,
  0.036116819784593156,
  0.046812713429578975,
  0.08169028098296557,
  0.07014592609775866,
  0.10665668220478712,
  0.0850450535484922,
  0.10330246911027059,
  0.10117561312898285,
  0.09359310249907848,
  0.08758846635466414,
  0.09420186663039029],
'srt': [0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0.004633189897270612,
  0.024305153753921782,
  0.03833441684749035,
  0.08781865302396812,
  0.18187895660994433,
  0.2308715195845152,
  0.12383072251502081,
  0.06498226728985446,
  0.07205343331438231,
  0.06319156426920752,
  0.03383469294272217,
  0.07426542995170213],
'snh': [0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0.032558768518375816,
  0.0398545685993813,
  0.07916089165626333,
  0.19792499600101693,
  0.08072135334100095,
  0.23016958461834078,
  0.07431944320123082,
  0.060312917951507804,
  0.04750652761228614,
  0.04465673096978262,
  0.050501718948393066,
  0.06231249858242057],
'trh': [0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0.004341736694677871,
  0.031127450980392157,
  0.04543657579310793,
  0.12538535763142467,
  0.18885699628688685,
  0.15779407693309883,
  0.16104752947690704,
  0.15840682203113804,
  0.12760345417236663],
'grt': [0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0.5,
  0.0,
  0.5],
'lieb': [0,
  0,
  0,
  0,
  0,
  0.0023809523809523807,
  0.006565790657330224,
  0.019866595449638563,
  0.07099025971967896,
  0.027679830883144275,
  0.026862955329578657,
  0.035801664997206696,
  0.06742944709631779,
  0.06975682015772929,
  0.10456278405231008,
  0.13445094743592248,
  0.15571622479328928,
  0.07637253487080846,
  0.06761414094648893,
  0.06786164310939982,
  0.06608740812020415]
} 


import torch

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

hexagonal_angles = [90, 90, 120]
square_angles = [90, 90, 90]


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
        self.num_atom = int(num_atom)
        self.type_known = type_known
        self.frac_z = frac_z
        self.c_vec_cons = c_vec_cons
        self.reduced_mask = reduced_mask
        self.device = device
        self.a_scale, self.b_scale, self.c_scale = None, None, None     # Initialize lattice scaling wrt self.bond_len
        self.mask_l = self.get_mask_l()    #TODO: write a function
        # print(f"Config for unit cell's c-vector: {self.c_vec_cons}")
        # print(f"The lattice mask: {self.mask_l}")
        self.frac_known = None  # Initialize known fractional coordinates
        self.num_known = None   # Initialize number of known atoms

    def get_cell(self, angles):
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
        mask = mask[:, 0].flatten()   #TODO: mask dimension must be (N, 3), which was transformed  from (N,) in the original code
        return fcoords, mask

    def atm_types_all(self):
        # self.num_atom, self.num_known, self.type_known
        types_idx_known = [chemical_symbols.index(self.type_known)] * self.num_known
        types_unk = random.choices(chemical_symbols[1:MAX_ATOMIC_NUM+1], k=int(self.num_atom-self.num_known))
        types_idx_unk = [chemical_symbols.index(elem) for elem in types_unk]    # list of unknown atom types (randomly chosen)
        types = torch.tensor(types_idx_known + types_idx_unk)
        mask = torch.zeros_like(types)
        mask[:self.num_known] = 1
        return types, mask


class SC_Vanilla(SC_Base):
    """
    Vanilla case with no constraints
    """
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 1, 1 
        self.cell = self.get_cell(square_angles)
        # coords
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z]]) 
        self.num_known = self.frac_known.shape[0]
        self.frac_coords, self.mask_x = self.frac_coords_all()
        # atom types
        self.atom_types, self.mask_t = self.atm_types_all()
        self.mask_x = torch.zeros_like(self.mask_x)
        self.mask_l = torch.zeros_like(self.mask_l)
        self.mask_t = torch.zeros_like(self.mask_t)
        
class SC_Triangular(SC_Base):
    """
    Triangular lattice
    """
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 1, 1 
        self.cell = self.get_cell(hexagonal_angles)
        # coords
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z]]) 
        self.num_known = self.frac_known.shape[0]
        self.frac_coords, self.mask_x = self.frac_coords_all()
        # atom types
        self.atom_types, self.mask_t = self.atm_types_all()

class SC_Honeycomb(SC_Base):
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = math.sqrt(3), math.sqrt(3) 
        self.cell = self.get_cell(hexagonal_angles)
        # coords
        self.frac_known = torch.tensor([[1/3, 2/3, self.frac_z], [2/3, 1/3, self.frac_z]]) 
        self.num_known = self.frac_known.shape[0]
        self.frac_coords, self.mask_x = self.frac_coords_all()
        # atom types
        self.atom_types, self.mask_t = self.atm_types_all()

class SC_Kagome(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 2, 2 
        self.cell = self.get_cell(hexagonal_angles)
        # coords
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z], [0.5, 0.0, self.frac_z], [0.0, 0.5, self.frac_z]])  
        self.num_known = self.frac_known.shape[0]
        self.frac_coords, self.mask_x = self.frac_coords_all()
        # atom types
        self.atom_types, self.mask_t = self.atm_types_all()

class SC_Square(SC_Base):
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 1, 1  
        self.cell = self.get_cell(square_angles)
        # coords
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z]]) 
        self.num_known = self.frac_known.shape[0]
        self.frac_coords, self.mask_x = self.frac_coords_all()
        # atom types
        self.atom_types, self.mask_t = self.atm_types_all()

class SC_ElongatedTriangular(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 1, math.sqrt((1/2)**2 + (1 + math.sqrt(3)/2)**2)  
        gamma_deg = math.degrees(math.atan((1 + math.sqrt(3)/2)/(1/2))) 
        self.cell = self.get_cell([90, 90, gamma_deg])  
        # coords
        cart_known_xy = torch.tensor([[0., 0.], 
                            [self.bond_len/2, self.bond_len*math.sqrt(3)/2]]).to(self.device)
        self.num_known = cart_known_xy.shape[0]
        cell_xy = self.cell[:2, :2]
        frac_z_known = self.frac_z*torch.ones((self.num_known, 1))
        self.frac_known = torch.cat([cart2frac(cart_known_xy.to(device), cell_xy.to(device)), frac_z_known.to(device)], dim=-1)    #TODO: need to set
        self.frac_coords, self.mask_x = self.frac_coords_all()
        # atom types
        self.atom_types, self.mask_t = self.atm_types_all()



class SC_SnubSquare(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        cell_vert0_xy = torch.tensor([[math.cos(Pi/12), -math.sin(Pi/12)]])*self.bond_len/math.sqrt(2)
        cell_vert1_2_xy = torch.tensor([[math.sqrt(3)*math.cos(Pi/6), math.sqrt(3)*math.sin(Pi/6)], 
                                [math.cos(Pi*2/3), math.sin(Pi*2/3)]])*(1+math.sqrt(3))*self.bond_len/2
        cell_xy = cell_vert1_2_xy - cell_vert0_xy
        self.a_scale, self.b_scale = cell_xy.norm(dim=-1)/self.bond_len  
        self.cell = self.get_cell(square_angles)
        # coords
        cart_known_xy = torch.tensor([[math.cos(Pi/6), math.sin(Pi/6)],
                                 [0, 1],
                                 [math.cos(Pi/3), 1 + math.sin(Pi/3)],
                                 [math.cos(Pi/6) + math.cos(Pi/3), math.sin(Pi/6) + math.sin(Pi/3)]])*self.bond_len - cell_vert0_xy
        cart_known_xy = cart_known_xy.to(self.device)
        self.num_known = cart_known_xy.shape[0]
        frac_z_known = self.frac_z*torch.ones((self.num_known, 1))
        self.frac_known = torch.cat([cart2frac(cart_known_xy.to(device), cell_xy.to(device)), frac_z_known.to(device)], dim=-1)  
        self.frac_coords, self.mask_x = self.frac_coords_all()
        # atom types
        self.atom_types, self.mask_t = self.atm_types_all()


class SC_TruncatedSquare(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 1+math.sqrt(2), 1+math.sqrt(2) 
        self.cell = self.get_cell(square_angles)
        # coords
        x = 1/(2+math.sqrt(2))
        self.frac_known = torch.tensor([[x, 0.0, self.frac_z],
                                        [1-x, 0.0, self.frac_z],
                                        [0.0, x, self.frac_z],
                                        [0.0, 1-x, self.frac_z]])
        self.num_known = self.frac_known.shape[0]
        self.frac_coords, self.mask_x = self.frac_coords_all()
        # atom types
        self.atom_types, self.mask_t = self.atm_types_all()


class SC_SmallRhombotrihexagonal(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        cell_xy = torch.tensor([[math.cos(Pi/6), math.sin(Pi/6)], 
                                [math.cos(5*Pi/6), math.sin(5*Pi/6)]])*self.bond_len*(1+math.sqrt(3))
        self.a_scale, self.b_scale = cell_xy.norm(dim=-1)/self.bond_len  
        self.cell = self.get_cell(hexagonal_angles)
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
        self.frac_coords, self.mask_x = self.frac_coords_all()
        # atom types
        self.atom_types, self.mask_t = self.atm_types_all()


class SC_SnubHexagonal(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        cell_xy = torch.tensor([[4., 2*math.sqrt(3)], 
                                [-5., math.sqrt(3)]])*self.bond_len/2
        self.a_scale, self.b_scale = cell_xy.norm(dim=-1)/self.bond_len  
        self.cell = self.get_cell(hexagonal_angles)
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
        self.frac_coords, self.mask_x = self.frac_coords_all()
        # atom types
        self.atom_types, self.mask_t = self.atm_types_all()

class SC_TruncatedHexagonal(SC_Base):  
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        theta = Pi/12
        cos_th = math.cos(theta)
        cell_xy = torch.tensor([[math.cos(Pi/6), math.sin(Pi/6)], 
                                [math.cos(5*Pi/6), math.sin(5*Pi/6)]])*self.bond_len/math.tan(theta)
        self.a_scale, self.b_scale = 1/math.tan(theta), 1/math.tan(theta)  
        self.cell = self.get_cell(hexagonal_angles)
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
        self.frac_coords, self.mask_x = self.frac_coords_all()
        # atom types
        self.atom_types, self.mask_t = self.atm_types_all()



class SC_GreatRhombotrihexagonal(SC_Base): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        cell_xy = torch.tensor([[1, 0.], [math.cos(2*Pi/3), math.sin(2*Pi/3)]]) * (3+math.sqrt(3))*self.bond_len
        self.a_scale, self.b_scale = cell_xy.norm(dim=-1)/self.bond_len  
        self.cell = self.get_cell(hexagonal_angles)
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
        self.frac_coords, self.mask_x = self.frac_coords_all()
        # atom types
        self.atom_types, self.mask_t = self.atm_types_all()


class SC_Lieb(SC_Base):
    def __init__(self, bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, c_vec_cons, reduced_mask, device)
        # Lattice 
        self.a_scale, self.b_scale = 2, 2 
        self.cell = self.get_cell(square_angles)
        # coords
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z], [0.5, 0.0, self.frac_z], [0.0, 0.5, self.frac_z]]) 
        self.num_known = self.frac_known.shape[0]
        self.frac_coords, self.mask_x = self.frac_coords_all()
        # atom types
        self.atom_types, self.mask_t = self.atm_types_all()


al_dict = {'tri': SC_Triangular, 'hon': SC_Honeycomb, 'kag': SC_Kagome, 
           'sqr': SC_Square, 'elt': SC_ElongatedTriangular, 'sns': SC_SnubSquare, 
           'tsq': SC_TruncatedSquare, 'srt': SC_SmallRhombotrihexagonal, 'snh': SC_SnubHexagonal, 
           'trh': SC_TruncatedHexagonal,'grt': SC_GreatRhombotrihexagonal, 'lieb': SC_Lieb, 'van': SC_Vanilla}  
num_known_dict = {'tri': 1, 'hon': 2, 'kag': 3, 'sqr': 1, 'elt': 2, 'sns': 4, 
                  'tsq': 4, 'srt': 6, 'snh': 6, 'trh': 6, 'grt': 12,'lieb': 3, 'van': 1} 

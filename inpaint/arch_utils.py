import torch
import math
import random
from sample import chemical_symbols
from inpcrysdiff.pl_modules.diffusion_w_type import MAX_ATOMIC_NUM 
 

arch_nickname = {
'triangular': 'tri',
'honeycomb': 'hon', 
'kagome': 'kag',
'4_6_12':	'grt',
'3p3_4p2':	'elt',
'3p2_4_3_4':	'sns',
'3p4_6':	'snh',
'3_4_6_4':	'srt',
'3_12p2':	'trh',
'4_8p2':	'tsq',
'lieb':	'lieb',
}

lattice_types = {'kagome': 'hexagonal', 'honeycomb': 'hexagonal', 'triangular': 'hexagonal'}


train_dist_arch = {'triangular': [0,
  0,
  0.21379310344827587,
  0.2045977011494253,
  0.1103448275862069,
  0.11724137931034483,
  0.09655172413793103,
  0.059770114942528735,
  0.034482758620689655,
  0.01839080459770115,
  0.02528735632183908,
  0.016091954022988506,
  0.02528735632183908,
  0.009195402298850575,
  0.0022988505747126436,
  0.029885057471264367,
  0.006896551724137931,
  0.013793103448275862,
  0.009195402298850575,
  0.0,
  0.006896551724137931],
 'honeycomb': [0,
  0,
  0.0,
  0.0074866310160427805,
  0.006951871657754011,
  0.041176470588235294,
  0.10855614973262032,
  0.06042780748663101,
  0.07112299465240642,
  0.07219251336898395,
  0.058823529411764705,
  0.06577540106951872,
  0.08663101604278074,
  0.06042780748663101,
  0.07593582887700535,
  0.06844919786096257,
  0.0679144385026738,
  0.040641711229946524,
  0.045989304812834225,
  0.032620320855614976,
  0.028877005347593583],
 'kagome': [0,
  0,
  0.0,
  0.0,
  0.0,
  0.0024295432458697765,
  0.033527696793002916,
  0.06462585034013606,
  0.13945578231292516,
  0.04664723032069971,
  0.04907677356656948,
  0.04616132167152575,
  0.06899902818270165,
  0.057337220602526724,
  0.06754130223517979,
  0.07531584062196307,
  0.09329446064139942,
  0.07240038872691934,
  0.07045675413022352,
  0.05782312925170068,
  0.05490767735665695], 
 'square': [0,
  0,
  0.3369876072449952,
  0.0776930409914204,
  0.13536701620591038,
  0.1306005719733079,
  0.08674928503336511,
  0.06005719733079123,
  0.05862726406101049,
  0.028122020972354625,
  0.014775977121067683,
  0.012869399428026692,
  0.012392755004766444,
  0.009532888465204958,
  0.009532888465204958,
  0.00667302192564347,
  0.009532888465204958,
  0.003336510962821735,
  0.0023832221163012394,
  0.0023832221163012394,
  0.0023832221163012394],
'4_6_12': [0,
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
'3p3_4p2': [0,
  0,
  0,
  0.0041249263406010605,
  0.03771361225692398,
  0.04890984089569829,
  0.09781968179139658,
  0.10312315851502651,
  0.08308780200353565,
  0.0854449027695934,
  0.07307012374779022,
  0.06835592221567471,
  0.08426635238656452,
  0.08367707719505009,
  0.06069534472598703,
  0.05598114319387154,
  0.04065998821449617,
  0.030642309958750738,
  0.025338833235120803,
  0.0076605774896876845,
  0.009428403064230996],
'3p2_4_3_4': [],
'3p4_6': [],
'3_4_6_4': [],
'3_12p2': [0,
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
  0.009478672985781991,
  0.014218009478672985,
  0.04265402843601896,
  0.08530805687203792,
  0.13270142180094788,
  0.0995260663507109,
  0.21800947867298578,
  0.1943127962085308,
  0.2037914691943128],
'4_8p2': [],
'lieb': [0,
  0,
  0,
  0,
  0,
  0.001122334455667789,
  0.004863449307893753,
  0.01833146277590722,
  0.06322484100261878,
  0.027684249906472128,
  0.026936026936026935,
  0.03367003367003367,
  0.06584362139917696,
  0.07482229704451927,
  0.10475121586232697,
  0.1305649083426861,
  0.14852225963337073,
  0.07631874298540965,
  0.07631874298540965,
  0.07482229704451927,
  0.0722035166479611]
} 



def hexagonal_cell(a, c, device):
    l1 = torch.tensor([a, 0, 0], dtype=torch.float, device=device)
    l2 = torch.tensor([a * -0.5, a * math.sqrt(3) / 2, 0], dtype=torch.float, device=device)
    l3 = torch.tensor([0, 0, c], dtype=torch.float, device=device)
    lattice_matrix = torch.stack([l1, l2, l3], dim=0)
    return lattice_matrix

def square_cell(a, c, device):
    l1 = torch.tensor([a, 0, 0], dtype=torch.float, device=device)
    l2 = torch.tensor([0, a, 0], dtype=torch.float, device=device)
    l3 = torch.tensor([0, 0, c], dtype=torch.float, device=device)
    lattice_matrix = torch.stack([l1, l2, l3], dim=0)
    return lattice_matrix

def frac_coords_all(n_atm, frac_known):
    fcoords_zero = torch.zeros(int(n_atm), 3)
    n_kn = frac_known.shape[0]
    fcoords, mask = fcoords_zero.clone(), fcoords_zero.clone()
    fcoords[:n_kn, :] = frac_known
    mask[:n_kn, :] = torch.ones_like(frac_known) 
    mask = mask[:, 0].flatten()
    return fcoords, mask

def atm_types_all(n_atm, n_kn, type_known):
    types_idx_known = [chemical_symbols.index(type_known)] * n_kn
    types_unk = random.choices(chemical_symbols[1:MAX_ATOMIC_NUM+1], k=int(n_atm-n_kn))
    types_idx_unk = [chemical_symbols.index(elem) for elem in types_unk]
    types = torch.tensor(types_idx_known + types_idx_unk)
    mask = torch.zeros_like(types)
    mask[:n_kn] = 1
    return types, mask

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

class AL_Template():
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        self.bond_len = bond_len
        self.num_atom = num_atom
        self.type_known = type_known
        self.frac_z = frac_z
        self.device = device
        self.mask_l = torch.tensor([[1,1,0]]) 


class AL_Triangular(AL_Template):
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 1
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z]])
        self.a_len = 1 * self.bond_len
        self.c_len = 1 * self.bond_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = hexagonal_cell(self.a_len, self.c_len, self.device)


class AL_Honeycomb(AL_Template):
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 2
        self.frac_known = torch.tensor([[1/3, 2/3, self.frac_z], [2/3, 1/3, self.frac_z]])
        self.a_len = math.sqrt(3) * self.bond_len
        self.c_len = math.sqrt(3) * self.bond_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = hexagonal_cell(self.a_len, self.c_len, self.device)

class AL_Kagome(AL_Template):
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 3
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z], [0.5, 0.0, self.frac_z], [0.0, 0.5, self.frac_z]])
        self.a_len = 2 * self.bond_len
        self.c_len = 2 * self.bond_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = hexagonal_cell(self.a_len, self.c_len, self.device)

class AL_Square(AL_Template):
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 1
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z]])
        self.a_len = 1 * self.bond_len
        self.c_len = 1 * self.bond_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = square_cell(self.a_len, self.c_len, self.device)

class AL_4_8p2(AL_Template): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 4
        x = 1/(2+math.sqrt(2))
        self.frac_known = torch.tensor([[x, 0.0, self.frac_z],
                                        [1-x, 0.0, self.frac_z],
                                        [0.0, x, self.frac_z],
                                        [0.0, 1-x, self.frac_z]])
        self.a_len = (1+math.sqrt(2))*self.bond_len
        self.c_len = 1 * self.bond_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = square_cell(self.a_len, self.c_len, self.device)

class AL_3p3_4p2(AL_Template): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 2
        a = bond_len
        lattice3d =  torch.tensor([[a, 0., 0.], [a/2, a*(1+math.sqrt(3)/2), 0.], [0., 0., a]])
        lattice2d = lattice3d[:2, :2]
        coords2d = torch.tensor([[0., 0.], 
                                 [a/2, a*math.sqrt(3)/2]])
        self.frac_known = torch.cat([cart2frac(coords2d, lattice2d), self.frac_z*torch.ones((self.num_known, 1))], dim=-1)
        self.c_len = a
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = lattice3d

class AL_3p2_4_3_4(AL_Template): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 4
        lat_p0 = torch.tensor([[math.cos(math.pi/12), -math.sin(math.pi/12)]])*bond_len/math.sqrt(2)
        lat_p12 = torch.tensor([[math.sqrt(3)*math.cos(math.pi/6), math.sqrt(3)*math.sin(math.pi/6)], 
                                [math.cos(math.pi*2/3), math.sin(math.pi*2/3)]])*(1+math.sqrt(3))*bond_len/2
        lattice2d =  lat_p12 - lat_p0
        coords2d = torch.tensor([[math.cos(math.pi/6), math.sin(math.pi/6)],
                                 [0, 1],
                                 [math.cos(math.pi/3), 1 + math.sin(math.pi/3)],
                                 [math.cos(math.pi/6) + math.cos(math.pi/3), math.sin(math.pi/6) + math.sin(math.pi/3)]])*bond_len - lat_p0
        # self.lat2d = lattice2d
        # self.cart2d = coords2d
        self.frac_known = torch.cat([cart2frac(coords2d, lattice2d), self.frac_z*torch.ones((self.num_known, 1))], dim=-1)
        self.a_len = lattice2d.norm(dim=-1).mean()
        self.c_len = self.a_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = square_cell(self.a_len, self.c_len, self.device)

class AL_3p4_6(AL_Template): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 6
        lattice2d = torch.tensor([[4., 2*math.sqrt(3)], 
                                [-5., math.sqrt(3)]])*bond_len/2
        coords2d = torch.tensor([[1., math.sqrt(3)], 
                                [2., 2*math.sqrt(3)],
                                [-1., math.sqrt(3)],
                                [0., 2*math.sqrt(3)],
                                [-3., math.sqrt(3)],
                                [-2, 2*math.sqrt(3)]])*bond_len/2
        # self.lat2d = lattice2d
        # self.cart2d = coords2d
        self.frac_known = torch.cat([cart2frac(coords2d, lattice2d), self.frac_z*torch.ones((self.num_known, 1))], dim=-1)
        self.a_len = lattice2d.norm(dim=-1).mean()
        self.c_len = self.a_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = hexagonal_cell(self.a_len, self.c_len, self.device)

class AL_4_6_12(AL_Template): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 12
        s = bond_len
        a = s * (3+math.sqrt(3))
        lattice2d = torch.tensor([[a, 0.], [a*math.cos(2*math.pi/3), a*math.sin(2*math.pi/3)]])
        vec_flip = lattice2d.mean(dim=0)
        line = vector_to_line_equation(vec_flip, torch.tensor([[0., 0.]]))[0]
        angles = torch.linspace(0, 2 * torch.pi, steps=7)[:-1]  # [0, 60, 120, ..., 300] degrees
        # Coordinates calculation
        x = s * torch.cos(angles)
        y = s * torch.sin(angles)
        # Stack coordinates in pairs
        coords = torch.stack((x, y), dim=1) + torch.tensor([[0.5*a, 0.5*a/math.sqrt(3)]])
        coords_refl = reflect_across_line(coords, line)
        self.frac_known = torch.cat([cart2frac(torch.cat([coords, coords_refl], dim=0), lattice2d), self.frac_z*torch.ones((self.num_known, 1))], dim=-1)
        self.a_len = a
        self.c_len = a
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = hexagonal_cell(self.a_len, self.c_len, self.device)


class AL_3_4_6_4(AL_Template): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 6
        lattice2d = torch.tensor([[math.sqrt(3), 1.], 
                                [-math.sqrt(3), 1.]])*bond_len*(1+math.sqrt(3))/2
        coords2d = torch.tensor([[1., math.sqrt(3)], 
                                [1+math.sqrt(3), 1+math.sqrt(3)],
                                [1, 2+math.sqrt(3)],
                                [-1., math.sqrt(3)], 
                                [-1-math.sqrt(3), 1+math.sqrt(3)],
                                [-1, 2+math.sqrt(3)]])*bond_len/2
        # self.lat2d = lattice2d
        # self.cart2d = coords2d
        self.frac_known = torch.cat([cart2frac(coords2d, lattice2d), self.frac_z*torch.ones((self.num_known, 1))], dim=-1)
        self.a_len = lattice2d.norm(dim=-1).mean()
        self.c_len = self.a_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = hexagonal_cell(self.a_len, self.c_len, self.device)

class AL_3_12p2(AL_Template): 
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 6
        theta = math.pi/12
        cos_th = math.cos(theta)
        a = bond_len/math.tan(theta)
        lattice2d = torch.tensor([[math.sqrt(3), 1.], 
                                [-math.sqrt(3), 1.]])*a/2
        coords2d = torch.tensor([[a/(2*math.sqrt(2)*cos_th), a/(2*math.sqrt(2)*cos_th)], 
                                [a/(2*math.sqrt(2)*cos_th), a/(2*math.sqrt(2)*cos_th) + bond_len],
                                [math.cos(5*theta)*a/(2*cos_th), math.sin(5*theta)*a/(2*cos_th)],
                                [-a/(2*math.sqrt(2)*cos_th), a/(2*math.sqrt(2)*cos_th)], 
                                [-a/(2*math.sqrt(2)*cos_th), a/(2*math.sqrt(2)*cos_th) + bond_len],
                                [-math.cos(5*theta)*a/(2*cos_th), math.sin(5*theta)*a/(2*cos_th)]])
        # self.lat2d = lattice2d
        # self.cart2d = coords2d
        self.frac_known = torch.cat([cart2frac(coords2d, lattice2d), self.frac_z*torch.ones((self.num_known, 1))], dim=-1)
        self.a_len = a #lattice2d.norm(dim=-1).mean()
        self.c_len = self.a_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = hexagonal_cell(self.a_len, self.c_len, self.device)

class AL_Lieb(AL_Template):
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 3
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z], [0.5, 0.0, self.frac_z], [0.0, 0.5, self.frac_z]])
        self.a_len = 2 * self.bond_len
        self.c_len = 2 * self.bond_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = square_cell(self.a_len, self.c_len, self.device)


al_dict = {'triangular': AL_Triangular, 'honeycomb': AL_Honeycomb, 'kagome': AL_Kagome, 'square': AL_Square, '4_8p2': AL_4_8p2, '3p3_4p2': AL_3p3_4p2, '3p2_4_3_4': AL_3p2_4_3_4, '3p4_6': AL_3p4_6, '4_6_12': AL_4_6_12, '3_4_6_4': AL_3_4_6_4, '3_12p2': AL_3_12p2,'lieb': AL_Lieb}   #!!
num_known_dict = {'triangular': 1, 'honeycomb': 2, 'kagome': 3, 'square': 1, '4_8p2': 4, '3p3_4p2': 2, '3p2_4_3_4': 4, '3p4_6': 6, '4_6_12': 12, '3_4_6_4': 6, '3_12p2': 6,'lieb': 3}  #!

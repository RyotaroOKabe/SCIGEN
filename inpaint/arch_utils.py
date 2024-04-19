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


train_dist_arch = {
'triangular': [0,
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
 'honeycomb': [0,
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
 'kagome': [0,
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
'square': [0,
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
'3p2_4_3_4': [0,
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
'3p4_6': [0,
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
'3_4_6_4': [0,
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
  0.004341736694677871,
  0.031127450980392157,
  0.04543657579310793,
  0.12538535763142467,
  0.18885699628688685,
  0.15779407693309883,
  0.16104752947690704,
  0.15840682203113804,
  0.12760345417236663],
'4_8p2': [0,
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

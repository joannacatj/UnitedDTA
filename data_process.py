from gensim.models import Word2Vec
from rdkit.Chem import Draw
from tqdm import tqdm
import os
import pickle
import timeit

import deepchem
import numpy as np
import pandas as pd
import torch
import dgl
from rdkit import Chem
from scipy import sparse as sp
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import distances
from itertools import product, groupby, permutations
from scipy.spatial import distance_matrix
from dgl import load_graphs
import warnings

warnings.filterwarnings("ignore")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model = Word2Vec.load(r"/mnt/sdb/home/hjy/Summary-DTA/word2vec_Davis.model")
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64
MAX_SMI_LEN=85#不足的补零，多余的截断
#处理为1d embedding
# (one_hot格式，适用于CNN环节)
#处理为64个key和最大长度设置为85
#这里输入的是一个smiles分子,一个原子的维度：[最大长度，key长度],即[85,64]
def one_hot_smiles(line,MAX_SMI_LEN,smi_ch_ind):
    X=np.zeros((MAX_SMI_LEN,len(smi_ch_ind)))
    for i,ch in enumerate(line[:MAX_SMI_LEN]):
        X[i,(smi_ch_ind[ch]-1)]=1
    return np.array(X)
#处理为2D embedding,即邻接矩阵和特征矩阵
#常用方法，把数字变为one-hot格式
def smiles_to_mol(smiles):
	try:
		mol=Chem.MolFromSmiles(smiles)
	except:
		raise RuntimeError("SMILES cannot been parsed!")
	return mol
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]
#特征矩阵，维度为[分子的原子个数，特征维度数]
#使用rdkit提取的特征
def atom_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(17),degree(7),formal charge(1),
    radical electrons(1),hybridization(6),aromatic(1),hydrogen atoms attached(5),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']  # 17-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']  # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [
                  atom.GetIsAromatic()]  # 17+7+2+6+1=33

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])  # 33+5=38
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 38+3 =41
    return results
def get_atoms_fea(smiles,explicit_H=False, use_chirality=True):
	mol=smiles_to_mol(smiles)
	num_atoms=mol.GetNumAtoms()
	atom_feats = np.array([atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
	if use_chirality:
		chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
												  useLegacyImplementation=False)
		chiral_arr = np.zeros([num_atoms, 3])
		for (i, rs) in chiralcenters:
			if rs == 'R':
				chiral_arr[i, 0] = 1
			elif rs == 'S':
				chiral_arr[i, 1] = 1
			else:
				chiral_arr[i, 2] = 1
		atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)
	return np.array(atom_feats),num_atoms
#处理为三维embedding（适用于Graph Transformer)
#获取bond feature
def bond_features(bond, use_chirality=True):
    """Generate bond features including bond type(4), conjugated(1), in ring(1), stereo(4)"""
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)
#输出的格式是【边数，边的特征】
#输出的是list,list和ndarray
def get_bond_fea(smiles,use_chirality=True):
    mol=smiles_to_mol(smiles)
    num_bonds = mol.GetNumBonds()
    src_list = []
    dst_list = []
    bond_feats_all = []
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_feats = bond_features(bond, use_chirality=use_chirality)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)
    return src_list,dst_list,np.array(bond_feats_all)
#获取位置信息
#输出的依旧是graph,但是加了位置encode的信息
def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g
#获取dgl格式的图
def smiles_to_graph(smiles,explicit_H=False,use_chirality=True):
    g=dgl.DGLGraph()
    atom_fea,num_atoms=get_atoms_fea(smiles)
    g.add_nodes(num_atoms)
    g.ndata["feats"]=torch.tensor(atom_fea)
    src,dst,bond_fea=get_bond_fea(smiles)
    g.add_edges(src,dst)
    g.edata["feats"]=torch.tensor(bond_fea)
    g=laplacian_positional_encoding(g,pos_enc_dim=8)
    return g

def Smiles2Img(smis, size=224):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png
    '''
    try:
        mol = Chem.MolFromSmiles(smis)
        img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
        return img
    except:
        return None

METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO", "PD", "AG", "CR", "FE", "V",
         "MN", "HG", 'GA',
         "CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", "CE", "PR", "ND",
         "GD", "TB", "DY", "ER",
         "TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH"]
RES_MAX_NATOMS = 24
#1D embedding
#one-hot表示方法，处理为25个key和最大长度设置为1200，适用于CNN
CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
				"U": 19, "T": 20, "W": 21,
				"V": 22, "Y": 23, "X": 24,
				"Z": 25 }

CHARPROTLEN = 25
MAX_SEQ_LEN=1200#不足的补零，多余的截断
#一个原子的维度：[最大长度，key长度],即[1200,25]即[seq_len,input_dim]
def one_hot_protein(line,MAX_SMI_LEN,smi_ch_ind):
    X=np.zeros((MAX_SMI_LEN,len(smi_ch_ind)))
    for i,ch in enumerate(line[:MAX_SMI_LEN]):
        X[i,(smi_ch_ind[ch]-1)]=1
    return np.array(X)

#word2vec方法，这里是需要进行预训练的！！
def seq_to_kmers(seq, k=3):
    """ Divide a string into a list of kmers strings.

    Parameters:
        seq (string)
        k (int), default 3
    Returns:
        List containing a list of kmers.
    """
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]
class Corpus(object):
    """ An iteratable for training seq2vec models. """

    def __init__(self,data, ngram):
        self.df = data
        self.ngram = ngram

    def __iter__(self):
        for no, data in enumerate(self.df):
            yield  seq_to_kmers(data,self.ngram)
def get_protein_word2vec(model,protein,maxlen):
    """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
    vec = np.zeros((maxlen,100))
    i = 0
    for word in protein[:maxlen]:
        vec[i, ] = model.wv[word]
        i += 1
    vec=torch.from_numpy(vec)
    return np.array(vec)
#获取图的方法
def obtain_self_dist(res):
    try:
        # xx = res.atoms.select_atoms("not name H*")
        xx = res.atoms
        dists = distances.self_distance_array(xx.positions)
        ca = xx.select_atoms("name CA")
        c = xx.select_atoms("name C")
        n = xx.select_atoms("name N")
        o = xx.select_atoms("name O")
        return [dists.max() * 0.1, dists.min() * 0.1, distances.dist(ca, o)[-1][0] * 0.1,
                distances.dist(o, n)[-1][0] * 0.1, distances.dist(n, c)[-1][0] * 0.1]
    except:
        return [0, 0, 0, 0, 0]


def obtain_dihediral_angles(res):
    try:
        if res.phi_selection() is not None:
            phi = res.phi_selection().dihedral.value()
        else:
            phi = 0
        if res.psi_selection() is not None:
            psi = res.psi_selection().dihedral.value()
        else:
            psi = 0
        if res.omega_selection() is not None:
            omega = res.omega_selection().dihedral.value()
        else:
            omega = 0
        if res.chi1_selection() is not None:
            chi1 = res.chi1_selection().dihedral.value()
        else:
            chi1 = 0
        return [phi * 0.01, psi * 0.01, omega * 0.01, chi1 * 0.01]
    except:
        return [0, 0, 0, 0]


def calc_res_features(res):
    return np.array(one_of_k_encoding_unk(obtain_resname(res),
                                          ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR',
                                           'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP',
                                           'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                                           'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +  # 32  residue type
					obtain_self_dist(res) +  # 5
					obtain_dihediral_angles(res)  # 4
					)


def obtain_resname(res):
    if res.resname[:2] == "CA":
        resname = "CA"
    elif res.resname[:2] == "FE":
        resname = "FE"
    elif res.resname[:2] == "CU":
        resname = "CU"
    else:
        resname = res.resname.strip()

    if resname in METAL:
        return "M"
    else:
        return resname


##'FE', 'SR', 'GA', 'IN', 'ZN', 'CU', 'MN', 'SR', 'K' ,'NI', 'NA', 'CD' 'MG','CO','HG', 'CS', 'CA',

def obatin_edge(u, cutoff=10.0):
    edgeids = []
    dismin = []
    dismax = []
    for res1, res2 in permutations(u.residues, 2):
        dist = calc_dist(res1, res2)
        if dist.min() <= cutoff:
            edgeids.append([res1.ix, res2.ix])
            dismin.append(dist.min() * 0.1)
            dismax.append(dist.max() * 0.1)
    return edgeids, np.array([dismin, dismax]).T


def check_connect(u, i, j):
    if abs(i - j) != 1:
        return 0
    else:
        if i > j:
            i = j
        nb1 = len(u.residues[i].get_connections("bonds"))
        nb2 = len(u.residues[i + 1].get_connections("bonds"))
        nb3 = len(u.residues[i:i + 2].get_connections("bonds"))
        if nb1 + nb2 == nb3 + 1:
            return 1
        else:
            return 0


def calc_dist(res1, res2):

    dist_array = distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    return dist_array


def load_protein(protpath, explicit_H=False, use_chirality=True):

    mol = Chem.MolFromPDBFile(protpath, sanitize=True, removeHs=True, flavor=0, proximityBonding=False)
    if use_chirality:
        Chem.AssignStereochemistryFrom3D(mol)
    return mol


def prot_to_graph(id, cutoff=10.0):
    """obtain the residue graphs"""
    prot_pdb = 'data/pdb/' + id + ".pdb"
    pk = deepchem.dock.ConvexHullPocketFinder()
    prot = Chem.MolFromPDBFile(prot_pdb, sanitize=True, removeHs=True, flavor=0, proximityBonding=False)
    Chem.AssignStereochemistryFrom3D(prot)
    u = mda.Universe(prot)
    g = dgl.DGLGraph()
    # Add nodes
    num_residues = len(u.residues)
    g.add_nodes(num_residues)
    res_feats = np.array([calc_res_features(res) for res in u.residues])

    g.ndata["feats"] = torch.tensor(res_feats)
    edgeids, distm = obatin_edge(u, cutoff)
    src_list, dst_list = zip(*edgeids)
    g.add_edges(src_list, dst_list)
    g.ndata["ca_pos"] = torch.tensor(np.array([obtain_ca_pos(res) for res in u.residues]))
    g.ndata["center_pos"] = torch.tensor(u.atoms.center_of_mass(compound='residues'))
    dis_matx_ca = distance_matrix(g.ndata["ca_pos"], g.ndata["ca_pos"])
    cadist = torch.tensor([dis_matx_ca[i, j] for i, j in edgeids]) * 0.1
    dis_matx_center = distance_matrix(g.ndata["center_pos"], g.ndata["center_pos"])
    cedist = torch.tensor([dis_matx_center[i, j] for i, j in edgeids]) * 0.1
    edge_connect = torch.tensor(np.array([check_connect(u, x, y) for x, y in zip(src_list, dst_list)]))
    g.edata["feats"] = torch.cat([edge_connect.view(-1, 1), cadist.view(-1, 1), cedist.view(-1, 1), torch.tensor(distm)], dim=1)
    g.ndata.pop("ca_pos")
    g.ndata.pop("center_pos")

    ca_pos = np.array(np.array([obtain_ca_pos(res) for res in u.residues]))

    pockets = pk.find_pockets(prot_pdb)
    for bound_box in pockets:
        x_min = bound_box.x_range[0]
        x_max = bound_box.x_range[1]
        y_min = bound_box.y_range[0]
        y_max = bound_box.y_range[1]
        z_min = bound_box.z_range[0]
        z_max = bound_box.z_range[1]
        idxs = []
        for idx in range(ca_pos.shape[0]):
            if x_min < ca_pos[idx][0] < x_max and y_min < ca_pos[idx][1] < y_max and z_min < ca_pos[idx][2] < z_max:
                idxs.append(idx)

    g_pocket = dgl.node_subgraph(g, idxs)
    g_pocket = laplacian_positional_encoding(g_pocket, pos_enc_dim=8)


    return g_pocket


def obtain_ca_pos(res):
    if obtain_resname(res) == "M":
        return res.atoms.positions[0]
    else:
        try:
            pos = res.atoms.select_atoms("name CA").positions[0]
            return pos
        except:  ##some residues loss the CA atoms
            return res.atoms.positions.mean(axis=0)


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    #A = g.adjacency_matrix().astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g

def Compound_data_construction(id,compound_values,dir_output):
    for no,data in enumerate(tqdm(id)):
        compounds_g=list()
        smile = compound_values[no]
        one_hot_embedding = one_hot_smiles(smile, MAX_SMI_LEN, CHARISOSMISET)
        #adj_matrix = get_adj_matrix(smile)
        #feature_embedding, num = get_atoms_fea(smile)
        dir = dir_output + 'compound_embedding/' + str(data) + '.npy'
        np.save(dir, one_hot_embedding)
        compound_graph = smiles_to_graph(smile)
        compounds_g.append(compound_graph)
        dgl.save_graphs(dir_output + 'compound_graph/' + str(data) + '.bin', list(compounds_g))
        img = Smiles2Img(smile)
        dir = dir_output + 'compound_image/' + str(data) + '.png'
        img.save(dir)

def Protein_data_construction(id,sequence,dir_output):
    for no,data in enumerate(tqdm(id)):
        #print(data)
        seq=sequence[no]
        one_hot_embedding=one_hot_protein(seq,MAX_SEQ_LEN,CHARPROTSET)
        word2vec_embedding=get_protein_word2vec(model,seq_to_kmers(seq),MAX_SEQ_LEN)
        dir=dir_output+'protein_embedding/'+str(data)+'.npz'
        np.savez(dir,one_hot_embedding,word2vec_embedding)
        #proteins_g = list()
        #rotein_graph=prot_to_graph(data)
        #proteins_g.append(protein_graph)
        #dgl.save_graphs(dir_output + 'protein_graph/' + str(data) + '.bin', list(proteins_g))

if __name__ == '__main__':

    dataset = 'Davis'
    file_path_protein = 'data/' + dataset + '/' + dataset +  '_compound_mapping.csv'
    dir_output = ('data/' + dataset + '/predata/')
    os.makedirs(dir_output, exist_ok=True)
    '''
    raw_data_protein = pd.read_csv(file_path_protein)
    protein_id_unique = raw_data_protein['COMPOUND_ID'].values
    protein_seq_unique=raw_data_protein['COMPOUND_SMILES'].values
    N = len(protein_id_unique)
    Compound_data_construction(id=protein_id_unique,compound_values=protein_seq_unique,dir_output=dir_output)
    '''
    file_path_protein = 'data/' + dataset + '/' + dataset + '_protein_mapping.csv'
    raw_data_protein = pd.read_csv(file_path_protein)
    protein_id_unique = raw_data_protein['PROTEIN_ID'].values
    protein_seq_unique = raw_data_protein['PROTEIN_SEQUENCE'].values
    N = len(protein_id_unique)
    Protein_data_construction(id=protein_id_unique,sequence=protein_seq_unique,dir_output=dir_output)

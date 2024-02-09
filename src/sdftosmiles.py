import pandas as pd
from pathlib import Path
from rdkit import Chem

data_path = '../data/'

for phase in ['training', 'validation', 'test']:
    print('__________________________________________________')
    print('__________________________________________________')
    print('__________________________________________________')
    print(phase)
    data_path = Path(data_path)
    smiles = []
    df = pd.read_csv(data_path / f"{phase}_mol.csv")
    smiles = list(df['smiles'])
    pdbs = list(df['pdbid'])
    compound_iso_smiles = []
    deal_pdb = []
    smile_graph = {}
    for index in range(len(smiles)):
        smile = smiles[index]
        pdb = pdbs[index]
        # mol2 = Chem.MolFromSmiles(smile, sanitize=True)
        mol = Chem.MolFromMolFile('../data/' + phase + '/sdfchange/' + pdb + '.sdf')
        # , sanitize = True, removeHs = True
        if mol is None:
            print(pdb)
        if index%1000 == 0 :
            print('index:', index)
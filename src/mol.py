import shutil
import pandas as pd
from pathlib import Path
import os
from rdkit import Chem

wrong = []
for phase in ['training', 'validation', 'test', 'test105']:
# for phase in ['test71']:
    print(phase)
    data_path = '../data/'
    data_path = Path(data_path)
    df = pd.read_csv(data_path / f"{phase}_smi.csv")
    pdbids = list(df['pdbid'])
    smiles = list(df['smiles'])
    i = 0
    j = 0
    for index in range(len(smiles)):
        smile = smiles[index]
        mol = Chem.MolFromSmiles(smile, sanitize=True)
        if mol is None:
            i = i + 1
            wrong.append(phase + pdbids[index])



    # for pdbid in pdbids:
    #     src_path = "F:/downloads/PDBbind_v2020_other_PL/v2020-other-PL/" + pdbid + "/" + pdbid + "_ligand.mol2"
    #     src_path2016 = "F:/downloads/pdbbind_v2016_general-set-except-refined/general-set-except-refined/" + pdbid + "/" + pdbid + "_ligand.mol2"
    #     src_path2016else = "F:/downloads/pdbbind_v2016_refined/refined-set/" + pdbid + "/" + pdbid + "_ligand.mol2"
    #
    #
    #     src_path2020 = "F:/downloads/PDBbind_v2020_mol2/mol2/" + pdbid + "_ligand.mol2"
    #
    #     dst_path = "F:/RPIStudy/GraphDTAF-master-try/data/" + phase + "/mol2/" + pdbid + ".mol2"
    #
    #     # shutil.copy(src_path, dst_path)
    #     if os.path.exists(src_path):
    #
    #         shutil.copy(src_path, dst_path)
    #
    #     elif os.path.exists(src_path2016):
    #
    #         shutil.copy(src_path2016, dst_path)
    #
    #     elif os.path.exists(src_path2016else):
    #
    #         shutil.copy(src_path2016else, dst_path)
    #
    #     elif os.path.exists(src_path2020):
    #
    #         shutil.copy(src_path2020, dst_path)
    #
    #     else :
    #         print(pdbid + "不存在")
    #         wrong.append(phase + pdbid)
    #         j = j + 1
    #
    #     i = i + 1
    #     if i%1000 == 0 :
    #         print('i:', i)

    # for pdbid in pdbids:
    #     src_path = "F:/downloads/PDBbind2020_ligand_sdf/PDBbind2020_ligand_sdf/" + pdbid + "_ligand.sdf"
    #
    #     dst_path = "F:/RPIStudy/GraphDTAF-master-try/data/" + phase + "/sdfchange/" + pdbid + ".sdf"
    #
    #     if os.path.exists(src_path):
    #
    #         shutil.copy(src_path, dst_path)
    #
    #     else :
    #         print(pdbid + "不存在")
    #         wrong.append(phase + pdbid)
    #         j = j + 1
    #
    #     i = i + 1
    #     if i%1000 == 0 :
    #         print('i:', i)


    print('i:', i)
    # print('j:',j)
    print('wrong:', wrong)
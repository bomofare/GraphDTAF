import os
import pandas as pd
from pathlib import Path
import shutil

# os.listdir()方法获取文件夹名字，返回数组
# def getAllFiles(targetDir):
#     listFiles = os.listdir(targetDir)
#     return listFiles
#
# files = getAllFiles(r"F:\RPIStudy\GraphDTAF-master-try\data\validation\sdfchange")

# 写入list到txt文件中
with open(r"../../GraphDTAF-master-try-1010/src/training.txt", 'w+', encoding='utf-8')as f:
    # 列表生成式
    # f.writelines([str(i).split('.')[0]+"\n" for i in files])
    # pdblist = []
    # for i in files :
    #     pdblist.append(str(i).split('.')[0])
    # print(pdblist)
    data_path = '../../GraphDTAF-master-try/data/'
    data_path = Path(data_path)
    df = pd.read_csv(data_path / f"test_mol.csv")
    smiles = list(df['smiles'])
    pdbs = list(df['pdbid'])
    df2 = pd.read_csv(data_path / f"affinity_data.csv")
    pdbids = list(df2['pdbid'])
    affinitys = list(df2['-logKd/Ki'])
    df3 = pd.read_csv(data_path / f"test_seq_.csv")
    ids = list(df3['id'])
    seqs = list(df3['seq'])
    outsmiles = []
    outaffinity = []
    outseq = []
    outid = []
    for index in range(len(smiles)):
        smile = smiles[index]
        pdb = pdbs[index]
        outsmiles.append(smile)
        outid.append(pdb)
        for index2 in range(len(affinitys)):
            affinity = affinitys[index2]
            pdbid = pdbids[index2]
            if pdbid == pdb:
                outaffinity.append(affinity)
                break
        for index3 in range(len(seqs)):
            seq = seqs[index3]
            id = ids[index3]
            if id == pdb:
                outseq.append(seq)
                break
    #     if pdb in pdblist :
    #         outsmiles.append(smile)
    #         outpdbs.append(pdb)
    #         shutil.copy(os.path.join(data_path/'training'/'sdfchange', f"{pdb}.sdf"), os.path.join(data_path/'validation'/'sdfchange', f"{pdb}.sdf"))
    #     # print(len(outpdbs))
    # for index in range(len(smiles2)):
    #     smile = smiles2[index]
    #     pdb = pdbs2[index]
    #     if pdb in pdblist :
    #         outsmiles.append(smile)
    #         outpdbs.append(pdb)
    #
    print(len(outsmiles))
    csv_data = {'pdbid': outid,'compound_iso_smiles': outsmiles,'target_sequence':outseq ,'affinity':outaffinity}
    # print(csv_data(Name))
    df = pd.DataFrame(csv_data)
    df.to_csv(data_path / f"datatest.csv", index=False)

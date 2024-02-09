from openbabel import openbabel
import pandas as pd
from pathlib import Path

for phase in ['training', 'validation', 'test', 'test71', 'test105']:
    print(phase)
    data_path = '../data/'
    data_path = Path(data_path)
    df = pd.read_csv(data_path / f"{phase}_smi.csv")
    pdbids = list(df['pdbid'])
    i = 0

    for pdbid in pdbids:
        print('pdbid:', pdbid)
        inputfile = "F:/RPIStudy/GraphDTAF-master-try/data/" + phase + "/mol2/" + pdbid + ".mol2"
        outputfile = "F:/RPIStudy/GraphDTAF-master-try/data/" + phase + "/mol/" + pdbid + ".mol"
        conv = openbabel.OBConversion()  # 使用openbabel模块中的OBConversion函数，用于文件格式转换的
        conv.OpenInAndOutFiles(inputfile, outputfile)  # 输入需要转换的文件的名字，以及定义转换后文件的文件名
        conv.SetInAndOutFormats("mol2", "mol")  # 定义转换文件前后的格式
        conv.Convert()  # 执行转换操作
        conv.CloseOutFile()  # 转换完成后关闭转换后的文件，完成转换
        i = i + 1
        if i % 1000 == 0:
            print('i:', i)

    print('i:', i)



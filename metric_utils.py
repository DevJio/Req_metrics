import numpy as np
import random
import pandas as pd
import sklearn.metrics as skm
from sklearn.metrics import confusion_matrix

cat_short3 = {   8: '0',
                16: '3А-сроки',
                18: '3А-к/р',
                10: '3A+сроки',
                11: '3А+реш',
                26: '3А-некор',
                29: '3А-вилка',
                31: '3Б+Аналог',
                34: '3А-оцен',
                37: '3Б',
                40: '3А-выбор',
                56: '4А-ЗТ',
                61: '3В',
                62: '4Б-ЗлПр',
                65: '3Ж-способ', 
                79: '3Ж-проц_реш',
                89: '4А+м_вед'
            }

labels=[8, 16, 18, 10, 26, 29, 34, 40, 11, 37, 31, 61, 65, 79, 56, 89, 62]
#print(labels)
catNames = [cat_short3[key] for key in labels]
catNames

def splitColValue(value):
    return list(map(int, value.split(sep=', ')))
        
def makeCheckSet(dFrameExpertCol):
    check_set = dict()
    splited_col = dFrameExpertCol.apply(splitColValue)
    for values in splited_col:
        for cat in values:
            if cat in check_set:
                check_set[cat] += 1
            else:
                check_set[cat] = 1
    return check_set

def prepateExpertCol(dFrameExpertCol):
    check_set=makeCheckSet(dFrameExpertCol)
    splited_col = dFrameExpertCol.apply(splitColValue)
    result = np.zeros((dFrameExpertCol.shape[0]), dtype=int)
    
    drop_map = { 8:8, 10:10, 11:11,
    16:16, 18:18,               19:10,  21:10, 
    26:26, 29:29, 34:34, 40:40,
    27:11, 28:11, 35:11, 
    37:37,                    31:31,
    61:61,
    65:65, 79:79,
    56:56, 89:89,
    62:62 }
    
    cat_list = list(set(drop_map.keys()))
    
    for ind, values in enumerate(splited_col):
        rez=0
        for cat in values:
            #print('cat', cat)
            if cat in cat_list:
                #print(check_set[cat])
                rez=int(cat)
                continue
            else: 
                if rez ==0:
                    rez = 8
        result[ind] = rez
    rez = pd.DataFrame(result, columns=['m_id'])
    rez = rez.m_id.map(drop_map)
    print(rez)
    return rez

def make_conf_matrix(expert, predict):
    conf_matrix = confusion_matrix(expert, predict, labels=labels )
    c_matr = pd.DataFrame(conf_matrix, columns=labels, index = labels)
    c_matr
    return c_matr


def make_cl_report(expert, predict):
    report = skm.classification_report(expert, predict, labels=labels, target_names=labels, output_dict=True)
    rep = pd.DataFrame(report)
    rep = rep.T[['precision', 'recall', 'f1-score', 'support']]
    return rep
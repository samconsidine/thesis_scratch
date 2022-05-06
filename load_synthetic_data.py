""" Code from VITAE https://github.com/jaydu1/VITAE/blob/master/VITAE/utils.py """


import h5py
import numpy as np
import pandas as pd
import os
import anndata
from anndata import AnnData

from typing import List


type_dict = {
    # dyno
    'dentate':'UMI', 
    'immune':'UMI', 
    'neonatal':'UMI', 
    'mouse_brain':'UMI', 
    'mouse_brain_miller':'UMI',
    'mouse_brain_merged':'UMI',
    'planaria_full':'UMI', 
    'planaria_muscle':'UMI',
    'aging':'non-UMI', 
    'cell_cycle':'non-UMI',
    'fibroblast':'non-UMI', 
    'germline':'non-UMI',    
    'human_embryos':'non-UMI', 
    'mesoderm':'non-UMI',
    
    # dyngen
    "linear_1":'non-UMI', 
    "linear_2":'non-UMI', 
    "linear_3":'non-UMI',
    'bifurcating_1':'non-UMI',
    'bifurcating_2':'non-UMI',
    "bifurcating_3":'non-UMI', 
    "cycle_1":'non-UMI', 
    "cycle_2":'non-UMI', 
    "cycle_3":'non-UMI',
    "trifurcating_1":'non-UMI', 
    "trifurcating_2":'non-UMI',         
    "converging_1":'non-UMI',
    
    # our model
    'linear':'UMI',
    'bifurcation':'UMI',
    'multifurcating':'UMI',
    'tree':'UMI',
}


def load_data(path, file_name):  
    '''Load h5df data.
    Parameters
    ----------
    path : str
        The path of the h5 files.
    file_name : str
        The dataset name.
    
    Returns:
    ----------
    data : dict
        The dict containing count, grouping, etc. of the dataset.
    '''     
    data = {}
    
    with h5py.File(os.path.join(path, file_name+'.h5'), 'r') as f:
        data['count'] = np.array(f['count'], dtype=np.float32)
        data['grouping'] = np.array(f['grouping']).astype(str)
        if 'gene_names' in f:
            data['gene_names'] = np.array(f['gene_names']).astype(str)
        else:
            data['gene_names'] = None
        if 'cell_ids' in f:
            data['cell_ids'] = np.array(f['cell_ids']).astype(str)
        else:
            data['cell_ids'] = None
            
        if 'milestone_network' in f:
            if file_name in ['linear','bifurcation','multifurcating','tree',                              
                            "cycle_1", "cycle_2", "cycle_3",
                            "linear_1", "linear_2", "linear_3", 
                            "trifurcating_1", "trifurcating_2", 
                            "bifurcating_1", 'bifurcating_2', "bifurcating_3", 
                            "converging_1"]:
                data['milestone_network'] = pd.DataFrame(
                    np.array(np.array(list(f['milestone_network'])).tolist(), dtype=str), 
                    columns=['from','to','w']
                ).astype({'w':np.float32})
            else:
                data['milestone_network'] = pd.DataFrame(
                    np.array(np.array(list(f['milestone_network'])).tolist(), dtype=str), 
                    columns=['from','to']
                )
            data['root_milestone_id'] = np.array(f['root_milestone_id']).astype(str)[0]            
        else:
            data['milestone_net'] = None
            data['root_milestone_id'] = None
            
        if file_name in ['mouse_brain', 'mouse_brain_miller']:
            data['grouping'] = np.array(['%02d'%int(i) for i in data['grouping']], dtype=object)
            data['root_milestone_id'] = dict(zip(['mouse_brain', 'mouse_brain_miller'], ['06', '05']))[file_name]
            data['covariates'] = np.array(np.array(list(f['covariates'])).tolist(), dtype=np.float32)
        if file_name in ['mouse_brain_merged']:
            data['grouping'] = np.array(data['grouping'], dtype=object)
            data['root_milestone_id'] = np.array(f['root_milestone_id']).astype(str)[0]
            data['covariates'] = np.array(np.array(list(f['covariates'])).tolist(), dtype=np.float32)

    data['type'] = type_dict[file_name]
    if data['type']=='non-UMI':
        scale_factor = np.sum(data['count'],axis=1, keepdims=True)/1e6
        data['count'] = data['count']/scale_factor
    
    return data  


def to_anndata(data):
    X = data['count']
    obs = data['cell_ids']
    var = data['gene_names']
    misc_1 = data['grouping']
    misc_2 = data['milestone_network']

    print(f'{misc_1=}')
    print(f'{misc_2=}')

    return AnnData(X=X, obs=obs, var=var)


if __name__ == "__main__":
    data = load_data('data', 'bifurcating_3')
    print(to_anndata(data))


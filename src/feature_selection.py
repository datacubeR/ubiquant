import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from timeit import default_timer as timer
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

import logging

log = logging.getLogger('Feature_Selector')

@hydra.main(config_path="conf", config_name="config")
def feature_selector(cfg: DictConfig):

    tic = timer()
    log.info(f'Starting {cfg.fs.name}')
    #======================================================
    # READ IN DATA
    #======================================================
    
    df = pd.read_feather(to_absolute_path('train.feather')).set_index('row_id')

    #======================================================
    # DOWNCASTING
    #======================================================
    
    FRAC = cfg.fs.frac
    df = df.sample(frac = FRAC)
    log.info(f'{FRAC*100}%: {df.memory_usage(index=True, deep=False).sum()} ')

    int_columns = df.select_dtypes(int).columns 
    df[int_columns] = df[int_columns].astype('int16')

    float_columns = df.select_dtypes(float).columns 
    df[float_columns] = df[float_columns].astype('float32')

    log.info(f'Downcasted: {df.memory_usage(index=True, deep=False).sum()} ')
    
    
    X = df.drop(columns = ['target'])
    y = df.target

    result = hydra.utils.call(cfg.fs.method, X = X, y = y)

    result.to_csv(to_absolute_path(cfg.output_file))
    toc = timer()
    log.info(f'Total elapsed time: {(toc - tic)/60} minutes')


if __name__ == '__main__':

    feature_selector()
    
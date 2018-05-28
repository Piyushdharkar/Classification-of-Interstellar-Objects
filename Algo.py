import pandas as pd
import numpy as np

dataset = pd.read_csv("dataset.csv")

removed_features = ['objid', 'run', 'rerun', 'fiberid',
                    'camcol', 'specobjid', 'plate', 'mjd',
                    'field']

dataset = dataset.drop(removed_features, axis=1)
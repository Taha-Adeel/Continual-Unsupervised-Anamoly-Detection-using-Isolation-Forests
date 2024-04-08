import numpy as np
import pandas as pd
#pre processing data
def preprocess(data):
    for i in data.columns:
        data = data[data[i] != "Infinity"]
        data = data[data[i] != np.nan]
        data = data[data[i] != ",,"]
    data[["Flow Bytes/s", " Flow Packets/s"]] = data[["Flow Bytes/s", " Flow Packets/s"]].apply(
        pd.to_numeric
    )


    # Removing these columns as their value counts are zero
    data.drop([" Bwd PSH Flags"], axis=1, inplace=True)
    data.drop([" Bwd URG Flags"], axis=1, inplace=True)
    data.drop(["Fwd Avg Bytes/Bulk"], axis=1, inplace=True)
    data.drop([" Fwd Avg Packets/Bulk"], axis=1, inplace=True)
    data.drop([" Fwd Avg Bulk Rate"], axis=1, inplace=True)
    data.drop([" Bwd Avg Bytes/Bulk"], axis=1, inplace=True)
    data.drop([" Bwd Avg Packets/Bulk"], axis=1, inplace=True)
    data.drop(["Bwd Avg Bulk Rate"], axis=1, inplace=True)

    # Replacing nans, infs with zero's
    data.replace([np.inf, -np.inf, -np.nan, np.nan], 0, inplace=True)
    print('Data preprocessing done.\n')
    return data
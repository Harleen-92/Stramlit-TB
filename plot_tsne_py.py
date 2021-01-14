from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import seaborn as sns
import os
import numpy as np
import pandas as pd

class PlotTSNE():
    def __init__(self, gendata_path, actualdata_path=None):
        self.data = pd.read_csv(gendata_path)
        self.preprocess(self.data)
        if isinstance(actualdata_path, str):
            self.data2 = pd.read_csv(actualdata_path)
            self.preprocess(self.data2)
        print(self.data.shape)
        print(self.data2.shape)

    def preprocess(self, df):
        if 'Unnamed: 0' in list(df.columns.values):
            df.drop(['Unnamed: 0'], axis=1, inplace=True)

    def plot_tsne(self, name="ud", plotpath=None):
        gendata = self.data
        actual = self.data2
        if gendata.shape[0]!=actual.shape[0]:
            if gendata.shape[0]>actual.shape[0]:
                gendata = gendata.iloc[:actual.shape[0],:]
            else:
                actual = actual.iloc[:gendata.shape[0],:]
        n_patients, n_genes = gendata.shape
        label1 = ["Generated Data"]*n_patients
        label2 = ["Actual Data"]*n_patients
        labels = pd.Series(label1+label2).to_frame()
        dfeatures = pd.concat([gendata, actual], ignore_index=True,
                              axis=0, sort=False)
        X_embedded = TSNE(n_components=2, random_state=0,
                          perplexity=100).fit_transform(dfeatures)
        X_embedded = pd.DataFrame(X_embedded, columns=['dim1', 'dim2'])
        X_embedded = pd.DataFrame(
            np.hstack([np.array(X_embedded), np.array(labels)]))
        X_embedded.columns = ['dim1', 'dim2', 'label']

        sns_fig = sns.lmplot(x='dim1', y='dim2', data=X_embedded, fit_reg=False, hue='label')
        filename = "tsne_plot_"
        filename = filename + name + ".png"
        plt.savefig(filename)
        plt.close()


#if __name__=="__main__":
    # Change this to data path
    # Order of params (Gendata path, actualData path)
    #rd = PlotTSNE("/home/ayushig/TB/WGAN/Group2Data/custom/gendata/gendata_unnorm.csv","/home/ayushig/saad18409/Group2Data.csv")
    #rd.plot_tsne()

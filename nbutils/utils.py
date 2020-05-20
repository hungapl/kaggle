import pandas as pd
from pandas_summary import DataFrameSummary
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def df_show_all():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)


def show_corr_table(df:pd.DataFrame, head=10, target_col=None):
    # Show features with high correlations.  Usually highly correlated features do not improve or worsen a model,
    # If correlation is > 0.8, consider remove one of the feature in the pair to speed up the training
    corr = pd.DataFrame(df.corr().abs().unstack())
    corr = corr.reset_index()
    corr.columns = ['v1', 'v2', 'c']
    corr['ordered-cols'] = corr.apply(lambda x: '-'.join(sorted([x['v1'], x['v2']])), axis=1)
    corr = corr.drop_duplicates(['ordered-cols'])
    corr.drop(['ordered-cols'], axis=1, inplace=True)
    corr = corr.query('v1 != v2').sort_values('c', ascending=False)
    print(corr.head(head))

    if target_col is not None:
        filter = (corr.v1 == target_col) | (corr.v2 == target_col)
        print(corr[filter].head(head))


def show_corr_matrix(df: pd.DataFrame):
    sns_corr = df.corr().abs()
    mask = np.zeros_like(sns_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(sns_corr, mask=mask, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


def show_corr_plot(df:pd.DataFrame, cols=[]):
    sns.set()
    if len(cols) == 0:
        cols=pd.columns
    sns.pairplot(df[cols], height=2.5)
    plt.show()


def show_missing_cols(df:pd.DataFrame):
    df_summ = DataFrameSummary(df)
    # Show columns with missing values
    col_missing_values = (df_summ.columns_stats[
                              df_summ.columns_stats.index == 'missing_perc'].values != '0%').flatten()
    df_missing = df_summ.columns_stats.iloc[:, col_missing_values]
    df_missing = df_missing.sort_values(by='missing_perc', ascending=False, axis=1)
    return df_missing


def categorical_cols(df:pd.DataFrame):
    types_df = pd.DataFrame(df.dtypes).reset_index()
    return types_df[types_df[0] == 'object']['index'].values

def numerical_cols(df:pd.DataFrame):
    types_df = pd.DataFrame(df.dtypes).reset_index()
    return types_df[types_df[0] != 'object']['index'].values

    
def fill_missing_cols(df:pd.DataFrame, cols, value=0):
    for c in cols:
        if c not in df.columns:
            df[c] = value
    return df

def plot_missing():
    import seaborn as sns
    sns.set_style("whitegrid")
    missing = train.isnull().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    missing.plot.bar()
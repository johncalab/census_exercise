# spot highly correlated features
def get_sorted_feats_by_corr(data):
    """
    data is pandas dataframe
    returns corr_vals sorted by abs value
    """
    cols = data.columns
    dupes_to_drop = { (data.columns[i],data.columns[j]) for i in range(data.shape[1]) for j in range(i+1) }
    
    return data.corr().abs().unstack().drop(labels=dupes_to_drop).sort_values(ascending=False)

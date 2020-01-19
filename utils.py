import pandas as pd
# load data from two separate sources
def load_data_from_two(path_to_data,path_to_cols):
    import pandas as pd
    """
    loads data and returns a pandas dataframe
    requires: import pandas as pd
    """
    col_names = []
    with open('census-income.columns', 'r') as f:
        for line in f.readlines():
            col_names.append(line[:-1])

    df = pd.read_csv('census-income.data', names=col_names)
    return df

# fill nans from hisp column
def fill_nan_hisp(data):
    """
    assumes data is a pandas dataframe
    modifies data in place
    """
    assert 'hispanic origin' in data.columns, "Hispanic column missing"
    data['hispanic origin'].fillna(value='All other', inplace=True)
    return None

# encode binary columns as 0/1
def encode_cols(data):
    """
    ad hoc function to create 0/1 column for label
    """
    from sklearn.preprocessing import LabelEncoder
    data['label_encoded'] = LabelEncoder().fit_transform(data['label'])
    data['sex_encoded'] = LabelEncoder().fit_transform(data['sex'])
    data['year_encoded'] = LabelEncoder().fit_transform(data['year'])
    return None

# group features according to type
def bucket_features(data):
    """
    assumes data is a pandas dataframe containing data types object, float, and int.
    returns two sets of column names, according to type float or other
    """
    numerical_types = ['int64', 'float64']
    var_cont = set()
    var_disc = set()
    for col in data.columns:
        if data.dtypes[col] in numerical_types:
            var_cont.add(col)
        else:
            var_disc.add(col)
            
    var_cont.remove('label_encoded')
    var_cont.remove('instance weight')

    to_swap=['veterans benefits',
            'own business or self employed',
            'sex_encoded',
            'year_encoded']

    for col_name in to_swap:
        var_cont.remove(col_name)
        var_disc.add(col_name)

    return (var_cont,var_disc)

# group entries from country column
def clean_country(data):
    def helper(country):
        if country == 'United-States' or country == "Mexico":
            return country
        else:
            return 'Other'
    
    data['country'] = data['country of birth mother'].apply(helper)
    return None
        

def clean_data(df,make_dummies=False):
    """
    assumes df comes from loaded_from_two
    """
    # fill nans from hispanic column
    fill_nan_hisp(df)
    # encode sex and label columns
    encode_cols(df)
    
    # clean country
    clean_country(df)
    
    # drop labels we don't use anymore
    # note: label_encoded and instance_weight are still in df
    df = df.drop(columns=['sex', 'year', 'label'])

    # drop correlated features
    to_drop = ['migration code-change in msa',
                'detailed industry recode',
                'detailed occupation recode',
                'migration code-move within reg',
                'live in this house 1 year ago',
                "fill inc questionnaire for veteran's admin",
                'migration code-change in reg',
                'migration prev res in sunbelt',
                'major industry code',
                'state of previous residence',
                'detailed household and family stat',
                'country of birth self',
                'country of birth father',
                'country of birth mother',
                'citizenship']

    df = df.drop(columns=to_drop)

    # group features by type
    var_cont,var_disc = bucket_features(df)

    if make_dummies:
        df = pd.get_dummies(df, columns=var_disc)

    return (df,var_cont,var_disc)

# load data and pre-process using the functions above
def load_for_training(path_to_data='census-income.data',
                        path_to_columns='census-income.columns',
                        make_dummies=False):
    # load data from files
    df = load_data_from_two('census-income.data', 'census-income.columns')
    return clean_data(df,make_dummies)

def ttsplit(data,strat_data):
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(data,
                                     random_state=7,
                                     test_size=0.20,
                                     shuffle=True,
                                     stratify=strat_data)
    return (df_train,df_test)

def load_ttsplit():
    import pandas as pd
    dftr = pd.read_pickle('train_df.pkl')
    dftst = pd.read_pickle('test_df.pkl')
    return dftr, dftst

def xyw(data):
    x = data.drop(columns=['label_encoded', 'instance weight'])
    y = data['label_encoded']
    w = data['instance weight']
    return (x,y,w)

def evaluate_preds(y_hat,y,w=None,t=0.5):

    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(y,y_hat,sample_weight=w)

    y_pred = (y_hat > t)
    y_pred = y_pred.astype('int')
    
    from sklearn.metrics import f1_score
    f1 = f1_score(y,y_pred,sample_weight=w)

    from sklearn.metrics import precision_score
    precision = precision_score(y,y_pred,sample_weight=w)
    
    from sklearn.metrics import recall_score
    recall = recall_score(y,y_pred,sample_weight=w)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y,y_pred,sample_weight=w)
    
    return {'auc':auc_score, 'f1':f1, 'precision':precision, 'recall':recall, 'accuracy':accuracy}


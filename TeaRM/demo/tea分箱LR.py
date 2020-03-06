from auto_bin_woe import *
from tea_utils import train_by_cv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def compute_ks(prob, target):
    """
    target: numpy array of shape (1,)
    prob: numpy array of shape (1,), predicted probability of the sample being positive
    returns:
    ks: float, ks score estimation
    """
    get_ks = lambda prob, target: ks_2samp(prob[target == 1], prob[target != 1]).statistic

    return get_ks(prob, target)

# -------------------------- 读数据 ---------------------------------
train_set = data
oot_set = data
unfeas = []
# ----------------------- 快速计算IV和PSI ----------------------------
data_matrix = feature_value_info(train_set.drop([i for i in unfeas if i!='is_bad'], axis=1), 
                                label_name='is_bad', bin_split=5)
data_matrix_oot = feature_value_info(oot_set.drop([i for i in unfeas if i!='is_bad'], axis=1), 
                                label_name='is_bad', bin_split=5, oot_dm=data_matrix)

feature_iv = {}
for f in data_matrix_oot:
    feature_iv[f] = [data_matrix[f]['iv'].sum(), data_matrix_oot[f]['iv'].sum()]

iv_df = pd.DataFrame(feature_iv, index=['iv', 'oot_iv']).T.sort_values('iv', ascending=False)
psi = tag_psi(data_matrix, data_matrix_oot)
psi_df = pd.DataFrame(psi, index=['psi']).T.sort_values('psi', ascending=False)
df_info = pd.merge(iv_df, psi_df, how='left', left_index=True, right_index=True)
df_info.index.name = 'feature'
df_info = df_info.reset_index()
describe = get_describe(pd.concat([X_train, X_oot]))
res = describe.merge(df_info, how='left', left_on=['变量名称'], right_on=['feature'])
best_feature = res[(res['空值个数占比']<0.9)
                    &(res['最常值个数占比']<0.9)
                    &(res['iv']>0.03)
                    &(res['psi']<0.1)]

# -------------------------  分箱 -----------------------------------
woe = AutoBinWOE(bins=10, bad_rate_merge=True, bad_rate_sim_threshold=0.1)
# bad_rate_sim_threshold参数调节保留箱数的多少，越高箱越少
woe.fit(X_train, y_train)
X_woe = woe.transform(X_train)
X_oot_woe = woe.transform(X_oot)


# -------------------------  训练  -----------------------------------
sss = StratifiedKFold(n_splits=5, random_state=11, shuffle=True)
clf = LogisticRegression(C=0.1, penalty='l2')
clf, stacking_train, stacking_oot = train_by_cv(X_woe, y_train, X_oot_woe, y_oot, sss, clf)


from TeaML.utils.tea_encoder import *
from TeaML.utils.tea_filter import *
from TeaML.utils.tea_utils import *
from TeaML.utils.auto_bin_woe import *
import TeaML

data = pd.read_csv("TeaML/examples.csv")

# encoder
ct = TeaBadRateEncoder(num=1)
me = TeaMeanEncoder(categorical_features=['city'])
t = TeaOneHotEncoder()
encoder = [me]

# woe & feature selection
woe = TeaML.WOE(bins=10, bad_rate_merge=True, bad_rate_sim_threshold=0.05, psi_threshold=0.1, iv_threshold=None)
iv = FilterIV(200, 100)
vif = FilterVif(50)
mod = FilterModel('lr', 70)
nova = FilterANOVA(40, 30)
coline = FilterCoLine({'penalty': 'l2', 'C': 0.01, 'fit_intercept': True})
fshap = FilterSHAP(70)
outlier = OutlierTransform()
filtercor = FilterCorr(20)
stepwise = FilterStepWise(method='p_value')
method = [woe, stepwise]

# main
tea = TeaML.Tea(['core_lend_request_id', 'lend_customer_id', 'customer_sex',
                'data_center_id', 'trace_back_time', 'mobile', 'user_id', 'id_no', 'task_id', 'id',
                 'id_district_name', 'id_province_name', 'id_city_name', 'pass_time'],
                'is_overdue_M0',
                datetime_feature='pass_time',
                split_method='oot',
                file_path='report.xlsx')
tea.wash(data, null_drop_rate=0.8, most_common_drop_rate=0.9)
tea.cook(encoder)
tea.select(method)
tea.drink(LogisticRegression(penalty='l2', C=1, class_weight='balanced'))
tea.sleep(woe.bins)



data_matrix = feature_value_info(train_set.drop([i for i in unfeas if i!='is_bad'], axis=1), label_name='is_bad', bin_split=5)
data_matrix_oot = feature_value_info(oot_set.drop([i for i in unfeas if i!='is_bad'], axis=1), label_name='is_bad', bin_split=5, oot_dm=data_matrix)

feature_iv = {}
for f in data_matrix_oot:
    feature_iv[f] = [data_matrix[f]['iv'].sum(), data_matrix_oot[f]['iv'].sum()]

iv_df = pd.DataFrame(feature_iv, index=['iv', 'oot_iv']).T.sort_values('iv', ascending=False)

psi = tag_psi(data_matrix, data_matrix_oot)
psi_df = pd.DataFrame(psi, index=['psi']).T.sort_values('psi', ascending=False)
df_info = pd.merge(iv_df, psi_df, how='left', left_index=True, right_index=True)
df_info.index.name = 'feature'
df_info = df_info.reset_index()

describe = get_describe(data_new.drop(unfeas, axis=1))
res = describe.merge(df_info, how='left', left_on=['变量名称'], right_on=['feature'])
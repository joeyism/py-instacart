import gc
import pandabase
import pandas as pd
from pandabase import DataSet
from time import gmtime, strftime
import numpy as np

rint = 0

def p(rint):
    rint +=1
    print("here {}".format(rint))
    return rint

def getAllMemory():
    mem_dict = []
    dic = globals().copy()
    for key, val in dic.items():
        mem_dict.append((sys.getsizeof(val), key))
    del dic, key, val
    return dict(mem_dict)

rint = p(rint)

d = pandabase.readFiles("data/")
d["priors_order_details"] = DataSet(d["orders"].merge(right=d["order_products__prior"], on="order_id"))
d["priors_order_details"].loc[:, "_user_buy_product_times"] = d["priors_order_details"].groupby(["user_id", "product_id"]).cumcount() + 1
agg_dict = {'user_id':{'_prod_tot_cnts':'count'},
           'reordered':{'_prod_reorder_tot_cnts':'sum'},
           '_user_buy_product_times': {'_prod_buy_first_time_total_cnt':lambda x: sum(x==1),
                                       '_prod_buy_second_time_total_cnt':lambda x: sum(x==2)}}
df_new = d["priors_order_details"]
grouped = df_new.groupby(["product_id"])
the_stats = grouped.agg(agg_dict)
the_stats.columns = the_stats.columns.droplevel(0)

the_stats['_prod_reorder_prob'] = the_stats["_prod_buy_second_time_total_cnt"] / the_stats["_prod_buy_first_time_total_cnt"]
the_stats['_prod_reorder_ratio'] = the_stats["_prod_reorder_tot_cnts"] / the_stats["_prod_tot_cnts"]
the_stats['_prod_reorder_times'] = 1 + the_stats["_prod_reorder_tot_cnts"] / the_stats["_prod_buy_first_time_total_cnt"]
the_stats = the_stats.reset_index()

#######
rint = p(rint)
agg_dict_2 = {'order_number':{'_user_total_orders':'max'},
              'days_since_prior_order':{'_user_sum_days_since_prior_order':'sum',
                                        '_user_mean_days_since_prior_order': 'mean'}}

the_stats2 = d["orders"][d["orders"].eval_set == "prior"].groupby(["user_id"]).agg(agg_dict_2)
the_stats2.columns = the_stats2.columns.droplevel(0)
the_stats2 = DataSet(the_stats2)


#######
rint = p(rint)
agg_dict_3 = {'reordered':
              {'_user_reorder_ratio':
               lambda x: sum(d["priors_order_details"].ix[x.index,'reordered']==1)/sum(d["priors_order_details"].ix[x.index,'order_number'] > 1)},
              'product_id':{'_user_total_products':'count',
                            '_user_distinct_products': lambda x: x.nunique()}}

the_stats3 = d["priors_order_details"].groupby(["user_id"]).agg(agg_dict_3)
the_stats3.columns = the_stats3.columns.droplevel(0)
the_stats3 = DataSet(the_stats3)


users = DataSet(the_stats2.merge(the_stats3, how="inner", left_index=True, right_index=True))

users["_user_average_basket"] = users["_user_total_products"]/users["_user_total_orders"]

us = d["orders"].loc[d["orders"]["eval_set"] != "prior", ["user_id", "order_id", "eval_set", "days_since_prior_order"]]
us.rename(index=str, columns={"days_since_prior_order": "time_since_last_order"}, inplace=True)
users = users.merge(us, how="inner", left_index=True, right_on="user_id")

#######
rint = p(rint)
agg_dict_4 = {'order_number':{'_up_order_count': 'count',
                              '_up_first_order_number': 'min',
                              '_up_last_order_number':'max'},
              'add_to_cart_order':{'_up_average_cart_position': 'mean'}}
the_stats4 = d["priors_order_details"].groupby(["user_id", "product_id"]).agg(agg_dict_4)
the_stats4.columns = the_stats4.columns.droplevel(0)
the_stats4 = DataSet(the_stats4)



the_stats4 = the_stats4.reset_index()

data = the_stats4.merge(the_stats, how="inner", on="product_id").merge(users, how="inner", on="user_id")

del the_stats, the_stats2, the_stats3, the_stats4, df_new, grouped
gc.collect()
#######
data['_up_order_rate'] = data._up_order_count / data._user_total_orders
data['_up_order_since_last_order'] = data._user_total_orders - data._up_last_order_number
data['_up_order_rate_since_first_order'] = data._up_order_count / (data._user_total_orders - data._up_first_order_number + 1)


#######
rint = p(rint)
t = d["orders"].merge(d["order_products__prior"], on="order_id")
t["order_number_rev"] = t.groupby("user_id")["order_number"].transform(np.max) - t["order_number"] + 1

t["total_cart_size"] = t.groupby(["user_id", "order_id"])["add_to_cart_order"].transform(np.max)
total_buy_n5 = t[(t["order_number_rev"] > 0) & (t["order_number_rev"] <= 5)].groupby(["user_id", "product_id"]).size().reset_index()
total_buy_n5.columns = ["user_id", "product_id", "total_buy_n5"]

#######
rint = p(rint)
train = d["order_products__train"]
train = train.merge(right=d["orders"][["order_id", "user_id"]], how = "left", on ="order_id")
#train = train.merge(right=total_buy_n5, how="left", on=["user_id", "product_id"]) #
data = data.merge(train[["user_id", "product_id", "reordered"]], on = ["user_id", "product_id"], how="left")


del d
gc.collect()

#######
rint = p(rint)
import xgboost
from sklearn.model_selection import train_test_split
#%matplotlib tk

train = data.loc[data["eval_set"] == "train",:]
X_test = data.loc[data["eval_set"] == "test", :]

train.drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis=1, inplace=True)
train.loc[:, 'reordered'] = train.reordered.fillna(0)

X_train, X_val, y_train, y_val = train_test_split(train.drop("reordered", axis=1), train["reordered"], test_size=1, random_state=25)

d_train = xgboost.DMatrix(X_train, y_train)
xgb_params = {
    "objective"        : "reg:logistic",
    "eval_metric"      : "logloss",
    "eta"              : 0.1,
    "max_depth"        : 10,
    "min_child_weight" : 8,
    "gamma"            : 0.70,
    "subsample"        : 0.76,
    "colsample_bytree" : 0.95,
    "alpha"            : 2e-05,
    "lambda"           : 10
}

bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=100, evals=[(d_train, "train")])
#xgboost.plot_importance(bst)

gc.collect()


#######
sample_submission = pd.read_csv("data/sample_submission.csv")
d_test = xgboost.DMatrix(X_test.drop(["eval_set", "user_id", "order_id", "reordered", "product_id"], axis=1))
X_test.loc[:, "reordered"] = (bst.predict(d_test) > 0.21).astype(int)
X_test.loc[:, "product_id"] = X_test["product_id"].astype(str)
agg_dict_5 = {
    "group_columns_list": ["order_id"],
    "target_columns_list": ["product_id"],
    "methods_list": [lambda x: " ".join(set(x))]
}
grouped_name = "order_id"
target_name = "product_id"
combine_name = [[grouped_name] + [method_name] + [target_name] for method_name in agg_dict_5["methods_list"]]

df_new = X_test[X_test["reordered"] == 1].copy() # get only ones where calculated to be reordered
the_stats5 = df_new.groupby(agg_dict_5["group_columns_list"]).agg(agg_dict_5["methods_list"]).reset_index()
the_stats5.columns = the_stats5.columns.droplevel(1)
the_stats5 = the_stats5.drop("eval_set", axis=1)
the_stats5.columns = sample_submission.columns

submit_final = sample_submission[['order_id']].merge(the_stats5, how='left').fillna('None')
time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
submit_final.to_csv("submission-" + time + ".csv", index=False)

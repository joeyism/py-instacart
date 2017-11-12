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

def groupByAgg(df, groupby_val, agg_dict):
    stats = df.groupby(groupby_val).agg(agg_dict)
    stats.columns = stats.columns.droplevel(0)
    return stats


rint = p(rint)

d = pandabase.readFiles("data/")
d["orders_details__prior"] = DataSet(d["orders"].merge(right=d["order_products__prior"], on="order_id"))
d["orders_details__prior"].loc[:, "no_of_times_user_bought_item"] = d["orders_details__prior"].groupby(["user_id", "product_id"]).cumcount() + 1



#######
rint = p(rint)
orders_prior_dict = {'order_number':{'total_orders':'max'},
              'days_since_prior_order':{'total_days_between_orders':'sum',
                                        'avg_days_between_orders': 'mean'}}
orders_prior_agg = groupByAgg(d["orders"][d["orders"]["eval_set"] == "prior"], ["user_id"], orders_prior_dict)


#######
rint = p(rint)
orders_details_prior_dict = {'reordered':
              {'reorder_ratio':
               lambda x: sum(d["orders_details__prior"].ix[x.index,'reordered']==1)/sum(d["orders_details__prior"].ix[x.index,'order_number'] > 1)},
              'product_id':{'total_products':'count',
                            'distinct_products': lambda x: x.nunique()}}
orders_details_prior_agg = groupByAgg(d["orders_details__prior"], ["user_id"], orders_details_prior_dict)


users_agg = orders_prior_agg.merge(orders_details_prior_agg, how="inner", left_index=True, right_index=True)

users_agg["average_no_item_per_order"] = users_agg["total_products"]/users_agg["total_orders"]

us = d["orders"].loc[d["orders"]["eval_set"] != "prior", ["user_id", "order_id", "eval_set", "days_since_prior_order"]]
us.rename(index=str, columns={"days_since_prior_order": "time_since_last_order"}, inplace=True)
users_agg = users_agg.merge(us, how="inner", left_index=True, right_on="user_id")

del us, orders_prior_agg, orders_details_prior_agg
gc.collect()

#######
rint = p(rint)

agg_dict = {'user_id':{'no_purchased':'count'},
           'reordered':{'no_reordered':'sum'},
           'no_of_times_user_bought_item': {'no_bought_first_time':lambda x: sum(x==1),
                                       'no_bought_second_time':lambda x: sum(x==2)}}
product_agg_prior = groupByAgg(d["orders_details__prior"], ["product_id"], agg_dict)

product_agg_prior['reorder_prob'] = product_agg_prior["no_bought_second_time"] / product_agg_prior["no_bought_first_time"]
product_agg_prior['reorder_ratio'] = product_agg_prior["no_reordered"] / product_agg_prior["no_purchased"]
product_agg_prior['no_times_reordered'] = 1 + product_agg_prior["no_reordered"] / product_agg_prior["no_bought_first_time"]
product_agg_prior = product_agg_prior.reset_index()


rint = p(rint)
user_product_prior_dict = {'order_number':{'no_of_orders': 'count',
                              'order_number_of_first_purchase': 'min',
                              'order_number_of_last_purchase':'max'},
              'add_to_cart_order':{'average_order_number': 'mean'}}
user_product_prior_agg = d["orders_details__prior"].groupby(["user_id", "product_id"]).agg(user_product_prior_dict)
user_product_prior_agg.columns = user_product_prior_agg.columns.droplevel(0)



user_product_prior_agg = user_product_prior_agg.reset_index()

data = user_product_prior_agg.merge(product_agg_prior, how="inner", on="product_id").merge(users_agg, how="inner", on="user_id")

del product_agg_prior, user_product_prior_agg, df_new, grouped
gc.collect()
#######
data['_up_order_rate'] = data.no_of_orders / data.total_orders
data['_up_order_since_last_order'] = data.total_orders - data.order_number_of_last_purchase
data['_up_order_rate_since_first_order'] = data.no_of_orders / (data.total_orders - data.order_number_of_first_purchase + 1)


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

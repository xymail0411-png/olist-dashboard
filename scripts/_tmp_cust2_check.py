# -*- coding: utf-8 -*-
import pandas as pd
BASE = r"C:\Users\董心雅\Desktop\数据分析\work\data\olist"
cust = pd.read_csv(f"{BASE}\\olist_customers_dataset.csv")
orders = pd.read_csv(f"{BASE}\\olist_orders_dataset.csv")
print("customers 行数:", len(cust))
print("customer_id 唯一:", cust["customer_id"].nunique())
print("customer_unique_id 唯一:", cust["customer_unique_id"].nunique())
print("orders 行数:", len(orders), " orders.customer_id 唯一:", orders["customer_id"].nunique())
# 用 unique_id 看订单是否有复购
oc = orders.merge(cust, on="customer_id", how="left")
print("orders 里 customer_unique_id 唯一数:", oc["customer_unique_id"].nunique())
print("orders 中 unique_id 出现>1次的客户数(即复购):", (oc["customer_unique_id"].value_counts() > 1).sum())

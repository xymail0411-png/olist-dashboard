# -*- coding: utf-8 -*-
import pandas as pd
om = pd.read_csv(r"C:\Users\董心雅\Desktop\数据分析\work\data\processed\orders_master.csv")
print("orders_master 行数:", len(om))
print("customer_id 唯一数:", om["customer_id"].nunique())
dup = om["customer_id"].value_counts()
print("出现>1次的客户数:", (dup > 1).sum())
print("其中复购(>=2单)客户数:", (dup >= 2).sum())
print("客户最大单数:", dup.max())

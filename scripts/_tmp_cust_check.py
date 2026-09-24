# -*- coding: utf-8 -*-
import pandas as pd
BASE = r"C:\Users\董心雅\Desktop\数据分析\work\data\olist"
orders = pd.read_csv(f"{BASE}\\olist_orders_dataset.csv")
print("orders 行数:", len(orders))
print("customer_id 唯一数:", orders["customer_id"].nunique())
dup = orders["customer_id"].value_counts()
print("出现>1次的客户数:", (dup > 1).sum())
print("重复客户最大单数:", dup.max())
print("\n示例重复客户(customer_id -> 单数):")
print(dup[dup > 1].head().to_string() if (dup > 1).any() else "无重复，所有客户仅1单")

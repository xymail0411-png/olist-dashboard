# -*- coding: utf-8 -*-
"""用原始 olist 9 表算一遍描述/诊断 KPI 权威值，供用户对照 Tableau。"""
import pandas as pd
from datetime import datetime

BASE = r"C:\Users\董心雅\Desktop\数据分析\work\data\olist"
orders = pd.read_csv(f"{BASE}\\olist_orders_dataset.csv")
items = pd.read_csv(f"{BASE}\\olist_order_items_dataset.csv")
customers = pd.read_csv(f"{BASE}\\olist_customers_dataset.csv")
reviews = pd.read_csv(f"{BASE}\\olist_order_reviews_dataset.csv")
payments = pd.read_csv(f"{BASE}\\olist_order_payments_dataset.csv")

for c in ["order_purchase_timestamp", "order_delivered_customer_date",
          "order_estimated_delivery_date"]:
    orders[c] = pd.to_datetime(orders[c], errors="coerce")

n_orders = len(orders)
n_cust = customers["customer_id"].nunique()

# GMV 用 order_items 金额（price+freight），按订单汇总
item_val = items.groupby("order_id")["price"].sum() + items.groupby("order_id")["freight_value"].sum()
orders_val = orders.merge(item_val.rename("order_value"), on="order_id", how="left")
# 只看已成交(非canceled/unavailable等有效单)？给全量+delivered两种
orders_ok = orders_val[orders_val["order_status"] == "delivered"].copy()
gmv_all = orders_val["order_value"].sum()
gmv_deliv = orders_ok["order_value"].sum()
n_deliv = len(orders_ok)
aov_all = gmv_all / n_orders
aov_deliv = gmv_deliv / n_deliv

# 配送天数 & 超时（仅 delivered 且两者日期都非空）
sub = orders_ok.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date", "order_purchase_timestamp"])
sub = sub.copy()
sub["deliver_days"] = (sub["order_delivered_customer_date"] - sub["order_purchase_timestamp"]).dt.days
sub["overdue"] = (sub["order_delivered_customer_date"] - sub["order_estimated_delivery_date"]).dt.days > 0
avg_deliver = sub["deliver_days"].mean()
overdue_rate = sub["overdue"].mean()
on_time_rate = 1 - overdue_rate

# 取消率
cancel_rate = orders["order_status"].eq("canceled").mean()

# 评分
avg_score = reviews["review_score"].mean()

print("==== 权威 KPI（基于原始 olist 表）====")
print(f"总订单数        : {n_orders}")
print(f"总客户数(去重)  : {n_cust}")
print(f"总GMV(全订单)   : {gmv_all:,.0f}")
print(f"GMV(仅delivered): {gmv_deliv:,.0f}")
print(f"客单价(全)      : {aov_all:.1f}")
print(f"客单价(delivered): {aov_deliv:.1f}")
print(f"delivered订单数 : {n_deliv}")
print(f"平均配送天数    : {avg_deliver:.1f} 天")
print(f"超时率          : {overdue_rate*100:.1f}%")
print(f"准时率          : {on_time_rate*100:.1f}%")
print(f"取消率          : {cancel_rate*100:.2f}%")
print(f"平均评分        : {avg_score:.2f}")
print()
print("==== status 分布 ====")
print(orders["order_status"].value_counts().to_string())

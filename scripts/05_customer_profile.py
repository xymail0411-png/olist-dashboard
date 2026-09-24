# -*- coding: utf-8 -*-
"""
用户画像特征计算骨架 —— 由你运行、修改并补充分析。
目标：对 customer_rfm.csv 的每一层（单次层 4 类 / 复购层 8 类），
计算 品类偏好 / 地区 / 客单支付 / 购买行为 / 履约 / 评价 六类画像指标，并跨层对比。
运行：python 05_customer_profile.py
产出：data/processed/customer_profile.csv（每层一个画像行）；你在此基础上画图、写分析。
"""
import pandas as pd

DATA = r"C:\Users\董心雅\Desktop\数据分析\work\data"
OUT = r"C:\Users\董心雅\Desktop\数据分析\work\data\processed"

# ---------- 1. 读数据 ----------
orders = pd.read_csv(f"{DATA}\\processed\\orders_master.csv")        # 订单主表（地区/价格/时间/履约/评价）
rfm = pd.read_csv(f"{DATA}\\processed\\customer_rfm.csv")            # 分层结果
items = pd.read_csv(f"{DATA}\\olist\\olist_order_items_dataset.csv") # 订单商品明细（order_id, product_id）
prods = pd.read_csv(f"{DATA}\\processed\\products_clean.csv")[["product_id", "product_category_name_english"]]

# ---------- 2. 关联出「客户—品类」 ----------
# 订单明细里一单可能多行（多件商品），先对每个 order_id 展开其品类
cust_item = items.merge(prods, on="product_id", how="left")          # order_id -> 品类
cust_item = cust_item.merge(orders[["order_id", "customer_unique_id"]], on="order_id", how="left")
# 注意：同一订单可能多个品类，先不聚合，保留 客户-品类 行

# ---------- 3. 贴分层 ----------
cust_item = cust_item.merge(rfm[["customer_unique_id", "stage", "segment"]],
                            on="customer_unique_id", how="left")

# ---------- 4. 按层聚合画像指标 ----------
def profile(grp):
    cats = grp["product_category_name_english"].dropna()
    top = cats.value_counts()
    return pd.Series({
        # 品类
        "客户数": grp["customer_unique_id"].nunique(),
        "Top1品类": top.index[0] if len(top) else None,
        "Top1品类占比": round(top.iloc[0] / len(cats), 3) if len(top) else None,
        "平均品类数": round(grp.groupby("customer_unique_id")["product_category_name_english"].nunique().mean(), 2),
        # 地区 / 客单 / 行为 / 履约 / 评价（需回 orders_master 按客户聚合，这里先以订单粒度近似）
    })

# 用「客户」粒度聚合更准：每个客户聚合其全部订单/品类后，再按层平均
cust_agg = orders.groupby("customer_unique_id").agg(
    客单中位=("total_value", "median"), 客单均值=("total_value", "mean"),
    平均件数=("item_count", "mean"), 分期中位=("installments_max", "median"),
    配送均天=("delivery_days", "mean"), 准时率=("is_late", lambda s: 1 - s.mean()),
    评分均值=("review_score", "mean"),
    周末占比=("order_purchase_timestamp", lambda s:
        pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S").dt.dayofweek.ge(5).mean()),
).reset_index()
cust_agg = cust_agg.merge(rfm[["customer_unique_id", "stage", "segment"]], on="customer_unique_id")

# 客户-品类 按客户聚合出 Top1 品类与品类数
cat_agg = cust_item.groupby("customer_unique_id")["product_category_name_english"].agg(
    lambda x: (x.value_counts().index[0], x.value_counts().iloc[0] / len(x))).reset_index()
cat_agg.columns = ["customer_unique_id", "top_cat"]
cat_agg["Top1品类"] = cat_agg["top_cat"].map(lambda t: t[0])
cat_agg["Top1品类占比"] = cat_agg["top_cat"].map(lambda t: round(t[1], 3))
cat_n = cust_item.groupby("customer_unique_id")["product_category_name_english"].nunique().rename("平均品类数")
cust_agg = cust_agg.merge(cat_agg[["customer_unique_id", "Top1品类", "Top1品类占比"]], on="customer_unique_id", how="left")
cust_agg = cust_agg.merge(cat_n, on="customer_unique_id", how="left")

# 地区（客户所在州，客户级属性）
state = orders[["customer_unique_id", "customer_state"]].drop_duplicates()
cust_agg = cust_agg.merge(state, on="customer_unique_id", how="left")

# ---------- 5. 按层汇总画像表 ----------
agg_cols = ["客单中位", "客单均值", "平均件数", "分期中位", "配送均天", "准时率",
            "评分均值", "周末占比", "Top1品类占比", "平均品类数"]
profile_df = cust_agg.groupby(["stage", "segment"]).agg(
    客户数=("customer_unique_id", "count"),
    **{c: (c, "mean") for c in agg_cols},
    覆盖州数=("customer_state", "nunique"),
).round(3).reset_index()
# Top1 品类 / Top1 州 用该层最常见值
top1 = cust_agg.groupby(["stage", "segment"])["Top1品类"].agg(lambda s: s.mode().iloc[0])
top_state = cust_agg.groupby(["stage", "segment"])["customer_state"].agg(lambda s: s.mode().iloc[0])
profile_df = profile_df.merge(top1.rename("Top1品类(最常见)"), on=["stage", "segment"], how="left")
profile_df = profile_df.merge(top_state.rename("Top1州(最常见)"), on=["stage", "segment"], how="left")

profile_df.to_csv(f"{OUT}\\customer_profile.csv", index=False, encoding="utf-8-sig")
print("画像汇总已写出:", f"{OUT}\\customer_profile.csv")
print(profile_df.to_string(index=False))

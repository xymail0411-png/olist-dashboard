# -*- coding: utf-8 -*-
"""
02 数据治理：把 MySQL 原始表清洗为"分析用干净数据"

产出 data/processed/ 下：
  - orders_master.csv   订单主宽表（每个订单一行：客户/金额/时效/评分/支付）
  - products_clean.csv  商品表（类目缺失 -> unknown）
  - geolocation_br.csv  过滤巴西范围后的地理表

治理口径：
  1. 有效订单 = order_status == 'delivered'
  2. 分析时间范围 = 2017-01-01 起（剔除 2016 平台冷启动测试期样本，与 EDA 口径一致）
  3. 用户口径 = customer_unique_id（一人多单去重）
  4. 类目缺失 -> 'unknown'
  5. 地理范围 = 巴西 (lat -35~6, lng -75~-30)
  6. GMV = order_items.price + freight_value 求和（不用 payments，避免一单多支付重复计）

用法：
    $env:MYSQL_PASSWORD='你的密码'
    python scripts/02_clean.py
"""
import os
from pathlib import Path

import pandas as pd
import pymysql

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DB_CONFIG = dict(
    host="localhost",
    user="root",
    password=os.environ.get("MYSQL_PASSWORD", ""),
    database="olist",
    charset="utf8mb4",
)

# 巴西大致经纬度范围（用于过滤 geolocation 离群坐标）
BR_LAT = (-35.0, 6.0)
BR_LNG = (-75.0, -30.0)


def read_table(conn, name):
    return pd.read_sql(f"SELECT * FROM `{name}`", conn)


def main():
    report = []  # 治理记录（每步：规则 -> 影响）

    conn = pymysql.connect(**DB_CONFIG)
    orders = read_table(conn, "orders")
    items = read_table(conn, "order_items")
    payments = read_table(conn, "payments")
    reviews = read_table(conn, "reviews")
    customers = read_table(conn, "customers")
    products = read_table(conn, "products")
    geo = read_table(conn, "geolocation")
    translation = read_table(conn, "product_category_name_translation")
    conn.close()

    # ---------- 1. 有效订单口径：只看 delivered ----------
    n_all = len(orders)
    valid = orders[orders["order_status"] == "delivered"].copy()
    report.append(f"有效订单：delivered {len(valid):,} / 全部 {n_all:,}（排除非 delivered {n_all - len(valid):,} 单）")

    # ---------- 1.5 分析时间范围：剔除 2016 平台冷启动测试期（EDA 口径 2017-01 起）----------
    n_before_t = len(valid)
    valid = valid[pd.to_datetime(valid["order_purchase_timestamp"]) >= "2017-01-01"].copy()
    report.append(f"时间范围：剔除 2016 冷启动 {n_before_t - len(valid):,} 单，保留 {len(valid):,} 单（2017-01 起，与 EDA 一致）")

    # ---------- 2. 订单金额：明细求和（GMV 口径）----------
    item_agg = (
        items.groupby("order_id")
        .agg(
            item_count=("order_item_id", "count"),
            sum_price=("price", "sum"),
            sum_freight=("freight_value", "sum"),
        )
        .reset_index()
    )
    item_agg["total_value"] = item_agg["sum_price"] + item_agg["sum_freight"]
    report.append(f"订单金额：明细求和得到 {len(item_agg):,} 单的 GMV（price + freight）")

    # ---------- 3. 支付聚合（用于核对，不参与 GMV）----------
    pay_agg = (
        payments.groupby("order_id")
        .agg(
            payment_type=("payment_type", lambda s: "/".join(sorted(set(s)))),
            payment_total=("payment_value", "sum"),
            installments_max=("payment_installments", "max"),
        )
        .reset_index()
    )

    # ---------- 4. 评论聚合：评分 + 是否有评语 ----------
    rev_agg = (
        reviews.groupby("order_id")
        .agg(
            review_score=("review_score", "mean"),
            has_comment=("review_comment_message", lambda s: int(s.notna().any())),
        )
        .reset_index()
    )
    report.append(f"评论聚合：{len(rev_agg):,} 单有评分记录")

    # ---------- 5. 客户信息（唯一用户口径）----------
    cust = customers[["customer_id", "customer_unique_id", "customer_city", "customer_state"]]

    # ---------- 6. 履约时效：实际 vs 预计 ----------
    valid["delivery_days"] = (
        pd.to_datetime(valid["order_delivered_customer_date"]) - pd.to_datetime(valid["order_purchase_timestamp"])
    ).dt.days
    valid["estimated_days"] = (
        pd.to_datetime(valid["order_estimated_delivery_date"]) - pd.to_datetime(valid["order_purchase_timestamp"])
    ).dt.days
    valid["is_late"] = valid["delivery_days"] > valid["estimated_days"]
    report.append(f"履约时效：超时订单 {int(valid['is_late'].sum()):,} / {len(valid):,}（实际送达晚于预计）")

    # ---------- 7. 组装订单主宽表 ----------
    master = (
        valid.merge(cust, on="customer_id", how="left")
        .merge(item_agg, on="order_id", how="left")
        .merge(pay_agg, on="order_id", how="left")
        .merge(rev_agg, on="order_id", how="left")
    )
    # 没有明细/支付/评论的订单置 0 / 空
    master["total_value"] = master["total_value"].fillna(0)
    master["item_count"] = master["item_count"].fillna(0).astype(int)
    master["review_score"] = master["review_score"].fillna(0).astype(int)
    master["has_comment"] = master["has_comment"].fillna(0).astype(int)

    keep_cols = [
        "order_id", "customer_unique_id", "customer_city", "customer_state",
        "order_purchase_timestamp", "order_delivered_customer_date",
        "order_estimated_delivery_date", "delivery_days", "estimated_days", "is_late",
        "item_count", "total_value", "payment_type", "payment_total", "installments_max",
        "review_score", "has_comment",
    ]
    master = master[keep_cols]
    master.to_csv(PROCESSED_DIR / "orders_master.csv", index=False, encoding="utf-8-sig")
    report.append(f"订单主宽表：{len(master):,} 行 x {master.shape[1]} 列 -> data/processed/orders_master.csv")

    # ---------- 8. 商品类目缺失 -> unknown ----------
    products_clean = products.copy()
    n_missing_cat = int(products_clean["product_category_name"].isna().sum())
    products_clean["product_category_name"] = products_clean["product_category_name"].fillna("unknown")
    products_clean = products.merge(translation, on="product_category_name", how="left")
    products_clean["product_category_name_english"] = products_clean["product_category_name_english"].fillna("unknown")
    products_clean.to_csv(PROCESSED_DIR / "products_clean.csv", index=False, encoding="utf-8-sig")
    report.append(f"商品类目：{n_missing_cat:,} 个缺失类目 -> 'unknown'，共 {len(products_clean):,} 个商品 -> products_clean.csv")

    # ---------- 9. 地理过滤（巴西范围）----------
    n_geo_before = len(geo)
    geo_br = geo[
        geo["geolocation_lat"].between(*BR_LAT) & geo["geolocation_lng"].between(*BR_LNG)
    ].copy()
    geo_br.to_csv(PROCESSED_DIR / "geolocation_br.csv", index=False, encoding="utf-8-sig")
    report.append(
        f"地理过滤：保留巴西范围 {len(geo_br):,} / {n_geo_before:,}（排除 {n_geo_before - len(geo_br):,} 个离群坐标）-> geolocation_br.csv"
    )

    # ---------- 输出治理报告 ----------
    print("=" * 70)
    print("数据治理报告（每条规则的执行结果）")
    print("=" * 70)
    for i, line in enumerate(report, 1):
        print(f"{i}. {line}")
    print("=" * 70)
    print(f"产出目录：{PROCESSED_DIR}")


if __name__ == "__main__":
    main()

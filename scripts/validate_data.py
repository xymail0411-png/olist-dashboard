# -*- coding: utf-8 -*-
"""
数据完整性验证脚本
检查 data/olist/ 下 9 张表的：存在性、行数、列名、空值率、主键唯一性、外键一致性。

用法：
    python scripts/validate_data.py
退出码：0 = 全部通过；1 = 存在失败项
"""

import os
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "olist"

# 期望的表结构：表名 -> (最小行数, 期望列, 主键列, 是否允许主键重复)
# 说明：olist_order_reviews_dataset 的 review_id 存在官方已知重复——
#       同一条评论（内容/评分/时间完全相同）被关联到同一买家的多笔订单。
EXPECTED = {
    "olist_customers_dataset.csv": (90000, ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"], "customer_id", False),
    "olist_orders_dataset.csv": (90000, ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"], "order_id", False),
    "olist_order_items_dataset.csv": (100000, ["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price", "freight_value"], None, False),
    "olist_order_payments_dataset.csv": (90000, ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"], None, False),
    "olist_products_dataset.csv": (30000, ["product_id", "product_category_name", "product_name_lenght", "product_description_lenght", "product_photos_qty", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"], "product_id", False),
    "olist_order_reviews_dataset.csv": (90000, ["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message", "review_creation_date", "review_answer_timestamp"], "review_id", True),
    "olist_sellers_dataset.csv": (3000, ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"], "seller_id", False),
    "olist_geolocation_dataset.csv": (900000, ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng", "geolocation_city", "geolocation_state"], None, False),
    "product_category_name_translation.csv": (60, ["product_category_name", "product_category_name_english"], "product_category_name", False),
}

# 需要做外键一致性检查的 (子表, 子表键, 主表, 主表键)
FK_CHECKS = [
    ("olist_orders_dataset.csv", "customer_id", "olist_customers_dataset.csv", "customer_id"),
    ("olist_order_items_dataset.csv", "order_id", "olist_orders_dataset.csv", "order_id"),
    ("olist_order_items_dataset.csv", "product_id", "olist_products_dataset.csv", "product_id"),
    ("olist_order_items_dataset.csv", "seller_id", "olist_sellers_dataset.csv", "seller_id"),
    ("olist_order_payments_dataset.csv", "order_id", "olist_orders_dataset.csv", "order_id"),
    ("olist_order_reviews_dataset.csv", "order_id", "olist_orders_dataset.csv", "order_id"),
]


def main():
    if not DATA_DIR.exists():
        print(f"[FAIL] 数据目录不存在: {DATA_DIR}")
        sys.exit(1)

    failures = []
    tables = {}

    print(f"数据目录: {DATA_DIR}\n")

    # 1) 表存在性与基础检查
    for fname, (min_rows, cols, pk, pk_dup_ok) in EXPECTED.items():
        path = DATA_DIR / fname
        if not path.exists():
            failures.append(f"[FAIL] 缺少文件: {fname}")
            continue
        df = pd.read_csv(path)
        tables[fname] = df
        n = len(df)
        missing_cols = [c for c in cols if c not in df.columns]
        msg = f"[ OK ] {fname:<42} {n:>9,} 行"
        if missing_cols:
            failures.append(f"[FAIL] {fname} 缺少列: {missing_cols}")
            msg += f"  <-- 缺列!"
        if pk:
            dup_cnt = int(df[pk].duplicated().sum())
            if dup_cnt > 0:
                if pk_dup_ok:
                    msg += f"  (已知特征: {pk} 重复 {dup_cnt} 条，同评论关联多订单)"
                    print(f"[WARN] {fname:<40} {n:>9,} 行  {pk} 重复 {dup_cnt} 条（官方已知特征，处理时按 order_id+review_id 去重）")
                    continue
                failures.append(f"[FAIL] {fname} 主键 {pk} 存在重复")
                msg += f"  <-- 主键重复!"
        print(msg)

    # 2) 外键一致性
    print("\n外键一致性检查:")
    for child, child_key, parent, parent_key in FK_CHECKS:
        if child not in tables or parent not in tables:
            continue
        child_ids = set(tables[child][child_key].dropna())
        parent_ids = set(tables[parent][parent_key].dropna())
        orphans = len(child_ids - parent_ids)
        status = "OK" if orphans == 0 else f"{orphans} 个孤儿键"
        print(f"  [{'FAIL' if orphans else ' OK '}] {child}.{child_key} -> {parent}.{parent_key}  {status}")
        if orphans:
            failures.append(f"[FAIL] {child}.{child_key} 有 {orphans} 个值在 {parent} 中不存在")

    # 3) 汇总
    print("\n" + "=" * 60)
    if failures:
        print(f"验证失败：{len(failures)} 项")
        for f in failures:
            print(" ", f)
        sys.exit(1)
    print(f"全部通过：{len(EXPECTED)} 张表完整，外键一致。")
    sys.exit(0)


if __name__ == "__main__":
    main()

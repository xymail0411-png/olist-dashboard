# -*- coding: utf-8 -*-
"""
03 EDA 探索性数据分析
数据源：data/processed/ 下的治理后数据
产出：docs/figures/*.png 图表 + 终端打印关键数字

6 个探索方向：
  1. 经营总览   - 订单量、GMV、客单价、复购率
  2. 时间趋势   - 月度订单/GMV（找促销峰值）
  3. 品类结构   - Top 品类（订单数、GMV）
  4. 地理分布   - 州级订单分布
  5. 履约时效   - 交付天数分布、超时率
  6. 评论评分   - 评分分布、评分与金额/超时的关系

用法：
    C:\\...\\python.exe scripts/03_eda.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接存文件
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Windows 中文字体（微软雅黑）；若乱码可改为 SimHei
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

master = pd.read_csv(PROCESSED / "orders_master.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"])
products = pd.read_csv(PROCESSED / "products_clean.csv")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   [图] docs/figures/{name}")


# ---------- 1. 经营总览 ----------
print("=" * 66)
print("1. 经营总览")
print("=" * 66)
total_orders = len(master)
total_gmv = master["total_value"].sum()
aov = total_gmv / total_orders
n_customers = master["customer_unique_id"].nunique()
repeat_ratio = (master.groupby("customer_unique_id")["order_id"].nunique() > 1).mean()
print(f"   有效订单数 : {total_orders:,}")
print(f"   总 GMV     : R$ {total_gmv:,.2f}（约 ${total_gmv / 4.5:,.0f} 美元）")
print(f"   客单价 AOV : R$ {aov:,.2f}")
print(f"   唯一客户   : {n_customers:,}")
print(f"   复购客户占比: {repeat_ratio:.1%}")

# 复购占比环形图（空心环，带人数与百分比标注）
import numpy as np
cust_orders = master.groupby("customer_unique_id")["order_id"].nunique()
one_time = int((cust_orders == 1).sum())
repeat_c = len(cust_orders) - one_time
repeat_orders = int(cust_orders[cust_orders > 1].sum())
print(f"   购买1次客户 : {one_time:,} 人（{one_time / len(cust_orders):.1%}）")
print(f"   复购客户    : {repeat_c:,} 人，贡献 {repeat_orders:,} 单（占订单 {repeat_orders / len(master):.1%}）")
fig, ax = plt.subplots(figsize=(7, 5.2))
sizes = [one_time, repeat_c]
colors = ["#C9D0D9", "#B84C3A"]  # 中性浅灰(单次) + 暖赤强调(复购，低占比关键点)
wedges, _ = ax.pie(
    sizes, colors=colors, startangle=90, counterclock=False,
    wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
)
labels = [
    f"仅购买 1 次\n{one_time:,} 人（{one_time / len(cust_orders):.1%}）",
    f"复购客户\n{repeat_c:,} 人（{repeat_c / len(cust_orders):.1%}）",
]
text_colors = ["#4b5563", "#B84C3A"]
for w, txt, tc in zip(wedges, labels, text_colors):
    ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
    x, y = 1.32 * np.cos(ang), 1.32 * np.sin(ang)
    ax.text(x, y, txt, ha="center", va="center", fontsize=10.5, color=tc)
ax.text(0, 0, "复购率\n3.0%", ha="center", va="center", fontsize=15, fontweight="bold", color="#B84C3A")
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-1.7, 1.7)
ax.set_title("客户复购占比（按客户数）")
save(fig, "eda_purchase_frequency.png")

# ---------- 2. 时间趋势 ----------
print("\n" + "=" * 66)
print("2. 时间趋势（月度订单量）")
print("=" * 66)
monthly = master.set_index("order_purchase_timestamp").resample("M").agg(
    orders=("order_id", "count"), gmv=("total_value", "sum")
)
top3 = monthly.nlargest(3, "orders")
print(monthly.to_string())
print("   订单量 Top3 月份：")
for dt, row in top3.iterrows():
    print(f"     {dt.strftime('%Y-%m')}: {int(row['orders']):,} 单（GMV R$ {row['gmv']:,.0f}）")
# 2016 年为平台启动期（全年仅 329 单，2016-11 无数据），趋势图从稳定运营期 2017-01 起
monthly_stable = monthly.loc["2017-01":]
fig, ax = plt.subplots(figsize=(12, 4.5))
bars = ax.bar(monthly_stable.index.strftime("%Y-%m"), monthly_stable["orders"], color="#4f81bd")
ax.bar_label(bars, fontsize=6.5, padding=1)
ax.set_title("月度订单量（2017-01 ~ 2018-08，稳定运营期）")
ax.set_ylabel("订单数"); ax.tick_params(axis="x", rotation=60, labelsize=8)
save(fig, "eda_monthly_orders.png")

# ---------- 3. 品类结构 ----------
print("\n" + "=" * 66)
print("3. 品类结构（Top 10 品类订单量）")
print("=" * 66)
items = pd.read_csv(PROCESSED / ".." / "olist" / "olist_order_items_dataset.csv")
items_cat = items.merge(products[["product_id", "product_category_name_english"]], on="product_id", how="left")
cat_orders = items_cat["product_category_name_english"].fillna("unknown").value_counts().head(10)
print(cat_orders.to_string())
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(cat_orders.index[::-1], cat_orders.values[::-1], color="#8fbc8f")
ax.bar_label(bars, fontsize=9, padding=2)
ax.set_title("Top 10 品类（按明细行数）")
ax.set_xlabel("明细行数")
save(fig, "eda_top_categories.png")

# ---------- 4. 地理分布 ----------
print("\n" + "=" * 66)
print("4. 地理分布（Top 10 州，按订单量）")
print("=" * 66)
state_orders = master["customer_state"].value_counts().head(10)
print(state_orders.to_string())
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(state_orders.index, state_orders.values, color="#c0504d")
ax.bar_label(bars, fontsize=9, padding=2)
ax.set_title("Top 10 客户所在州（订单量）")
ax.set_xlabel("州代码"); ax.set_ylabel("订单数")
save(fig, "eda_state_orders.png")

# ---------- 5. 履约时效 ----------
print("\n" + "=" * 66)
print("5. 履约时效")
print("=" * 66)
late_rate = master["is_late"].mean()
print(f"   平均交付天数 : {master['delivery_days'].mean():.1f} 天")
print(f"   交付中位数   : {master['delivery_days'].median():.0f} 天")
print(f"   超时率       : {late_rate:.1%}")
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(master["delivery_days"].clip(0, 60), bins=40, color="#7799cc")
med = master["delivery_days"].median()
ax.axvline(med, color="red", ls="--", label=f"中位数 {med:.0f} 天")
ax.text(med + 0.5, ax.get_ylim()[1] * 0.95, f" 中位数 {med:.0f} 天", color="red", fontsize=10, va="top")
ax.set_title("交付天数分布")
ax.set_xlabel("交付天数"); ax.legend()
save(fig, "eda_delivery_days.png")

# ---------- 6. 评论评分 ----------
print("\n" + "=" * 66)
print("6. 评论评分")
print("=" * 66)
scored = master[master["review_score"] > 0]
print(scored["review_score"].value_counts().sort_index().to_string())
avg_by_late = scored.groupby("is_late")["review_score"].mean()
print(f"   准时订单平均评分 : {avg_by_late.get(False, float('nan')):.2f}")
print(f"   超时订单平均评分 : {avg_by_late.get(True, float('nan')):.2f}")
fig, ax = plt.subplots(figsize=(7, 4))
vc = scored["review_score"].value_counts().sort_index()
bars = ax.bar(vc.index.astype(str), vc.values, color="#e8a33d")
ax.bar_label(bars, fontsize=9, padding=2)
ax.set_title("评分分布（1-5 星）")
ax.set_xlabel("评分"); ax.set_ylabel("订单数")
save(fig, "eda_review_score.png")

print("\nEDA 完成，图表在 docs/figures/，关键数字见上（用于填写 EDA 报告）。")

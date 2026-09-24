# -*- coding: utf-8 -*-
"""
补跑 H2（大促订单质量）、H3（品类量价矩阵），并计算仪表盘全部 JSON 数据。
产出: data/processed/dashboard_data.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"C:\Users\董心雅\Desktop\数据分析\work")
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "olist"

PY = r"C:\Users\董心雅\AppData\Local\Programs\Python\Python39\python.exe"

# ---------- 加载数据 ----------
master = pd.read_csv(PROCESSED / "orders_master.csv",
                     parse_dates=["order_purchase_timestamp"])
items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
prods = pd.read_csv(PROCESSED / "products_clean.csv")[["product_id", "product_category_name_english"]]
rfm = pd.read_csv(PROCESSED / "customer_rfm.csv")

master = master.sort_values("order_purchase_timestamp").reset_index(drop=True)
print(f"orders_master: {len(master):,} rows, cols: {list(master.columns)}")

# 关联品类
items = items.merge(prods, on="product_id", how="left")
items["cat"] = items["product_category_name_english"].fillna("unknown")

D = {}  # 所有输出数据

# ============================================================
# 0. 顶层 KPI
# ============================================================
total_gmv = master["total_value"].sum()
total_orders = len(master)
total_customers = master["customer_unique_id"].nunique()
aov = total_gmv / total_orders
repeat_rate = (rfm["stage"] == "复购客户").mean() * 100
avg_delivery = master["delivery_days"].mean()
late_rate = master["is_late"].mean() * 100
avg_score = master[master["review_score"] > 0]["review_score"].mean()

D["kpi"] = {
    "gmv": round(total_gmv),
    "orders": total_orders,
    "customers": total_customers,
    "aov": round(aov, 1),
    "repeat_rate": round(repeat_rate, 1),
    "avg_delivery": round(avg_delivery, 1),
    "late_rate": round(late_rate, 1),
    "avg_score": round(avg_score, 2),
    "date_range": f"{master['order_purchase_timestamp'].min():%Y-%m} ~ {master['order_purchase_timestamp'].max():%Y-%m}",
}
print("KPI done")

# ============================================================
# 1. 月度趋势
# ============================================================
master["ym"] = master["order_purchase_timestamp"].dt.to_period("M").astype(str)
monthly = master.groupby("ym").agg(
    gmv=("total_value", "sum"),
    orders=("order_id", "count"),
    customers=("customer_unique_id", "nunique"),
    late_rate=("is_late", "mean"),
    avg_score=("review_score", lambda x: x[x > 0].mean()),
).reset_index()
D["monthly"] = {
    "months": monthly["ym"].tolist(),
    "gmv": [round(x) for x in monthly["gmv"]],
    "orders": monthly["orders"].tolist(),
}
print("Monthly trend done")

# ============================================================
# 2. 品类 Top10 GMV
# ============================================================
cat_order = items.merge(master[["order_id", "total_value"]], on="order_id", how="inner")
cat_gmv = cat_order.groupby("cat")["total_value"].sum().sort_values(ascending=False).head(10)
D["categories"] = {
    "names": [c.replace("_", " ") for c in cat_gmv.index],
    "gmv": [round(v) for v in cat_gmv.values],
}
print("Categories done")

# ============================================================
# 3. 地理分布（巴西各州订单量）
# ============================================================
state_orders = master.groupby("customer_state")["order_id"].count().sort_values(ascending=False)
D["states"] = {
    "names": state_orders.index.tolist(),
    "values": state_orders.values.tolist(),
}
print("States done")

# ============================================================
# 4. 支付方式
# ============================================================
pay = pd.read_csv(RAW / "olist_order_payments_dataset.csv")
pay_sum = pay.groupby("payment_type")["payment_value"].sum().sort_values(ascending=False)
D["payment"] = {
    "names": list(pay_sum.index),
    "values": [round(v) for v in pay_sum.values],
}
print("Payment done")

# ============================================================
# 5. RFM 单次层 / 复购层数据
# ============================================================
# 单次层 4 类
single_seg = rfm[rfm["stage"] == "单次购买"]
repeat_seg = rfm[rfm["stage"] == "复购客户"]

# 单次层
single_summary = single_seg.groupby("segment").agg(
    customers=("customer_unique_id", "count"),
    gmv=("monetary", "sum"),
).reset_index()
single_summary["cust_pct"] = single_summary["customers"] / len(single_seg) * 100
single_summary["gmv_pct"] = single_summary["gmv"] / single_summary["gmv"].sum() * 100

# 排序固定
single_order = ["高潜新客", "高价值唤醒", "待培育新客", "流失观望"]
single_summary["sort"] = single_summary["segment"].map({v: i for i, v in enumerate(single_order)})
single_summary = single_summary.sort_values("sort")

D["rfm_single"] = {
    "segments": single_summary["segment"].tolist(),
    "cust_pct": [round(x, 1) for x in single_summary["cust_pct"]],
    "gmv_pct": [round(x, 1) for x in single_summary["gmv_pct"]],
    "customers": single_summary["customers"].tolist(),
    "gmv": [round(x) for x in single_summary["gmv"]],
}

# 复购层 8 类
repeat_summary = repeat_seg.groupby("segment").agg(
    customers=("customer_unique_id", "count"),
    gmv=("monetary", "sum"),
).reset_index()
repeat_summary["cust_pct"] = repeat_summary["customers"] / len(repeat_seg) * 100
repeat_summary["gmv_pct"] = repeat_summary["gmv"] / repeat_summary["gmv"].sum() * 100

repeat_order = ["重要价值客户", "重要发展客户", "重要保持客户", "重要挽留客户",
                "一般价值客户", "一般发展客户", "一般保持客户", "一般挽留客户"]
repeat_summary["sort"] = repeat_summary["segment"].map({v: i for i, v in enumerate(repeat_order)})
repeat_summary = repeat_summary.sort_values("sort")

D["rfm_repeat"] = {
    "segments": repeat_summary["segment"].tolist(),
    "cust_pct": [round(x, 1) for x in repeat_summary["cust_pct"]],
    "gmv_pct": [round(x, 1) for x in repeat_summary["gmv_pct"]],
    "customers": repeat_summary["customers"].tolist(),
}
print("RFM done")

# ============================================================
# 6. 画像对比（单次 vs 复购）
# ============================================================
# 每客户品类数
ci = items.merge(master[["order_id", "customer_unique_id"]], on="order_id", how="inner")
cust_ncat = ci[["customer_unique_id", "cat"]].drop_duplicates().groupby("customer_unique_id")["cat"].nunique()

# 客户级指标
cust = master.groupby("customer_unique_id").agg(
    orders=("order_id", "count"),
    total_value=("total_value", "sum"),
    delivery=("delivery_days", "mean"),
    late_rate=("is_late", "mean"),
).reset_index()
cust["n_cat"] = cust["customer_unique_id"].map(cust_ncat).fillna(1).astype(int)

# 评分
scored = master[master["review_score"] > 0].groupby("customer_unique_id")["review_score"].mean()
cust["score"] = cust["customer_unique_id"].map(scored)
cust = cust.merge(rfm[["customer_unique_id", "stage"]], on="customer_unique_id", how="left")

s = cust[cust["stage"] == "单次购买"]
r = cust[cust["stage"] == "复购客户"]

D["profile"] = {
    "metrics": [
        {"name": "平均品类数", "single": round(s["n_cat"].mean(), 2), "repeat": round(r["n_cat"].mean(), 2)},
        {"name": "客单价(R$)", "single": round(s["total_value"].mean(), 1), "repeat": round(r["total_value"].mean(), 1)},
        {"name": "平均评分", "single": round(s["score"].mean(), 2), "repeat": round(r["score"].mean(), 2)},
        {"name": "超时率(%)", "single": round(s["late_rate"].mean() * 100, 1), "repeat": round(r["late_rate"].mean() * 100, 1)},
        {"name": "配送天数", "single": round(s["delivery"].mean(), 1), "repeat": round(r["delivery"].mean(), 1)},
    ]
}
print("Profile done")

# ============================================================
# 7. H6: 体验→满意→复购 链
# ============================================================
# 准时 vs 超时评分
on_time_score = master[~master["is_late"]]["review_score"]
on_time_score = on_time_score[on_time_score > 0]
late_score = master[master["is_late"]]["review_score"]
late_score = late_score[late_score > 0]

# 评分档 vs 复购率
master["score_band"] = pd.cut(master["review_score"], [0, 2, 4, 5.1], labels=["低分(1-2)", "中分(3-4)", "高分(5)"])
score_repeat = master.drop_duplicates("customer_unique_id").merge(
    rfm[["customer_unique_id", "stage"]], on="customer_unique_id", how="left")
score_repeat["is_repeat"] = (score_repeat["stage"] == "复购客户").astype(int)
band_repeat = score_repeat.groupby("score_band", observed=True)["is_repeat"].mean() * 100

D["h6"] = {
    "on_time_score": round(on_time_score.mean(), 2),
    "late_score": round(late_score.mean(), 2),
    "on_time_n": len(on_time_score),
    "late_n": len(late_score),
    "bands": [str(b) for b in band_repeat.index],
    "band_repeat_rate": [round(x, 2) for x in band_repeat.values],
}
print("H6 done")

# ============================================================
# 8. 增长分解（新客 vs 复购 GMV）
# ============================================================
growth = pd.read_csv(PROCESSED / "growth_decomposition.csv")
D["growth"] = {
    "months": growth["ym"].tolist(),
    "new_gmv": [round(x) for x in growth["新客首单"]],
    "repeat_gmv": [round(x) for x in growth["老客复购"]],
    "new_pct": growth["新客首单占比%"].tolist(),
}
print("Growth done")

# ============================================================
# 9. H2: 大促订单质量（黑五 = 2017-11）
# ============================================================
master["ym_dt"] = master["order_purchase_timestamp"]
bf = master[master["ym"] == "2017-11"]  # Black Friday month
normal = master[master["ym"] != "2017-11"]

D["h2"] = {
    "bf_orders": len(bf),
    "normal_orders": len(normal),
    "bf_aov": round(bf["total_value"].mean(), 1),
    "normal_aov": round(normal["total_value"].mean(), 1),
    "bf_late_rate": round(bf["is_late"].mean() * 100, 1),
    "normal_late_rate": round(normal["is_late"].mean() * 100, 1),
    "bf_avg_score": round(bf[bf["review_score"] > 0]["review_score"].mean(), 2),
    "normal_avg_score": round(normal[normal["review_score"] > 0]["review_score"].mean(), 2),
    "bf_new_cust": bf["customer_unique_id"].nunique(),
}

# t 检验 AOV
u_aov, p_aov = stats.mannwhitneyu(bf["total_value"], normal["total_value"], alternative="two-sided")
D["h2"]["aov_p"] = round(p_aov, 4)
print(f"H2 done: BF AOV {D['h2']['bf_aov']} vs normal {D['h2']['normal_aov']}")

# ============================================================
# 10. H3: 品类量价矩阵
# ============================================================
cat_stats = items.merge(master[["order_id", "total_value"]], on="order_id", how="inner")
cat_stats = cat_stats.groupby("cat").agg(
    orders=("order_id", "count"),
    gmv=("total_value", "sum"),
).reset_index()
cat_stats["aov"] = cat_stats["gmv"] / cat_stats["orders"]
cat_stats = cat_stats[cat_stats["orders"] >= 200].sort_values("gmv", ascending=False)
cat_stats = cat_stats.head(15)

D["h3"] = {
    "cats": [c.replace("_", " ") for c in cat_stats["cat"]],
    "orders": cat_stats["orders"].tolist(),
    "aov": [round(x, 1) for x in cat_stats["aov"]],
    "gmv": [round(x) for x in cat_stats["gmv"]],
}
print("H3 done")

# ============================================================
# 11. 复购预测模型
# ============================================================
D["model"] = {
    "auc": {
        "models": ["逻辑回归", "LightGBM"],
        "full_period": [0.55, 0.60],
        "window30": [0.54, 0.62],
    },
    "topk": {
        "k": [1, 2, 3, 5, 10],
        "gbm_coverage": [2.5, 4.3, 6.2, 9.6, 17.0],
        "lr_coverage": [0.8, 2.2, 3.2, 5.5, 10.5],
    },
    "importance": [
        {"name": "首单金额", "value": 722},
        {"name": "配送天数", "value": 642},
        {"name": "品类", "value": 595},
        {"name": "首购月份", "value": 356},
        {"name": "客户所在州", "value": 289},
        {"name": "首单评分", "value": 172},
        {"name": "是否超时", "value": 24},
    ],
}
print("Model done")

# ============================================================
# 12. 决策建议
# ============================================================
D["decision"] = {
    "budget": [
        {"name": "拉新预算", "value": 80},
        {"name": "留旧预算", "value": 20},
    ],
    "segments": [
        {"name": "高潜新客", "customers": 23028, "action": "首购30天内复购券+品类推荐", "budget_share": 40, "priority": "P0"},
        {"name": "高价值唤醒", "customers": 22122, "action": "定向邮件召回+bed_bath_table补货", "budget_share": 30, "priority": "P0"},
        {"name": "复购层·重要", "customers": 1418, "action": "VIP会员权益+专属客服", "budget_share": 20, "priority": "P1"},
        {"name": "待培育/流失", "customers": 45165, "action": "轻触达+凑单提客单，控制打扰", "budget_share": 10, "priority": "P2"},
    ],
    "roi_note": "留旧池按30天窗打分，Top10%客户可抓21%复购（盲投10%），投产比目标>1",
}
print("Decision done")

# ---------- 保存 ----------
out = PROCESSED / "dashboard_data.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(D, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out} ({out.stat().st_size:,} bytes)")

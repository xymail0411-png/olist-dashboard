# -*- coding: utf-8 -*-
"""
06 H1 显著性检验：复购客户 vs 单次客户 的画像差异
数据源：
  - data/processed/orders_master.csv        订单主宽表（金额/履约/评分/地区）
  - data/processed/customer_rfm.csv         两阶段分层结果（stage: 单次购买/复购客户）
  - data/processed/products_clean.csv       商品类目（英文）
  - data/olist/olist_order_items_dataset.csv 订单明细（order_id -> product_id）
产出：
  - data/processed/h1_test_results.csv      连续指标检验结果
  - data/processed/h1_categorical_results.csv 品类/州卡方检验结果
  - docs/figures/h1_metrics_compare.png     关键指标 单次 vs 复购 对比
  - docs/figures/h1_category_divergence.png 品类分布差异

H1 表述：复购客户与单次客户在客单价、品类、地区分布上存在显著差异。

检验设计（重要）：
  大样本下（单次 9 万 vs 复购 0.28 万）p 值几乎必然显著，故结论以效应量为准——
    连续指标 -> Mann-Whitney U + 秩双列相关 r（|r|: <.1 无 / .1~.3 小 / .3~.5 中 / >.5 大）
    类别指标 -> 卡方检验 + Cramér's V（<.1 小 / .1~.3 中 / >.3 大）

指标与主线的映射（结构 vs 体验）：
  结构性（改不了）：客单价、首单金额、品类、地区 —— 决定"复购者是谁"
  体验性（改得了）：评分、超时率 —— 此处仅作描述，因果检验归 H6（本层对比存在混杂：
    复购者有 2+ 单，其评分/超时是多次的均值，不能直接归因"体验好才复购"）。

用法：
    C:\\...\\python.exe scripts/06_h1_test.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "olist"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

C1, C2 = "#C9D0D9", "#52647D"   # 单次(浅) / 复购(深)


def mw_test(a, b):
    """Mann-Whitney U 双侧检验，返回 (U, p, 秩双列相关 r)。r>0 表示复购组更高。"""
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    r = 1 - 2 * u / (n1 * n2)
    return u, p, r


def eff_label_r(r):
    ar = abs(r)
    return "大" if ar > .5 else "中" if ar > .3 else "小" if ar > .1 else "无"


def eff_label_v(v):
    return "大" if v > .3 else "中" if v > .1 else "小"


def cramers_v(chi2, n, r, k):
    return float(np.sqrt(chi2 / (n * min(r - 1, k - 1))))


def chi2_test(s_single, s_repeat, top_n):
    """两组类别分布卡方检验；取并集 top_n 类，其余归'其他'。返回 (chi2, p, dof, V, table)。"""
    comb = pd.concat([s_single, s_repeat])
    top = comb.value_counts().head(top_n).index.tolist()

    def map_cat(x):
        return x if x in top else "其他"

    cats = top + ["其他"]
    table = pd.DataFrame({
        "单次": s_single.map(map_cat).value_counts().reindex(cats, fill_value=0),
        "复购": s_repeat.map(map_cat).value_counts().reindex(cats, fill_value=0),
    })
    chi2, p, dof, _ = stats.chi2_contingency(table.values)
    v = cramers_v(chi2, table.values.sum(), table.shape[0], table.shape[1])
    return chi2, p, dof, v, table


def main():
    master = pd.read_csv(PROCESSED / "orders_master.csv",
                         parse_dates=["order_purchase_timestamp"]).sort_values("order_purchase_timestamp")
    rfm = pd.read_csv(PROCESSED / "customer_rfm.csv")

    # ---------- 1. 客户级指标 ----------
    scored = master[master["review_score"] > 0]
    score_mean = scored.groupby("customer_unique_id")["review_score"].mean().rename("评分均值")

    cust = (
        master.groupby("customer_unique_id")
        .agg(订单数=("order_id", "count"),
             总金额=("total_value", "sum"),
             首单金额=("total_value", "first"),          # 已按时间排序，first = 首单
             超时率=("is_late", "mean"),
             配送均天=("delivery_days", "mean"))
        .reset_index()
    )
    cust["客单价"] = cust["总金额"] / cust["订单数"]      # AOV = 总 GMV / 订单数
    cust = cust.merge(score_mean, on="customer_unique_id", how="left")
    cust = cust.merge(rfm[["customer_unique_id", "stage"]], on="customer_unique_id", how="left")

    # 地区（客户级属性，取去重后首值）
    state = master[["customer_unique_id", "customer_state"]].drop_duplicates("customer_unique_id")
    cust = cust.merge(state, on="customer_unique_id", how="left")

    # 品类：订单明细 -> 品类，统计每客户 Top1 品类与品类数
    items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    prods = pd.read_csv(PROCESSED / "products_clean.csv")[["product_id", "product_category_name_english"]]
    ci = items.merge(prods, on="product_id", how="left")
    ci = ci.merge(master[["order_id", "customer_unique_id"]], on="order_id", how="left")
    ci["cat"] = ci["product_category_name_english"].fillna("unknown")
    top1 = (ci.groupby(["customer_unique_id", "cat"]).size().rename("n").reset_index()
            .sort_values(["customer_unique_id", "n"], ascending=[True, False])
            .drop_duplicates("customer_unique_id"))[["customer_unique_id", "cat"]].rename(columns={"cat": "Top1品类"})
    ncat = ci[["customer_unique_id", "cat"]].drop_duplicates().groupby("customer_unique_id")["cat"].nunique().rename("品类数")
    cust = cust.merge(top1, on="customer_unique_id", how="left").merge(ncat, on="customer_unique_id", how="left")

    # ---------- 2. 分组 ----------
    single = cust[cust["stage"] == "单次购买"]
    repeat = cust[cust["stage"] == "复购客户"]
    print(f"单次购买 {len(single):,} 人 ｜ 复购客户 {len(repeat):,} 人")

    # ---------- 3. 连续指标检验 ----------
    cont_metrics = [
        ("客单价", "客单价", "结构"),
        ("首单金额", "首单金额", "结构"),
        ("评分均值", "评分均值", "体验(描述)"),
        ("超时率", "超时率", "体验(描述)"),
        ("品类数", "品类数", "结构(注:由订单数决定)"),
    ]
    rows = []
    for name, col, dim in cont_metrics:
        a, b = single[col].dropna(), repeat[col].dropna()
        u, p, r = mw_test(a, b)
        rows.append({
            "指标": name, "维度": dim,
            "单次中位": a.median(), "复购中位": b.median(),
            "单次均值": a.mean(), "复购均值": b.mean(),
            "检验": "Mann-Whitney U", "统计量": u, "p值": p,
            "效应量r": r, "效应量级别": eff_label_r(r),
            "差异方向": "复购更高" if r > 0 else "复购更低",
        })
    cont_df = pd.DataFrame(rows)

    # ---------- 4. 类别指标检验（卡方） ----------
    cat_rows = []
    share_parts = []
    for name, col, top_n in [("Top1品类", "Top1品类", 12), ("客户州", "customer_state", 8)]:
        chi2, p, dof, v, table = chi2_test(single[col].dropna(), repeat[col].dropna(), top_n)
        cat_rows.append({
            "变量": name, "检验": "卡方", "chi2": chi2, "df": dof, "p值": p,
            "CramersV": v, "效应量级别": eff_label_v(v),
        })
        # 另存各类占比供画图
        share = table.copy()
        share["单次占比"] = share["单次"] / share["单次"].sum()
        share["复购占比"] = share["复购"] / share["复购"].sum()
        share = share.reset_index()
        share = share.rename(columns={share.columns[0]: "类别"})  # 首列=value_counts 的 index 名，按位置重命名
        share["变量"] = name
        share_parts.append(share)
    cat_df = pd.DataFrame(cat_rows)
    share_all = pd.concat(share_parts, ignore_index=True)

    # ---------- 5. 输出 ----------
    print("\n" + "=" * 90)
    print("连续指标检验结果（结论以效应量 r 为准，p 值因大样本几乎必然 < .05）")
    print("=" * 90)
    print(cont_df.round(3).to_string(index=False))

    print("\n" + "=" * 90)
    print("类别指标卡方检验结果（结论以 Cramér's V 为准）")
    print("=" * 90)
    print(cat_df.round(3).to_string(index=False))

    cont_df.to_csv(PROCESSED / "h1_test_results.csv", index=False, encoding="utf-8-sig")
    cat_df.to_csv(PROCESSED / "h1_categorical_results.csv", index=False, encoding="utf-8-sig")
    share_all.to_csv(PROCESSED / "h1_category_state_share.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存: data/processed/h1_test_results.csv / h1_categorical_results.csv / h1_category_state_share.csv")

    # ---------- 6. 图1 关键指标对比 ----------
    # 钱类指标右偏 -> 用中位数；率/评分类 -> 用均值（中位数会因大量 0 或 5 星而失真）
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    plot_metrics = [
        ("客单价",   "客单价 中位数 (R$)",  "median", "%.1f", 1),
        ("首单金额", "首单金额 中位数 (R$)", "median", "%.1f", 1),
        ("评分均值", "平均评分 (1-5)",      "mean",   "%.2f", 1),
        ("超时率",   "超时率 均值 (%)",     "mean",   "%.1f", 100),
    ]
    for ax, (col, label, stat, fmt, scale) in zip(axes.flat, plot_metrics):
        a, b = single[col].dropna(), repeat[col].dropna()
        u, p, r = mw_test(a, b)
        va = a.median() if stat == "median" else a.mean()
        vb = b.median() if stat == "median" else b.mean()
        vals = [va * scale, vb * scale]
        bars = ax.bar(["单次购买", "复购客户"], vals, color=[C1, C2], width=0.55)
        ax.bar_label(bars, fmt=fmt, fontsize=10)
        ax.set_title(f"{label}（秩双列 r={r:.2f}，{eff_label_r(r)}效应）", fontsize=10)
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.margins(y=0.18)
    fig.suptitle("H1：复购 vs 单次 关键指标对比", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "h1_metrics_compare.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- 7. 图2 品类分布差异 ----------
    cat_share = share_all[share_all["变量"] == "Top1品类"].copy()
    cat_share = cat_share[cat_share["类别"] != "其他"]
    cat_share["lift"] = cat_share["复购占比"] / cat_share["单次占比"].replace(0, np.nan)
    cat_share["显示名"] = cat_share["类别"].str.replace("_", " ")
    cat_share = cat_share.sort_values("lift", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ypos = np.arange(len(cat_share))
    ax.barh(ypos - 0.2, cat_share["单次占比"], height=0.36, color=C1, label="单次购买 Top1 品类占比")
    ax.barh(ypos + 0.2, cat_share["复购占比"], height=0.36, color=C2, label="复购客户 Top1 品类占比")
    ax.set_yticks(ypos)
    ax.set_yticklabels(cat_share["显示名"], fontsize=9)
    ax.set_xlabel("占组内客户比例")
    ax.set_title("H1：Top1 品类分布 单次 vs 复购")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", color="#E2E2E2", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "h1_category_divergence.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("已保存: docs/figures/h1_metrics_compare.png, h1_category_divergence.png")


if __name__ == "__main__":
    main()

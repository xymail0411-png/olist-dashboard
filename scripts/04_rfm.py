# -*- coding: utf-8 -*-
"""
04 RFM 用户分层（两阶段方案二）
数据源：data/processed/orders_master.csv（订单主宽表）
产出：
  - data/processed/customer_rfm.csv  客户 RFM 打分与两阶段分层
  - docs/figures/rfm_segment.png      分层贡献对比图

RFM 口径：
  R Recency   = 基准日（最晚下单日）- 最近一次下单天数（越小越活跃）
  F Frequency = 订单数（frequency）
  M Monetary  = total_value 求和（GMV 口径，price + freight）

两阶段方案（对比原"中位数二分 → 8 类"的改进，改进原因详见 docs/rfm_report.md §1.2）：
  阶段一：按 F 是否 ≥2 切分 -> 单次购买(97.0%) / 复购客户(3.0%)
  阶段二：在各自层内只使用"有区分度"的维度分层
    单次层 F 恒为 1（零区分度）-> 退化为 R×M，分 4 类：
        高潜新客 / 待培育新客 / 高价值唤醒 / 流失观望
    复购层 F 有真实差异（2~15 单）-> 经典 R×F×M 8 类：
        重要价值/重要发展/重要保持/重要挽留/一般价值/一般发展/一般保持/一般挽留
        其中 F 用业务档（F≥3 = 复购≥2 次）而非中位数——复购层 91.8% 挤在 F=2，
        中位数=2 时 "F>2" 只抓 8%，业务档更稳。

用法：
    C:\\...\\python.exe scripts/04_rfm.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

REPEAT_SEG_ORDER = [
    "重要价值客户", "重要发展客户", "重要保持客户", "重要挽留客户",
    "一般价值客户", "一般发展客户", "一般保持客户", "一般挽留客户",
]
SINGLE_SEG_ORDER = ["高潜新客", "待培育新客", "高价值唤醒", "流失观望"]


def single_segment(r, m):
    """单次层：RFM 退化为 R×M（F 恒为 1，剔除）"""
    if r and m:
        return "高潜新客"        # 近期且高额 -> 最有转化潜力的新客
    if r and not m:
        return "待培育新客"      # 近期但低额 -> 培育客单
    if not r and m:
        return "高价值唤醒"      # 沉睡但高额 -> 召回
    return "流失观望"            # 沉睡且低额


def repeat_segment(r, f, m):
    """复购层：经典 R×F×M 8 类（F 用业务档 F≥3）"""
    if r and f and m:
        return "重要价值客户"
    if r and not f and m:
        return "重要发展客户"
    if not r and f and m:
        return "重要保持客户"
    if not r and not f and m:
        return "重要挽留客户"
    if r and f and not m:
        return "一般价值客户"
    if r and not f and not m:
        return "一般发展客户"
    if not r and f and not m:
        return "一般保持客户"
    return "一般挽留客户"


def main():
    master = pd.read_csv(PROCESSED / "orders_master.csv", parse_dates=["order_purchase_timestamp"])

    # ---------- RFM 计算 ----------
    ref_date = master["order_purchase_timestamp"].max()
    rfm = (
        master.groupby("customer_unique_id")
        .agg(
            recency=("order_purchase_timestamp", lambda s: (ref_date - s.max()).days),
            frequency=("order_id", "count"),
            monetary=("total_value", "sum"),
        )
        .reset_index()
    )

    # ---------- 阶段一：单次 vs 复购 ----------
    rfm["stage"] = np.where(rfm["frequency"] >= 2, "复购客户", "单次购买")
    single = rfm[rfm["stage"] == "单次购买"].copy()
    repeat = rfm[rfm["stage"] == "复购客户"].copy()

    # ---------- 阶段二a：单次层 R×M（F 恒为 1，剔除） ----------
    r_med_s, m_med_s = single["recency"].median(), single["monetary"].median()
    single["recency_score"] = (single["recency"] <= r_med_s).astype(int)
    single["monetary_score"] = (single["monetary"] > m_med_s).astype(int)
    single["frequency_score"] = np.nan
    single["segment"] = [
        single_segment(r, m) for r, m in zip(single["recency_score"], single["monetary_score"])
    ]

    # ---------- 阶段二b：复购层 R×F×M（F 用业务档 F≥3） ----------
    r_med_r, m_med_r = repeat["recency"].median(), repeat["monetary"].median()
    repeat["recency_score"] = (repeat["recency"] <= r_med_r).astype(int)
    repeat["monetary_score"] = (repeat["monetary"] > m_med_r).astype(int)
    repeat["frequency_score"] = (repeat["frequency"] >= 3).astype(int)
    repeat["segment"] = [
        repeat_segment(r, f, m)
        for r, f, m in zip(repeat["recency_score"], repeat["frequency_score"], repeat["monetary_score"])
    ]

    rfm = pd.concat([single, repeat], ignore_index=True)

    # ---------- 切分点信息 ----------
    print("=" * 70)
    print("两阶段 RFM 切分点")
    print("=" * 70)
    print(f"阶段一：F 中位数 = {rfm['frequency'].median():.0f} 单；单次(F=1) {len(single):,} 人 "
          f"({len(single)/len(rfm):.2%})，复购(F≥2) {len(repeat):,} 人 ({len(repeat)/len(rfm):.2%})")
    print(f"单次层：R 中位数 {r_med_s:.0f} 天 (≤ 视为近期)，M 中位数 R$ {m_med_s:,.2f} (> 视为高额)")
    print(f"复购层：R 中位数 {r_med_r:.0f} 天 (≤ 视为近期)，M 中位数 R$ {m_med_r:,.2f} (> 视为高额)，"
          f"F 业务档 F≥3 ({len(repeat[repeat['frequency']>=3]):,} 人 / {(repeat['frequency']>=3).mean():.1%} 为复购≥2次)")

    # ---------- 分层汇总（单次 4 类 + 复购 8 类） ----------
    rows = []
    for stage, order in [("单次购买", SINGLE_SEG_ORDER), ("复购客户", REPEAT_SEG_ORDER)]:
        sub = rfm[rfm["stage"] == stage]
        seg_stat = (
            sub.groupby("segment")
            .agg(客户数=("customer_unique_id", "count"),
                 总订单=("frequency", "sum"),
                 总GMV=("monetary", "sum"),
                 人均消费=("monetary", "mean"),
                 人均订单=("frequency", "mean"))
            .reindex(order)
            .reset_index()
        )
        seg_stat.insert(0, "层级", stage)
        seg_stat["客户占比"] = seg_stat["客户数"] / len(rfm)
        seg_stat["GMV占比"] = seg_stat["总GMV"] / rfm["monetary"].sum()
        rows.append(seg_stat)
    seg_all = pd.concat(rows, ignore_index=True)

    print("\n" + "=" * 70)
    print("两阶段分层汇总（客户占比/GMV占比按全量客户口径）")
    print("=" * 70)
    print(seg_all.to_string(index=False))

    # ---------- 保存 ----------
    col_order = ["customer_unique_id", "recency", "frequency", "monetary",
                 "stage", "recency_score", "frequency_score", "monetary_score", "segment"]
    rfm[col_order].to_csv(PROCESSED / "customer_rfm.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存: data/processed/customer_rfm.csv（{len(rfm):,} 个客户）")

    # ---------- 分层贡献图（两张：单次 4 类 + 复购 8 类） ----------
    # 说明：两阶段各一张图。单次层用全量口径；复购层只占 3%，用"复购层内口径"放大，
    # 才能看清 8 类的相对结构（若复购 8 类画进全量口径，每类只有 0.03%~0.76%，会挤成细条）。

    # 图A 单次层 4 类（全量客户口径）
    plot_single = seg_all[seg_all["层级"] == "单次购买"].copy()[["segment", "客户占比", "GMV占比"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ypos = np.arange(len(plot_single))[::-1]
    ax.barh(ypos - 0.2, plot_single["客户占比"], height=0.38, color="#C9D0D9", label="客户占比")
    ax.barh(ypos + 0.2, plot_single["GMV占比"], height=0.38, color="#52647D", label="GMV占比")
    for y, p in zip(ypos - 0.2, plot_single["客户占比"]):
        ax.text(p + 0.004, y, f"{p:.1%}", va="center", fontsize=9, color="#6b7280")
    for y, p in zip(ypos + 0.2, plot_single["GMV占比"]):
        ax.text(p + 0.004, y, f"{p:.1%}", va="center", fontsize=9, color="#1f2937")
    ax.set_yticks(ypos)
    ax.set_yticklabels(plot_single["segment"], fontsize=10)
    ax.set_xlim(0, 0.45)
    ax.set_xlabel("全量客户口径占比")
    ax.set_title("单次层：客户占比 vs GMV 占比（4 类）")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rfm_segment_single.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 图B 复购层 8 类（复购层内部口径）
    # 8 类 + 双占比指标，横向条形必须给足行高才不会拥挤：拉高画布、按"重要/一般"分组并留空隙、
    # 放大字号，保证 8 行 × 2 组数值标签互不重叠。（散点气泡在右上/右下两点坐标过近，标注避不开）
    rep = rfm[rfm["stage"] == "复购客户"]
    rep_rows = (
        rep.groupby("segment")
        .agg(客户数=("customer_unique_id", "count"), GMV=("monetary", "sum"))
        .reindex(REPEAT_SEG_ORDER)
        .reset_index()
    )
    rep_rows["客户占比"] = rep_rows["客户数"] / len(rep)
    rep_rows["GMV占比"] = rep_rows["GMV"] / rep["monetary"].sum()
    order = ["重要价值客户", "重要发展客户", "重要保持客户", "重要挽留客户",
             None,  # 组间空隙
             "一般价值客户", "一般发展客户", "一般保持客户", "一般挽留客户"]
    val = rep_rows.set_index("segment")
    ypos = np.arange(len(order), 0, -1, dtype=float)
    labels = [s if s else "" for s in order]
    client = [val.loc[s, "客户占比"] if s else 0 for s in order]
    gmv = [val.loc[s, "GMV占比"] if s else 0 for s in order]
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    ax.barh(ypos - 0.2, client, height=0.34, color="#C9D0D9", label="复购层内客户占比")
    ax.barh(ypos + 0.2, gmv, height=0.34, color="#52647D", label="复购层内GMV占比")
    for yy, s in zip(ypos, order):
        if not s:
            continue
        c, g = val.loc[s, "客户占比"], val.loc[s, "GMV占比"]
        ax.text(c + 0.005, yy - 0.2, f"{c:.1%}", va="center", fontsize=9.5, color="#6b7280")
        ax.text(g + 0.005, yy + 0.2, f"{g:.1%}", va="center", fontsize=9.5, color="#1f2937")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 0.38)
    ax.set_xlabel("复购层内口径占比")
    ax.set_title("复购层：客户占比 vs GMV 占比（8 类，复购层内口径，重要/一般分组）")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rfm_segment_repeat.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("已保存: docs/figures/rfm_segment_repeat.png（分组横向条形版）")
    print("已保存: docs/figures/rfm_segment_single.png, rfm_segment_repeat.png")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
10 复购预测模型（预测层）：
  标签 = 客户是否复购（复购=1，单次=0）
  特征 = RFM（R/M）+ 画像（品类数/Top1占比/客单/评分/履约/周末占比）+ 州
  方法 = 逻辑回归（可解释基准） vs LightGBM（性能）
  评估 = AUC / PR-AUC / Top-K 命中率（预算视角：取 top5% 客户看抓到多少复购）
主线：诊断层说"复购率3%是结构使然"，预测层回答"在单次客里谁能被拉回来复购"，
      让有限的留旧预算投在最值的人身上。
数据源：customer_rfm.csv(stage/R/M) + customer_profile.csv(画像) + orders_master(履约)
产出：data/processed/repeat_model_data.csv、repeat_model_results.csv
      docs/figures/model_auc_compare.png、model_topk.png、model_feature_importance.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    rfm = pd.read_csv(PROCESSED / "customer_rfm.csv")   # stage, recency, monetary, ...
    master = pd.read_csv(PROCESSED / "orders_master.csv")

    # ---------- 关键：只用「首购时刻」已知的信息，避免数据泄漏 ----------
    # 复购=订单数>=2，累计量特征(n_orders/total_value/avg)会直接把标签喂给模型，AUC=1.0 是假的。
    # 因此只取每客户第 1 单的特征（下单时平台即知）：
    master = master.sort_values("order_purchase_timestamp")
    master["order_rank"] = master.groupby("customer_unique_id").cumcount() + 1
    first = master[master["order_rank"] == 1].copy()
    first["first_month"] = pd.to_datetime(first["order_purchase_timestamp"]).dt.month

    agg = first.groupby("customer_unique_id").agg(
        first_value=("total_value", "first"),
        first_score=("review_score", lambda s: s.iloc[0]),
        first_late=("is_late", lambda s: float(s.iloc[0])),
        first_delivery_days=("delivery_days", "first"),
        first_month=("first_month", "first"),
        state=("customer_state", "first"),
    ).reset_index()
    # is_late 若为 object 转 float
    agg["first_late"] = agg["first_late"].astype(float)

    # 首单品类（order_items ⨝ products_clean 取第 1 单的品类）
    items = pd.read_csv(ROOT / "data" / "olist" / "olist_order_items_dataset.csv")
    prods = pd.read_csv(PROCESSED / "products_clean.csv")[["product_id", "product_category_name_english"]]
    ci = items.merge(prods, on="product_id", how="left").merge(
        first[["order_id", "customer_unique_id"]], on="order_id", how="left")
    ci["cat"] = ci["product_category_name_english"].fillna("unknown")
    fcat = ci.groupby("customer_unique_id")["cat"].agg(lambda s: s.mode().iloc[0]).rename("first_category")
    agg = agg.merge(fcat, on="customer_unique_id", how="left")

    # 合并 stage 标签
    df = rfm[["customer_unique_id", "stage"]].merge(agg, on="customer_unique_id", how="left")
    df["label"] = (df["stage"] == "复购客户").astype(int)

    # ---------- 特征工程 ----------
    feat_cols = ["first_value", "first_score", "first_late", "first_delivery_days",
                 "first_month"]
    # 州 / 首单品类做 category encoding
    df["state_id"] = df["state"].fillna("SP").astype("category").cat.codes
    df["first_category_id"] = df["first_category"].fillna("unknown").astype("category").cat.codes
    feat_cols += ["state_id", "first_category_id"]
    df = df.dropna(subset=feat_cols).reset_index(drop=True)

    X = df[feat_cols].copy()
    y = df["label"]
    print(f"客户总数 {len(df):,}，正样本(复购) {y.sum():,}（{y.mean()*100:.2f}%）")

    # ---------- 训练/评估 ----------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

    # 逻辑回归
    lr = LogisticRegression(max_iter=2000, C=0.5)
    lr.fit(X_tr_s, y_tr)
    lr_prob = lr.predict_proba(X_te_s)[:, 1]
    lr_auc = roc_auc_score(y_te, lr_prob)
    lr_pr = average_precision_score(y_te, lr_prob)

    # LightGBM
    gbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             verbose=-1)
    gbm.fit(X_tr, y_tr)
    gbm_prob = gbm.predict_proba(X_te)[:, 1]
    gbm_auc = roc_auc_score(y_te, gbm_prob)
    gbm_pr = average_precision_score(y_te, gbm_prob)

    # ---------- Top-K 命中率（预算视角） ----------
    def topk_hit(prob, y_true, k):
        # 取概率最高的 k% 客户，看抓到多少正样本
        cutoff = int(len(prob) * k / 100)
        if cutoff == 0:
            return 0, 0
        top = np.argsort(-prob)[:cutoff]
        n_hit = int(y_true.iloc[top].sum())
        total_pos = int(y_true.sum())
        return n_hit, n_hit / total_pos

    topk_rows = []
    for k in [1, 2, 3, 5, 10]:
        lh, lr_cov = topk_hit(lr_prob, y_te, k)
        gh, gb_cov = topk_hit(gbm_prob, y_te, k)
        topk_rows.append([k, lh, lr_cov, gh, gb_cov])
    topk_df = pd.DataFrame(topk_rows, columns=["top_k%", "LR命中数", "LR覆盖率%", "GBM命中数", "GBM覆盖率%"])
    topk_df["LR覆盖率%"] = (topk_df["LR覆盖率%"] * 100).round(1)
    topk_df["GBM覆盖率%"] = (topk_df["GBM覆盖率%"] * 100).round(1)

    # ---------- 特征重要性（LightGBM） ----------
    imp = pd.DataFrame({"feature": gbm.feature_importances_}, index=gbm.feature_name_).sort_values(
        "feature", ascending=False).reset_index().rename(columns={"index": "feature", "feature": "importance"})

    results = pd.DataFrame({
        "model": ["逻辑回归", "LightGBM"],
        "AUC": [lr_auc, gbm_auc],
        "PR_AUC": [lr_pr, gbm_pr]})
    results.to_csv(PROCESSED / "repeat_model_results.csv", index=False, encoding="utf-8-sig")
    topk_df.to_csv(PROCESSED / "repeat_model_topk.csv", index=False, encoding="utf-8-sig")
    imp.to_csv(PROCESSED / "repeat_model_importance.csv", index=False, encoding="utf-8-sig")
    df.to_csv(PROCESSED / "repeat_model_data.csv", index=False, encoding="utf-8-sig")

    print("\n模型对比（测试集 30%）：")
    for _, r in results.iterrows():
        print(f"  {r['model']}: AUC={r['AUC']:.3f}  PR-AUC={r['PR_AUC']:.3f}")
    print("\nTop-K 命中（取概率最高前 k% 客户，捕获复购的覆盖率）：")
    print(topk_df.to_string(index=False))
    print("\nTop 10 特征重要性（LightGBM）：")
    print(imp.head(10).to_string(index=False))

    # ---------- 图 ----------
    # 图1 AUC/PR 对比
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2); w = 0.28
    ax.bar(x - w/2, results["AUC"], w, label="AUC", color="#52647D")
    ax.bar(x + w/2, results["PR_AUC"], w, label="PR-AUC", color="#C9D0D9")
    ax.set_xticks(x); ax.set_xticklabels(results["model"])
    for xi, (a, p) in enumerate(zip(results["AUC"], results["PR_AUC"])):
        ax.text(xi - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(xi + w/2, p + 0.01, f"{p:.3f}", ha="center", fontsize=9)
    ax.set_title("复购预测：模型 AUC / PR-AUC 对比（复购率3%严重不平衡）")
    ax.legend(frameon=False); ax.grid(axis="y", color="#E2E2E2", linewidth=.5); ax.set_axisbelow(True)
    ax.margins(y=.2)
    fig.tight_layout(); fig.savefig(FIG / "model_auc_compare.png", dpi=130); plt.close(fig)

    # 图2 Top-K 覆盖率
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = topk_df["top_k%"]
    ax.plot(x, topk_df["LR覆盖率%"], marker="o", label="逻辑回归", color="#8D9AAB")
    ax.plot(x, topk_df["GBM覆盖率%"], marker="s", label="LightGBM", color="#52647D")
    for xi, yi in zip(x, topk_df["LR覆盖率%"]):
        ax.text(xi, yi + 1, f"{yi:.0f}", ha="center", fontsize=8)
    for xi, yi in zip(x, topk_df["GBM覆盖率%"]):
        ax.text(xi, yi - 2.5, f"{yi:.0f}", ha="center", fontsize=8, color="#52647D")
    ax.set_xlabel("取概率最高前 k% 客户"); ax.set_ylabel("捕获复购覆盖率 %")
    ax.set_title("Top-K 命中：预算投前 k% 客户能抓到多少复购")
    ax.legend(frameon=False); ax.grid(color="#E2E2E2", linewidth=.5); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "model_topk.png", dpi=130); plt.close(fig)

    # 图3 特征重要性 top10
    imp10 = imp.head(10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp10["feature"][::-1], imp10["importance"][::-1], color="#52647D")
    for yi, v in zip(range(len(imp10)), imp10["importance"][::-1]):
        ax.text(v + 2, yi, str(int(v)), va="center", fontsize=8)
    ax.set_xlabel("重要性（分裂增益）"); ax.set_title("LightGBM 特征重要性 Top10")
    ax.margins(x=.15)
    fig.tight_layout(); fig.savefig(FIG / "model_feature_importance.png", dpi=130); plt.close(fig)

    print("\n已保存: repeat_model_data/results/topk/importance.csv + 3 张图")


if __name__ == "__main__":
    main()

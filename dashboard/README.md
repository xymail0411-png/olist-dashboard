# Olist 巴西电商经营分析仪表盘

交互式数据可视化作品集，基于 Olist 公开数据集（2017-01 ~ 2018-08），回答一个核心业务问题：**平台 97% 的客户只买一次，运营资源该投"拉新"还是"留旧"？**

## 在线访问

部署到 GitHub Pages 后访问：`https://<你的用户名>.github.io/<仓库名>/dashboard/`

## 本地运行

直接用浏览器打开 `index.html` 即可，无需服务器。依赖 ECharts CDN。

## 项目结构

```
dashboard/
├── index.html    # 单文件交互式仪表盘（ECharts）
└── README.md     # 本文件
```

## 仪表盘包含

- **项目概览**：分析方法论（描述→诊断→预测→处方）
- **经营总览**：KPI、月度趋势、品类、地理分布、用户画像
- **客户价值分层**：两阶段 RFM、复购客 vs 单次客画像
- **为什么只买一次**：6 个假设检验（H1~H6）
- **复购预测模型**：LightGBM，AUC=0.62，Top10% 抓 17% 复购
- **决策建议**：拉新 80% / 留旧 20%，分人群运营动作

## 技术栈

Python (pandas, scikit-learn, LightGBM) → ECharts 交互式可视化 → GitHub Pages 部署

## 数据来源

[Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)，9 张关系表，96,211 笔已交付订单。

# 数据字典：Olist Brazilian E-Commerce

- **来源**：Olist 公司官方发布（Kaggle：https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce）
- **许可**：CC BY-NC-SA 4.0（非商业）
- **数据范围**：2016-09-04 ~ 2018-10-17，巴西全境
- **说明**：真实订单数据（脱敏），共 9 张关系表。以下字段均以官方说明为准。

## 关系结构

```
customers (customer_id) 1──n orders (order_id) 1──n order_items (product_id, seller_id)
                                        │
                                        ├──n payments (order_id)
                                        └──1 reviews (order_id)
products (product_id, product_category_name) ── product_category_name_translation
sellers (seller_id)
geolocation (geolocation_zip_code_prefix) ── customers/sellers (zip_code_prefix)
```

## 表 1：olist_orders_dataset（99,441 行 · 订单主表）

| 字段 | 类型 | 说明 |
|------|------|------|
| order_id | str | 订单唯一标识（主键）|
| customer_id | str | 客户标识（外键→customers）|
| order_status | str | 订单状态：delivered / shipped / canceled / invoiced / processing / created / unavailable / approved |
| order_purchase_timestamp | datetime | 下单时间 |
| order_approved_at | datetime | 支付审批时间 |
| order_delivered_carrier_date | datetime | 交付承运商时间（未送达为空）|
| order_delivered_customer_date | datetime | 送达客户时间（未送达为空）|
| order_estimated_delivery_date | datetime | 预估送达时间 |

> 履约时效 = delivered_customer_date − purchase_timestamp（真实可算）；canceled/unavailable 为异常样本。

## 表 2：olist_customers_dataset（99,441 行 · 客户维度）

| 字段 | 说明 |
|------|------|
| customer_id | 客户主键（每笔订单一个，注意同一人可能多 ID）|
| customer_unique_id | 客户唯一 ID（去重用，96,096 个唯一客户）|
| customer_zip_code_prefix | 邮编前缀（关联 geolocation）|
| customer_city / customer_state | 城市 / 州 |

## 表 3：olist_order_items_dataset（112,650 行 · 订单明细）

| 字段 | 说明 |
|------|------|
| order_id | 订单（外键）|
| order_item_id | 订单内商品序号（1,2,3…）|
| product_id | 商品（外键→products）|
| seller_id | 卖家（外键→sellers）|
| shipping_limit_date | 卖家最迟发货期限 |
| price | 单价（不含运费）|
| freight_value | 运费 |

## 表 4：olist_order_payments_dataset（103,886 行 · 支付信息）

| 字段 | 说明 |
|------|------|
| order_id | 订单（外键）|
| payment_sequential | 支付序号（一笔订单可多次支付）|
| payment_type | credit_card / boleto / voucher / debit_card / not_defined |
| payment_installments | 分期数 |
| payment_value | 支付金额 |

## 表 5：olist_products_dataset（32,951 行 · 商品维度）

| 字段 | 说明 |
|------|------|
| product_id | 商品主键 |
| product_category_name | 类目（葡语，关联翻译表）|
| product_name_lenght | 名称字符数（官方拼写如此）|
| product_description_lenght | 描述字符数 |
| product_photos_qty | 图片数 |
| product_weight_g | 重量（克）|
| product_length_cm / height_cm / width_cm | 尺寸（厘米）|

## 表 6：olist_order_reviews_dataset（99,224 行 · 评论）

| 字段 | 说明 |
|------|------|
| review_id | 评论主键 |
| order_id | 订单（外键）|
| review_score | 1~5 星评分 |
| review_comment_title | 评论标题（可空）|
| review_comment_message | 评语正文（可空）|
| review_creation_date | 评论创建时间 |
| review_answer_timestamp | 商家回复时间 |

> 评分为数值型，可直接做情感/满意度分析；评语为文本，可做 NLP。

## 表 7：olist_sellers_dataset（3,095 行 · 卖家维度）

| 字段 | 说明 |
|------|------|
| seller_id | 卖家主键 |
| seller_zip_code_prefix / seller_city / seller_state | 卖家位置 |

## 表 8：olist_geolocation_dataset（1,000,163 行 · 地理位置）

| 字段 | 说明 |
|------|------|
| geolocation_zip_code_prefix | 邮编前缀 |
| geolocation_lat / geolocation_lng | 纬度 / 经度 |
| geolocation_city / geolocation_state | 城市 / 州 |

## 表 9：product_category_name_translation（71 行 · 类目翻译）

| 字段 | 说明 |
|------|------|
| product_category_name | 葡语类目名 |
| product_category_name_english | 英语类目名 |

## 已知数据特征

1. **时间**：2016-09 ~ 2018-10，约 25 个月
2. **状态分布**：delivered 为主，含 canceled（约 600+）等真实异常样本
3. **空值**：orders 中送达时间戳为空 = 未送达（含 canceled/unavailable）；reviews 评语可选填
4. **多对多**：一单多支付、一单多商品、一客户多单
5. **review_id 重复**：814 个 review_id 出现 2-3 次，内容/评分/时间完全相同，仅关联不同订单——官方已知特征（同一条评论被关联到同一买家的多笔订单），分析时按 `order_id + review_id` 维度去重

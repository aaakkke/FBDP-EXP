from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, sum as _sum, lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
import pandas as pd

# 初始化 Spark 会话
spark = SparkSession.builder.appName("purchase_redeem_forecast").getOrCreate()

# 加载数据
user_balance_path = "/input/user_balance_table.csv"  # 数据路径
df = spark.read.option("header", "true").csv(user_balance_path, inferSchema=True)

# 数据预处理：将 report_date 转换为日期格式，并按日期和 total_purchase_amt、total_redeem_amt 聚合
df = df.withColumn("report_date", to_date(col("report_date"), "yyyyMMdd"))

# 计算2014年8月之前每日的总申购和赎回
df_filtered = df.filter((col("report_date") >= '2014-02-05') & (col("report_date") <= '2014-08-31'))


# 按日期进行聚合，计算每日的总申购和总赎回
daily_totals = df_filtered.groupBy("report_date").agg(
    _sum("total_purchase_amt").alias("total_purchase_amt"),
    _sum("total_redeem_amt").alias("total_redeem_amt")
)

# 将数据转化为Pandas DataFrame以便操作
daily_totals_pd = daily_totals.toPandas()
daily_totals_pd['report_date'] = pd.to_datetime(daily_totals_pd['report_date'])
daily_totals_pd.set_index('report_date', inplace=True)

# 设定节假日列表，并将日期格式转换为字符串
holidays = ['2014-02-14', '2014-04-05', '2014-05-01', '2014-06-02', '2014-08-02', '2014-09-01', '2014-09-08']
holidays = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in holidays]  # 转换为字符串列表

# 创建特征：例如，日期的年份、月份、星期、节假日等
daily_totals_pd['year'] = daily_totals_pd.index.year
daily_totals_pd['month'] = daily_totals_pd.index.month
daily_totals_pd['day'] = daily_totals_pd.index.day
daily_totals_pd['day_of_week'] = daily_totals_pd.index.dayofweek
daily_totals_pd['is_holiday'] = daily_totals_pd.index.isin(holidays).astype(int)

# 将日期特征和目标变量转换为 PySpark DataFrame
spark_df = spark.createDataFrame(daily_totals_pd)

# 特征列
feature_cols = ['year', 'month', 'day', 'day_of_week', 'is_holiday']

# 特征向量化
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 初始化回归模型
purchase_rf = RandomForestRegressor(featuresCol="features", labelCol="total_purchase_amt",seed=42)
redeem_rf = RandomForestRegressor(featuresCol="features", labelCol="total_redeem_amt",seed=42)

# 构建 Pipeline
pipeline_purchase = Pipeline(stages=[assembler, purchase_rf])
pipeline_redeem = Pipeline(stages=[assembler, redeem_rf])

# 训练模型
model_purchase = pipeline_purchase.fit(spark_df)
model_redeem = pipeline_redeem.fit(spark_df)

# 生成2014年9月的日期特征
forecast_dates = pd.date_range(start="2014-09-01", periods=30, freq='D')
forecast_df = pd.DataFrame({
    'forecast_date': forecast_dates,
    'year': forecast_dates.year,
    'month': forecast_dates.month,
    'day': forecast_dates.day,
    'day_of_week': forecast_dates.dayofweek
})

# 生成2014年9月1日至30日的节假日特征
forecast_df['is_holiday'] = forecast_df['forecast_date'].isin(holidays).astype(int)

# 将预测数据转化为 Spark DataFrame
forecast_spark_df = spark.createDataFrame(forecast_df)

# 使用模型进行预测
purchase_predictions = model_purchase.transform(forecast_spark_df)
redeem_predictions = model_redeem.transform(forecast_spark_df)

# 提取预测结果
purchase_forecast = purchase_predictions.select("forecast_date", "prediction").withColumnRenamed("prediction", "predicted_purchase_amt")
redeem_forecast = redeem_predictions.select("forecast_date", "prediction").withColumnRenamed("prediction", "predicted_redeem_amt")

# 合并预测结果
forecast_result = purchase_forecast.join(redeem_forecast, on="forecast_date")

forecast_result = forecast_result.withColumn(
    "predicted_purchase_amt",
    (col("predicted_purchase_amt") * (1 + (col("forecast_date").cast("string").isin(holidays).cast("int") * 0.2))).cast("int")
)

forecast_result = forecast_result.withColumn(
    "predicted_redeem_amt",
    (col("predicted_redeem_amt") * (1 + (col("forecast_date").cast("string").isin(holidays).cast("int") * 0.2))).cast("int")
)

# 结果转换为 Pandas DataFrame 并保存
forecast_result_pd = forecast_result.select("forecast_date", "predicted_purchase_amt", "predicted_redeem_amt").toPandas()

# 输出结果
print(forecast_result_pd)

# 保存预测结果
forecast_result_pd.to_csv("tc_comp_predict_table.csv", index=False, header=False)

# 结束 Spark 会话
spark.stop()

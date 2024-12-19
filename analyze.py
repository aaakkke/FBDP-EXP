from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.use('Agg')  # 强制使用Agg后端

# 初始化Spark会话
spark = SparkSession.builder.appName("UserBalanceProcessing").getOrCreate()

# 假设数据存储为CSV文件，可以使用Spark读取数据
# 请替换为你的实际文件路径
file_path = "/input/user_balance_table.csv"
df = spark.read.option("header", "true").csv(file_path)

# 转换数据类型
df = df.withColumn("report_date", col("report_date").cast("int")) \
       .withColumn("total_purchase_amt", col("total_purchase_amt").cast("int")) \
       .withColumn("total_redeem_amt", col("total_redeem_amt").cast("int"))

# 添加日期列，并提取年和月
df = df.withColumn("date", to_date(col("report_date").cast("string"), "yyyyMMdd"))

# 过滤出2014年数据
df_2014 = df.filter(df.date.startswith("2014"))

# 按日期分组，计算每天的申购和赎回金额
daily_data = df_2014.groupBy("date").agg(
    {"total_purchase_amt": "sum", "total_redeem_amt": "sum"}
)

# 重命名列
daily_data = daily_data.withColumnRenamed("sum(total_purchase_amt)", "total_purchase_amt") \
                       .withColumnRenamed("sum(total_redeem_amt)", "total_redeem_amt")

# 将数据收集到Pandas DataFrame
daily_data_pandas = daily_data.toPandas()

# 确保日期列按升序排列
daily_data_pandas['date'] = pd.to_datetime(daily_data_pandas['date'])
daily_data_pandas.sort_values('date', inplace=True)

# 每隔5天选择一个点
daily_data_pandas_sampled = daily_data_pandas.iloc[::5]

# 绘制折线图
plt.figure(figsize=(20, 12))
plt.plot(daily_data_pandas_sampled['date'], daily_data_pandas_sampled['total_purchase_amt'], label="Total Purchase Amount", color='blue')
plt.plot(daily_data_pandas_sampled['date'], daily_data_pandas_sampled['total_redeem_amt'], label="Total Redeem Amount", color='red')

# 设置图形标题和标签
plt.title('Total Purchase and Redeem Amount per Day in 2014')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.xticks(rotation=45)

# 显示图例
plt.legend()

# 保存图像为文件，例如 PNG 文件
plt.tight_layout()
plt.savefig("plot2.png")  # 设定你想保存的路径

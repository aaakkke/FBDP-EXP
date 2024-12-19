from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg

# 初始化 SparkSession
spark = SparkSession.builder.appName("AverageBalanceByCity").getOrCreate()

# 加载数据
user_balance_df = spark.read.option("header", "true").csv("/input/user_balance_table.csv")
user_profile_df = spark.read.option("header", "true").csv("/input/user_profile_table.csv")
user_balance_df = user_balance_df.withColumn("tBalance", col("tBalance").cast("float")).withColumn("report_date", col("report_date").cast("int"))

# 筛选出2014年3月1日的记录
filtered_balance_df = user_balance_df.filter(col("report_date") == 20140301)

# 连接 user_balance_df 和 user_profile_df 按 user_id
joined_df = filtered_balance_df.join(user_profile_df, "user_id")

# 按 city 分组，计算每个城市的平均余额
city_avg_balance_df = joined_df.groupBy("city").agg(avg("tBalance").alias("avg_balance"))

sorted_city_avg_balance_df = city_avg_balance_df.orderBy(col("avg_balance").desc())
# 将结果写入到文本文件，使用 tab 作为分隔符
sorted_city_avg_balance_df.write.option("header", "false") .option("delimiter", "\t").csv("city.txt")
                                 
sorted_city_avg_balance_df.show()

# 关闭 SparkSession
spark.stop()

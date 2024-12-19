from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, rank
from pyspark.sql.window import Window


spark = SparkSession.builder.appName("TopUsersByCity").getOrCreate()


user_balance_df = spark.read.option("header", "true").csv("/input/user_balance_table.csv")
user_profile_df = spark.read.option("header", "true").csv("/input/user_profile_table.csv")

user_balance_df_filtered = user_balance_df.filter(col('report_date').rlike('^201408[0-9]{2}$'))
user_balance_df_filtered = user_balance_df_filtered.withColumn(
    'total_flow',
    col('total_purchase_amt').cast("int") + col('total_redeem_amt').cast("int")
)

# 聚合每个用户的总流量
user_total_flow_df = user_balance_df_filtered.groupBy('user_id').agg(
    sum('total_flow').alias('total_flow')
)
 
user_data_df = user_total_flow_df.join(user_profile_df, on='user_id', how='inner')
# 排名每个城市中的用户总流量
window_spec = Window.partitionBy('city').orderBy(col('total_flow').desc())
user_data_ranked_df = user_data_df.withColumn('rank', rank().over(window_spec))

# 选择排名前三的用户
top_users_df = user_data_ranked_df.filter(col('rank') <= 3)

# 选择所需字段
result_df = top_users_df.select('city', 'user_id', 'total_flow')

# 将结果保存为txt文件，不包含表头
result_df.rdd.map(lambda row: f"{row['city']} {row['user_id']} {row['total_flow']}") \
    .saveAsTextFile("Top_users_by_city.txt")

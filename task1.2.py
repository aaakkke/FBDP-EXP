from pyspark import SparkContext, SparkConf
from datetime import datetime

# 初始化 Spark
conf = SparkConf().setAppName("ActiveUsersInAugust2014")
sc = SparkContext(conf=conf)

# 读取数据
data = sc.textFile("/input/user_balance_table.csv")

# 解析每一行数据
def parse_line(line):
    parts = line.split(",")
    user_id = parts[0]  # 用户 ID
    report_date = parts[1]  # 报告日期
    return (user_id, report_date)

# 处理并解析数据
parsed_data = data.map(parse_line)

# 过滤出 2014 年 8 月的数据
def filter_august_data(record):
    report_date = record[1]
    return report_date.startswith("201408")

august_data = parsed_data.filter(filter_august_data)

# 计算每个用户的活跃天数
def count_active_days(records):
    # records 是一个 (user_id, [date1, date2, ...]) 形式的元组
    user_id = records[0]
    dates = records[1]  # 获取该用户的所有日期列表
    unique_days = set(dates)  # 使用 set 去重日期
    return (user_id, len(unique_days))  # 返回用户ID和去重后的活跃天数

# 统计每个用户的活跃天数
user_activity = august_data.groupByKey().mapValues(list)  # groupByKey 聚合相同用户的所有日期
user_active_days = user_activity.map(count_active_days)  # 计算每个用户的活跃天数

# 筛选出活跃用户（至少 5 天记录）
active_users = user_active_days.filter(lambda x: x[1] >= 5)  # 筛选活跃天数 >= 5 的用户

# 计算活跃用户总数
active_users_count = active_users.count()

# 输出结果
print(f"2014年8月的活跃用户总数是: {active_users_count}")

with open("active.txt", "w") as f:
    f.write(f"{active_users_count}")

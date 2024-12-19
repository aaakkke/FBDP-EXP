from pyspark import SparkContext
from pyspark.sql import SparkSession

# 创建SparkSession和SparkContext
spark = SparkSession.builder.appName("FundsFlowAnalysis").getOrCreate()
sc = spark.sparkContext
data_file = "/input/user_balance_table.csv"

rdd = sc.textFile(data_file)

# 定义函数来解析每一行数据
def parse_line(line):
    fields = line.split(",")
    try:
        report_date = fields[1]
        total_purchase_amt = int(fields[4]) if fields[4] else 0
        total_redeem_amt = int(fields[8]) if fields[8] else 0
        return (report_date, total_purchase_amt, total_redeem_amt)
    except IndexError:
        # 如果数据格式不符合预期，返回一个空元组
        return None

# 解析数据，去掉表头
header = rdd.first()
rdd = rdd.filter(lambda line: line != header)

# 解析数据并过滤掉无效行
parsed_rdd = rdd.map(parse_line).filter(lambda x: x is not None)


aggregated_rdd = parsed_rdd.map(lambda x: (x[0], (x[1], x[2]))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
sorted_rdd = aggregated_rdd.sortByKey()

# 转换为需要的输出格式
result_rdd = sorted_rdd.map(lambda x: f"{x[0]} {x[1][0]} {x[1][1]}")
for line in result_rdd.collect():
    print(line)

result_rdd.saveAsTextFile("fund_flow")

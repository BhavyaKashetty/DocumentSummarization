from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import split, explode, desc
from pyspark.sql.types import StructField, StringType, StructType, ArrayType,FloatType
# import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import math
from graphframes import *

def tokenize(inputDF):
    tokenizer = Tokenizer(inputCol='sentences', outputCol='tokenizedwords')
    tokenized = tokenizer.transform(inputDF)
    return tokenized


def stopwords_removal(tokenizedDF):
    remover = StopWordsRemover(inputCol='tokenizedwords', outputCol="filtered_stopwords")
    return remover.transform(tokenizedDF)

def createEdges(vertices):
    edgelist = []
    lemmatizer = WordNetLemmatizer()
    for vertex in vertices.collect():
        for u in vertices.collect():
            vertex1 = []
            vertex2 = []
            for i in vertex.filtered_stopwords:
                vertex1.append(lemmatizer.lemmatize(i))
            for j in u.filtered_stopwords:
                vertex2.append(lemmatizer.lemmatize(j))
            score = similaritymeasure(vertex1,vertex2)
            edgelist.append((vertex.sentences,u.sentences,float(score)))
    return edgelist


def similaritymeasure(vertex1,vertex2):
    lenSimilarwords = len(set(vertex1).intersection(set(vertex2)))
    score = 0
    if len(vertex1)!=0:
        logs1 = math.log(len(vertex1))
    else:
        logs1 = 0
    if len(vertex2) != 0:
        logs2 = math.log(len(vertex2))
    else:
        logs2=0
    if logs1+logs2 != 0:
        score = lenSimilarwords / (logs1 + logs2)
    return score

# read input file into dataframe
#spark = SparkSession.builder.appName("Summarization").getOrCreate()
conf = SparkConf().setAppName("Summarization").set("spark.executor.heartbeatInterval", "360000")
sc = SparkContext.getOrCreate(conf)
#sc = spark.sparkContext
input = spark.read.csv("s3://project.summary/news_summary_demo.csv", escape='"', multiLine=True, header=True, sep=",").select("ctext\r")
#input = sc.textFile("s3://project.bk/news_summary_1.csv")
input = input.na.drop()  # remove null values
input = input.withColumnRenamed("ctext\r","ctext")

# tokenize API
cols = ["ctext","originalText"]
field = [StructField("ctext",StringType(), True),StructField("originalText", StringType(), True)]
original_schema = StructType(field)
final_df = spark.createDataFrame([], original_schema)

for row in input.collect():
    df = spark.createDataFrame([], original_schema)
    text = row.ctext
    newRow = spark.createDataFrame([(text, text)], cols)
    df = df.union(newRow)
    #df = df.withColumnRenamed("ctext\r","ctext")
    sentencesDF = df.select(explode(split(df.ctext, "\.")).alias("sentences"))
    sentencesDF = sentencesDF.na.drop()
    tokenized = tokenize(sentencesDF)
    edited = stopwords_removal(tokenized)
    edgelist = createEdges(edited)
    edgeData = sc.parallelize(edgelist)
    schema = StructType([StructField("src",StringType(),True), StructField("dst",StringType(),True), StructField("score",FloatType(),True)])
    edgeDF = spark.createDataFrame(edgeData,schema)
    vertices = edited.withColumnRenamed("sentences","id")
    gFrame = GraphFrame(vertices,edgeDF)
    ranks = gFrame.pageRank(resetProbability=0.5, maxIter=20)
    sorted_ranks = ranks.vertices.orderBy(desc("pagerank")).limit(5)
    sentence_final = ""
    for srow in sorted_ranks.collect():
        sentence_final = sentence_final + srow.id + "."

    final_df_row = spark.createDataFrame([(text, sentence_final)], cols)
    final_df = final_df.union(final_df_row)
	

final_df.repartition(1).write.csv("s3://project.summary/output.prob.05.max.20/.csv")
print("End of Summarization")
import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek


config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__),'dl.cfg'))

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    
    """
    Description:
        Process the songs data files and create extract songs table and artist table data from it.
    :param spark: a spark session instance
    :param input_data: input file path
    :param output_data: output file path
    """
    # get filepath to song data file
    #{"num_songs": 1, "artist_id": "ARJIE2Y1187B994AB7", "artist_latitude": null, "artist_longitude": null, "artist_location": "", "artist_name": "Line Renaud", "song_id": "SOUPIRU12A6D4FA1E1", "title": "Der Kleine Dompfaff", "duration": 152.92036, "year": 0}
    song_data = os.path.join(input_data, "song_data/*/*/*")
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select("song_id", "title", "artist_id", "year", "duration")
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode("overwrite").partitionBy("year","artist_id").parquet(os.path.join(output_data, "songs"))

    # extract columns to create artists table
    artists_table = df.select("artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude")
    
    # write artists table to parquet files
    artists_table.write.mode("overwrite").parquet(os.path.join(output_data, "artists"))


def process_log_data(spark, input_data, output_data):
    
    """
    Description:
        Process the log data files and create  song, time_table and songplays table data from it.
    :param spark: a spark session instance
    :param input_data: input file path
    :param output_data: output file path
    """
    # get filepath to log data file
    #{"artist":"Des'ree","auth":"Logged In","firstName":"Kaylee","gender":"F","itemInSession":1,"lastName":"Summers","length":246.30812,"level":"free","location":"Phoenix-Mesa-Scottsdale, AZ","method":"PUT","page":"NextSong","registration":1540344794796.0,"sessionId":139,"song":"You Gotta Be","status":200,"ts":1541106106796,"userAgent":"\"Mozilla\/5.0 (Windows NT 6.1; WOW64) AppleWebKit\/537.36 (KHTML, like Gecko) Chrome\/35.0.1916.153 Safari\/537.36\"","userId":"8"}
    log_data = os.path.join(input_data, "log-data/")

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(col("page") == 'NextSong')

    # extract columns for users table    
    artists_table = df.select("userId", "firstName", "lastName", "gender", "level").drop_duplicates()
    
    # write users table to parquet files
    artists_table.write.mode("overwrite").parquet(os.path.join(output_data, "users"))

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.utcfromtimestamp(float(x)/1000.))
    df = df.withColumn("start_time", get_timestamp("ts"))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda: date_format(x, "yyyy-MM-dd hh:mm"))
    df = df.withColumn("time_stamp", get_datetime("ts"))
    
    # extract columns to create time table
    time_table = df.withColumn("hour", hour("start_time"))\
                   .withColumn("day", dayofmonth("start_time"))\
                   .withColumn("week", weekofyear("start_time"))\
                .withColumn("month", month("start_time"))\
                .withColumn("year", year("start_time"))\
                .withColumn("weekday", dayofweek("start_time"))\
                .select("ts", "start_time", "hour", "day", "week", "month", "year", "weekday").drop_duplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.mode("overwrite").partitionBy("year", "month").parquet(os.path.join(output_data,"time_table"))

    # read in song data to use for songplays table
    song_df = spark.read.parquet(os.path.join(output_data, "songs"))

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df.join(song_df, (col("song") == col("title"))).select(monotonically_increasing_id().alias("songplay_id"), "start_time", col("userId").alias("user_id"), "level", "song_id", "artist_id", col("sessionId").alias("session_id"), "location", col("userAgent").alias("user_agent"))
    
    songplays_table = songplays_table.join(time_table, songplays_table.start_time == time_table.start_time)\
    .select("songplay_id", songplays_table.start_time, "user_id", "level", "song_id", "artist_id", "session_id", "location", "user_agent", "year", "month")

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode("overwrite").partitionBy("year", "month").parquet(os.path.join(output_data, "songplays"))


def main():
    spark = create_spark_session()
    input_data = "s3://udacity-spark-project/"
    output_data = "s3://udacity-spark-project/output/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()

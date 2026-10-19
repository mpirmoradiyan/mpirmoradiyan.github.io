---
title: "Which row had the max? The greatest-n-per-group pattern in SQL and Spark"
slug: greatest-n-per-group
date: 2020-05-26
topic: Programming & Data
tags: [university-lab, sql, spark-sql, window-functions, group-by, dataframes]
summary: "Why GROUP BY can give you the maximum per group but not the row it came from, and the two standard fixes: join back to the aggregate, or rank rows with a window function."
sources:
  - title: "732A54 Big Data Analytics (course lab)"
    author: "Linköping University"
    type: course
  - title: "PySpark API reference: pyspark.sql.Window"
    author: "Apache Spark"
    url: "https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.Window.html"
    type: docs
  - title: "MySQL 8.0 Reference Manual: Window Function Concepts and Syntax"
    author: "Oracle"
    url: "https://dev.mysql.com/doc/refman/8.0/en/window-functions-usage.html"
    type: docs


status: draft
---

"What was the highest temperature each year?" is one line of SQL. "Which station measured it?" sounds like the same question, but it isn't. I ran into this in the Spark SQL lab of my Big Data course (a pair lab), where we had to report the yearly max and min temperature together with the station. It turns out this comes up all the time (latest order per customer, best model per experiment, top reading per sensor), and it even has a name: the *greatest-n-per-group* problem.

## Why GROUP BY alone doesn't work

The first thing you try is to just add `station` to the query. Here's why that can't work. `GROUP BY year` squashes all the rows of a year into one row. `MAX(temperature)` is fine, because it's computed from the whole group. But `station` has many different values inside the group, and nothing says which one belongs on the output row:

```sql
-- Rejected by PostgreSQL, and by MySQL 5.7.5+ (ONLY_FULL_GROUP_BY is on by default):
SELECT year, station, MAX(temperature) FROM readings GROUP BY year;
```

Older MySQL, or MySQL with `ONLY_FULL_GROUP_BY` turned off, accepts this and gives you the station from *some* row in the group, not necessarily the row with the max. That's a silent wrong answer, which is worse than an error.

## Fix 1: aggregate, then join back

First compute the max per group. Then join it back to the original table, matching on both the group key and the value:

```sql
SELECT r.year, r.station, r.temperature
FROM readings r
JOIN (SELECT year, MAX(temperature) AS t_max
      FROM readings GROUP BY year) m
  ON r.year = m.year AND r.temperature = m.t_max;
```

This is an inner join on purpose: we only want the rows that match the max. (If inner vs left joins are fuzzy, I wrote about them in [inner vs left joins](/notes/sql-inner-vs-outer-joins/).)

This is what we did in the lab with Spark DataFrames: `groupBy('year').agg(F.max(...))`, then an inner join on `year` and temperature. It works. But watch out for **ties**: if two stations both hit 36.1 in the same year, you get both rows. Sometimes that's what you want, sometimes not. It's also two passes over the data plus a join.

## Fix 2: rank the rows inside each group

A window function computes something over a set of related rows *without* squashing them into one. `PARTITION BY` says which rows belong together. `ORDER BY` sorts the rows inside each of those groups. (MySQL has window functions from version 8.0.)

```sql
SELECT year, station, temperature
FROM (SELECT year, station, temperature,
             ROW_NUMBER() OVER (PARTITION BY year ORDER BY temperature DESC) AS rn
      FROM readings) ranked
WHERE rn = 1;
```

So every row keeps its station, gets a rank within its year, and you keep rank 1. You also get to choose what happens with ties. `ROW_NUMBER()` picks exactly one row (an arbitrary one among ties, unless you add a tie-breaker to `ORDER BY`). `RANK()` gives tied rows the same rank, so `rn = 1` returns all of them. And "top 3 per group" is just `rn <= 3`.

The same thing in PySpark:

```python
from pyspark.sql import Window, functions as F

w = Window.partitionBy("year").orderBy(F.desc("temperature"))
yearly_max = (readings
              .withColumn("rn", F.row_number().over(w))
              .filter(F.col("rn") == 1)
              .select("year", "station", "temperature"))
```

In the plain RDD API there's a third way, with no join at all: make the value a `(temperature, station)` tuple and `reduceByKey(max)`. Python compares tuples element by element, so the pair with the highest temperature wins (on a tie, the station ID that sorts last wins). That's the same key-value thinking as in my note on [Spark key-value aggregation](/notes/spark-key-value-aggregation/).

## A trap I hit along the way: reusing the same DataFrame name

This has nothing to do with window functions, but it bit us in the same script. We kept writing

```python
df = df.filter(df["year"].between("1950", "2014"))
df = df.filter(df["temperature"] > 10)        # needed for one question only
...
df = df.filter(df["year"].between("1960", "2014"))   # later question: monthly averages
```

Each line *replaces* `df`. So every later question was quietly computed on readings above 10 degrees only. In our revised script, the monthly average temperatures and the per-station maximum temperatures were both built from that filtered frame. For the monthly averages that's clearly wrong: you can't average winter months from readings above 10 °C. DataFrames are immutable and a filter is cheap to describe, so there's no reason to reuse one name. Give each filtered version its own name (`readings_50_14`, `warm_readings`) and start each question from the right one.

## Key takeaways

- `GROUP BY` gives aggregates per group, not the row an aggregate came from.
- Fix 1: aggregate, then join back on group key and value. Ties return several rows.
- Fix 2: `ROW_NUMBER()` / `RANK()` over `PARTITION BY group ORDER BY value DESC`, keep rank 1. Clear tie handling and easy top-n.
- Both work the same in SQL and in Spark DataFrames.
- Don't overwrite the same DataFrame variable with question-specific filters.

## Resources

- 732A54 Big Data Analytics, Linköping University, Spark SQL lab (2020).
- Apache Spark, [pyspark.sql.Window](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.Window.html).
- MySQL 8.0 Reference Manual, [Window Function Concepts and Syntax](https://dev.mysql.com/doc/refman/8.0/en/window-functions-usage.html).

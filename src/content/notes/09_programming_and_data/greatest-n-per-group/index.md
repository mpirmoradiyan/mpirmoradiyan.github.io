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

"What was the highest temperature each year?" is one line of SQL. "Which station measured it?" is surprisingly not. This comes up all the time (latest order per customer, best model per experiment, top reading per sensor) and it has a name: the *greatest-n-per-group* problem. I met it in the Spark SQL lab of my Big Data course (a pair lab), where we had to report the yearly max and min temperature together with the station.

## Why GROUP BY alone doesn't work

`GROUP BY year` collapses all rows of a year into one. Aggregates like `MAX(temperature)` are fine, because they're computed from the group. But `station` has many different values inside the group, and there's no rule saying which one belongs on the output row:

```sql
-- Not valid in standard SQL (and in MySQL with ONLY_FULL_GROUP_BY):
SELECT year, station, MAX(temperature) FROM readings GROUP BY year;
```

Some databases used to accept this and return the station from *some* row in the group, not necessarily the one with the max. That's a silent wrong answer, which is worse than an error.

## Fix 1: aggregate, then join back

Compute the max per group, then join it back to the original table on both the group key and the value:

```sql
SELECT r.year, r.station, r.temperature
FROM readings r
JOIN (SELECT year, MAX(temperature) AS t_max
      FROM readings GROUP BY year) m
  ON r.year = m.year AND r.temperature = m.t_max;
```

This is what we did in the lab with Spark DataFrames: `groupBy('year').agg(F.max(...))`, then an inner join on `year` and temperature. It works, but note the behaviour with **ties**: if two stations both hit 36.1 in the same year, you get both rows. Sometimes that's what you want, sometimes it isn't. It's also two passes over the data plus a join.

## Fix 2: rank inside each group with a window function

A window function computes something over a set of related rows *without* collapsing them. `PARTITION BY` says which rows belong together, `ORDER BY` sorts inside each partition:

```sql
SELECT year, station, temperature
FROM (SELECT year, station, temperature,
             ROW_NUMBER() OVER (PARTITION BY year ORDER BY temperature DESC) AS rn
      FROM readings) ranked
WHERE rn = 1;
```

Every row keeps its station, gets a rank within its year, and you keep rank 1. You choose how to handle ties: `ROW_NUMBER()` picks exactly one row (arbitrarily among ties unless you add a tie-breaker to `ORDER BY`), while `RANK()` gives tied rows the same rank, so `rn = 1` returns all of them. It also generalizes to "top 3 per group" by changing the filter to `rn <= 3`.

The same thing in PySpark:

```python
from pyspark.sql import Window, functions as F

w = Window.partitionBy("year").orderBy(F.desc("temperature"))
yearly_max = (readings
              .withColumn("rn", F.row_number().over(w))
              .filter(F.col("rn") == 1)
              .select("year", "station", "temperature"))
```

## A trap I hit along the way: reassigning the same DataFrame variable

This one has nothing to do with window functions, but it bit us in the same script. We kept writing

```python
df = df.filter(df["year"].between("1950", "2014"))
df = df.filter(df["temperature"] > 10)        # needed for one question only
...
df = df.filter(df["year"].between("1960", "2014"))   # later question: monthly averages
```

Each line *replaces* `df`, so every later question was silently computed on readings above 10 degrees. In our revised script, the monthly average temperatures and the per-station maximum temperatures were both built from that filtered frame. For the monthly averages that's clearly wrong (winter months can't be averaged from readings above 10 °C only). DataFrames are immutable and filters are cheap to describe, so there's no reason to reuse one name: give each filtered version its own name (`readings_50_14`, `warm_readings`) and start each question from the right one.

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

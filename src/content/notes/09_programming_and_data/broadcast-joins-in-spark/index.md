---
title: "Broadcast the small side: joining big data with a small lookup table in Spark"
slug: broadcast-joins-in-spark
date: 2020-06-08
topic: Programming & Data
tags: [university-lab, spark, broadcast-variables, joins, big-data, pyspark]
summary: "Why you should ship a small lookup table to every worker instead of pulling the big dataset back to the driver, and how broadcast variables do that in Spark."
sources:
  - title: "732A54 Big Data Analytics (course lab)"
    author: "Linköping University"
    type: course
  - title: "RDD Programming Guide: Broadcast Variables"
    author: "Apache Spark"
    url: "https://spark.apache.org/docs/latest/rdd-programming-guide.html#broadcast-variables"
    type: docs
  - title: "Spark SQL Performance Tuning: Join Strategy Hints"
    author: "Apache Spark"
    url: "https://spark.apache.org/docs/latest/sql-performance-tuning.html"
    type: docs


status: draft
---

A very common shape in data work: one huge table (millions of sensor readings) and one small table (a few hundred stations with their coordinates or region). You need information from the small one for every row of the big one. In Spark the right move is to send the small table to every worker and do the lookup there. In my Big Data labs we did the opposite at first, and the code only ran on the small test files. This note is what I'd tell myself back then. (These were pair labs.)

## Where the data lives

A Spark program has one **driver** (the Python process running your script) and many **executors** (worker processes on the cluster). An RDD or DataFrame is split into partitions that sit on the executors. Two calls move data across that boundary:

- `rdd.collect()` pulls *every* element back into the driver's memory as a Python list.
- `sc.parallelize(list)` sends a Python list from the driver out to the executors as a new RDD.

`collect()` on a big dataset defeats the purpose of Spark: everything goes through one machine, and if the data doesn't fit in the driver's memory the job dies.

## The anti-pattern we wrote

In one lab we needed only the stations from one region. Our code collected the *whole* monthly precipitation result to the driver, filtered it with a Python list comprehension, and then parallelized it back:

```python
totals = station_month_precip.reduceByKey(add).collect()      # big -> driver
totals = sc.broadcast(totals).value                           # does nothing useful here
region = [x for x in totals if x[0][0] in region_stations]    # filtering on one machine
rdd = sc.parallelize(region)                                  # driver -> cluster again
```

The `sc.broadcast(...).value` in the middle shows the misunderstanding. Calling `.value` right away on the driver just hands back the same Python object. Broadcasting only helps when the broadcast object is used *inside* functions that run on the executors.

In the kernel regression lab it got worse: we collected all temperature readings, matched them to station distances with a nested list comprehension on the driver, and parallelized the result. Our report said honestly that the code worked on small data but not on the full dataset, and that the broadcasting approach was the problem. We had it backwards: we broadcast the wrong thing. The revised version of that lab switched to the pattern below (station coordinates collected with `collectAsMap()`, broadcast, and looked up inside a `map` over the readings), and that one ran on the full data.

## The right pattern

Collect the **small** table, turn it into a dictionary, broadcast it, and use it inside `map` or `filter` on the **big** RDD:

```python
# small: station -> (lat, lon), a few hundred entries
stations = sc.textFile("stations.csv").map(lambda l: l.split(";")) \
             .map(lambda r: (r[0], (float(r[3]), float(r[4])))).collectAsMap()
bc = sc.broadcast(stations)          # shipped once to each executor, read-only

# big: millions of readings; the lookup happens on the executors
readings = sc.textFile("temperature-readings.csv").map(lambda l: l.split(";"))
with_coords = readings.filter(lambda r: r[0] in bc.value) \
                      .map(lambda r: (r[0], bc.value[r[0]], r[1], r[2], float(r[3])))
```

Nothing big ever comes back to the driver. Each executor gets one copy of the dictionary (instead of one copy per task, which is what happens if you just reference a normal Python variable in a lambda), and every row is enriched where it already lives. A set works the same way for pure filtering ("keep only stations in this region").

## Broadcast join vs shuffle join

You could also do `big.join(small)`. A regular join is a **shuffle join**: both sides are repartitioned by key so that matching keys end up on the same executor. That means moving the big table across the network. A **broadcast join** skips that: the small side is copied to every executor and the big side stays put.

As a rough cost picture, with $N$ rows in the big table, $M$ in the small one and $E$ executors:

$$
\text{shuffle join moves} \approx N + M \text{ rows}, \qquad \text{broadcast join moves} \approx E \cdot M \text{ rows}.
$$

When $M$ is tiny and $N$ is huge, $E \cdot M \ll N$, so broadcasting wins by a lot. When both tables are big, broadcasting would copy a big table to every executor and blow up memory, so a shuffle join is the right choice.

With DataFrames, Spark does this automatically when one side is smaller than `spark.sql.autoBroadcastJoinThreshold` (10 MB by default), and you can ask for it explicitly:

```python
from pyspark.sql import functions as F
joined = readings_df.join(F.broadcast(stations_df), on="station")
```

## Key takeaways

- `collect()` brings everything to one machine. Use it only on results you know are small.
- `sc.broadcast(x).value` on the driver does nothing special. Broadcasts matter when used inside executor-side functions.
- Collect the small table into a dict or set, broadcast it, and look things up inside `map`/`filter` on the big RDD.
- Broadcast join = copy small side everywhere; shuffle join = repartition both sides. Pick broadcast when one side is small.
- If code works on a sample but dies on the full data, look for a `collect()` on something big.

## Resources

- 732A54 Big Data Analytics, Linköping University, Spark labs 1 and 3 (2020).
- Apache Spark, [RDD Programming Guide: Broadcast Variables](https://spark.apache.org/docs/latest/rdd-programming-guide.html#broadcast-variables).
- Apache Spark, [Spark SQL Performance Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html) (join strategy hints, `autoBroadcastJoinThreshold`).

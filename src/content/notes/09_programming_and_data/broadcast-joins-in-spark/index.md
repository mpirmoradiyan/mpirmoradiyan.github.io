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

In my Big Data labs we had code that ran fine on the small test files and then died on the real data. The shape of the problem is very common: one huge table (millions of sensor readings) and one small table (a few hundred weather stations with their coordinates or region). For every row of the big table you need something from the small one. The trick is to send the small table to every worker and do the lookup there. We did the opposite at first. This note is what I'd tell myself back then. (These were pair labs.)

## First, where does the data live?

This is the part I hadn't really pictured. A Spark program has one **driver**, which is the Python process running your script. It also has many **executors**, which are worker processes on the cluster. An RDD or DataFrame is cut into pieces (partitions), and those pieces sit on the executors, not on the driver.

Two calls move data across that line:

- `rdd.collect()` pulls *every* element back into the driver's memory as a Python list.
- `sc.parallelize(list)` sends a Python list from the driver out to the executors as a new RDD.

So `collect()` on a big dataset throws away the whole point of Spark. Everything squeezes through one machine. If it doesn't fit in the driver's memory, the job dies.

## What we wrote first

In one lab we only needed the stations from one region. Our code collected the *whole* monthly precipitation result to the driver, filtered it with a normal Python list comprehension, and then sent it back out:

```python
totals = station_month_precip.reduceByKey(add).collect()      # big -> driver
totals = sc.broadcast(totals).value                           # does nothing useful here
region = [x for x in totals if x[0][0] in region_stations]    # filtering on one machine
rdd = sc.parallelize(region)                                  # driver -> cluster again
```

That `sc.broadcast(...).value` line in the middle shows what I didn't get. Calling `.value` right away, on the driver, just gives you back the same Python object. Nothing is gained. A broadcast only helps when you use it *inside* functions that run on the executors.

The kernel regression lab was worse. We collected all temperature readings, matched them to station distances with a nested list comprehension on the driver, and parallelized the result. Our report said, honestly, that the code worked on small data but not on the full dataset, and that the broadcasting was the problem. We had it backwards: we broadcast the wrong thing. The revised version switched to the pattern below (station coordinates collected with `collectAsMap()`, broadcast, and looked up inside a `map` over the readings). That one ran on the full data.

## The right way round

Collect the **small** table, make it a dictionary, broadcast it, and use it inside `map` or `filter` on the **big** RDD:

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

Now nothing big ever comes back to the driver. Each executor gets one copy of the dictionary. (If you just used a normal Python variable inside the lambda, Spark would ship a copy with every task instead.) Every row gets its coordinates right where it already lives. A set works the same way if you only want to filter, like "keep only stations in this region".

## Broadcast join vs shuffle join

Why not just `big.join(small)`? A normal join is a **shuffle join**. Spark moves both tables around the cluster so that rows with the same key end up on the same executor. That means sending the big table over the network. A **broadcast join** skips that. The small table is copied to every executor, and the big one stays where it is. (Which rows survive a join, inner or outer, is a separate question; I wrote about that in [inner vs left joins](/notes/sql-inner-vs-outer-joins/).)

A rough way to see the cost: say the big table has $N$ rows, the small one $M$ rows, and there are $E$ executors.

$$
\text{shuffle join moves} \approx N + M \text{ rows}, \qquad \text{broadcast join moves} \approx E \cdot M \text{ rows}.
$$

When $M$ is tiny and $N$ is huge, $E \cdot M \ll N$, so broadcasting wins by a lot. When both tables are big, broadcasting would copy a big table to every executor and blow up memory. Then the shuffle join is the right choice.

With DataFrames, Spark does this on its own when it estimates one side is smaller than `spark.sql.autoBroadcastJoinThreshold` (10 MB by default). You can also ask for it directly:

```python
from pyspark.sql import functions as F
joined = readings_df.join(F.broadcast(stations_df), on="station")
```

The shuffle side of all this, how `reduceByKey` combines values per key across partitions, is in my note on [key-value aggregation in Spark](/notes/spark-key-value-aggregation/).

## Key takeaways

- `collect()` brings everything to one machine. Use it only on results you know are small.
- `sc.broadcast(x).value` on the driver does nothing special. Broadcasts matter when they're used inside executor-side functions.
- Collect the small table into a dict or set, broadcast it, and look things up inside `map`/`filter` on the big RDD.
- Broadcast join = copy the small side everywhere; shuffle join = move both sides by key. Pick broadcast when one side is small.
- If code works on a sample but dies on the full data, look for a `collect()` on something big.

## Resources

- 732A54 Big Data Analytics, Linköping University, Spark labs 1 and 3 (2020).
- Apache Spark, [RDD Programming Guide: Broadcast Variables](https://spark.apache.org/docs/latest/rdd-programming-guide.html#broadcast-variables).
- Apache Spark, [Spark SQL Performance Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html) (join strategy hints, `autoBroadcastJoinThreshold`).

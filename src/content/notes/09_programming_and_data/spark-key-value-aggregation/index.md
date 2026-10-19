---
title: "Thinking in key-value pairs: aggregations with Spark RDDs"
slug: spark-key-value-aggregation
date: 2020-05-25
topic: Programming & Data
tags: [university-lab, spark, mapreduce, pyspark, big-data, aggregation]
summary: "How to turn a 'per group' question into map and reduceByKey steps in Spark, and why the reduce function has to be associative and commutative."
sources:
  - title: "732A54 Big Data Analytics (course lab)"
    author: "Linköping University"
    type: course
  - title: "RDD Programming Guide"
    author: "Apache Spark"
    url: "https://spark.apache.org/docs/latest/rdd-programming-guide.html"
    type: docs
  - title: "MapReduce: Simplified Data Processing on Large Clusters"
    author: "Jeffrey Dean, Sanjay Ghemawat"
    url: "https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/"
    type: paper


status: draft
---

The first Spark lab in my Big Data course was all about weather data: millions of temperature and precipitation readings, one row per station per timestamp. Every question was some version of "for each X, compute Y". In Spark's low-level RDD API, the way to answer that is always the same: turn each row into a `(key, value)` pair, then let Spark combine all values that share a key. Once that clicked, most questions became a small puzzle about what the key and the value should be. (This was a pair lab; the code was written together with a classmate.)

## The basic recipe

An RDD (resilient distributed dataset) is a big list split into partitions that live on different machines. You don't loop over it. You describe steps:

1. **map**: turn each row into `(key, value)`. The key is the "for each" part of the question; the value is what you want to combine.
2. **filter**: drop rows you don't need, as early as possible, so less data moves around.
3. **reduceByKey(f)**: for each key, combine all its values with a function `f(a, b)` that takes two values and returns one.

For "highest temperature per year" that's:

```python
from pyspark import SparkContext
sc = SparkContext(appName="yearly-max")

rows = sc.textFile("temperature-readings.csv").map(lambda l: l.split(";"))
# station;date;time;temperature;quality  ->  (year, temperature)
pairs = rows.map(lambda r: (r[1][:4], float(r[3])))
pairs = pairs.filter(lambda kv: 1950 <= int(kv[0]) <= 2014)
yearly_max = pairs.reduceByKey(max)          # max(a, b) is safe, see below
yearly_max.sortBy(lambda kv: kv[1], ascending=False).take(5)
```

## Why the reduce function must be associative and commutative

`reduceByKey` does not see all values of a key at once. It first combines values inside each partition (a local "combiner" step, so less data has to travel over the network), then merges the partial results from different partitions, in whatever order they arrive. So for the result to be correct, `f` has to give the same answer no matter how the values are grouped and ordered:

$$
f(f(a, b), c) = f(a, f(b, c)) \quad\text{(associative)}, \qquad f(a, b) = f(b, a) \quad\text{(commutative)}.
$$

`max`, `min` and `+` pass. An average does not.

This is the mistake I actually made. For the daily mean temperature we wanted $(T_{\min} + T_{\max})/2$ per station and day, and our code was:

```python
daily = pairs.reduceByKey(lambda a, b: (min(a, b) + max(a, b)) / 2)   # wrong
```

But $(\min(a,b) + \max(a,b))/2$ is just $(a+b)/2$, the average of two numbers. With three readings, $((a+b)/2 + c)/2$ gives $c$ half the weight, and the answer depends on the order Spark happens to use. It's neither the day's mean nor $(T_{\min}+T_{\max})/2$. The fix is to carry everything you need through the reduce, and only compute the final number afterwards:

```python
# value = (min, max); both parts combine safely
daily = pairs.mapValues(lambda t: (t, t)) \
             .reduceByKey(lambda a, b: (min(a[0], b[0]), max(a[1], b[1]))) \
             .mapValues(lambda mm: (mm[0] + mm[1]) / 2)
```

The same trick gives a true mean: map each value to `(value, 1)`, reduce with element-wise `+` to get `(sum, count)`, then divide once at the end. We did use this `(sum, count)` pattern for the monthly averages, so we knew the idea; we just didn't notice that the daily step broke it.

## Re-keying between steps

Many questions need two levels of grouping. "Maximum daily precipitation per station" means: first sum the readings for each `(station, day)`, then take the max over days for each `station`. Between the two steps you have to **change the key**:

```python
daily_total = precip.map(lambda r: ((r[0], r[1]), float(r[3]))).reduceByKey(lambda a, b: a + b)
station_max = daily_total.map(lambda kv: (kv[0][0], kv[1])).reduceByKey(max)   # key is now station
```

In our lab code we skipped the middle `map`, so the second `reduceByKey` ran on the same `(station, day)` key and did nothing. Then we joined that RDD with one keyed by station. A join only matches identical keys, and `"12345"` never equals `("12345", "2010-07-01")`, so the join came back empty. Our report said "there is no data under the given conditions". Looking back, that empty result was most likely caused by the key mismatch, not by the data. I can't tell from what's left of the lab what the correct answer would have been.

A related slip from the Spark SQL version of the same lab: we took the max of single precipitation readings instead of daily totals. "Daily" in the question means "sum per day first".

## Counting distinct things

"How many different stations had a reading above 10 degrees in each month?" The trick is to make the pair itself the thing you want to be unique, call `distinct()`, and only then count:

```python
month_station = hot.map(lambda r: (r[1][:7], r[0])).distinct()     # (year-month, station), unique
counts = month_station.map(lambda kv: (kv[0], 1)).reduceByKey(lambda a, b: a + b)
```

## Key takeaways

- Every "for each X, compute Y" question becomes `map` to `(X, value)` then `reduceByKey`.
- The reduce function must be associative and commutative, because Spark combines partial results in any grouping and order. `max`, `min`, `+` are fine; "average of two" is not.
- For means, carry `(sum, count)` (or `(min, max)`) through the reduce and divide at the end.
- Two-level questions need a `map` that changes the key between the two reduces. A join on mismatched keys silently returns nothing, so an empty result is a reason to check your keys.
- Filter early: less data to shuffle between machines.

## Resources

- 732A54 Big Data Analytics, Linköping University, Spark lab 1 (2020).
- Apache Spark, [RDD Programming Guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html), sections on key-value pairs and shuffle operations.
- Jeffrey Dean and Sanjay Ghemawat, [MapReduce: Simplified Data Processing on Large Clusters](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/), OSDI 2004. Section on combiner functions explains why the reduce step must be associative and commutative.

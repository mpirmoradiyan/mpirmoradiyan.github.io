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

The first Spark lab in my Big Data course was all weather data: millions of temperature and precipitation readings, one row per station per timestamp. Every question was some version of "for each X, compute Y". At first each question felt like a new problem. Then it clicked that in Spark's low-level RDD API the answer always has the same shape. Turn each row into a `(key, value)` pair, and let Spark combine all the values that share a key. After that, most questions became a small puzzle: what should the key be, and what should the value be? (This was a pair lab; we wrote the code together.)

## The basic recipe

An RDD (resilient distributed dataset) is basically a big list cut into pieces (partitions) that live on different machines. You don't loop over it. You describe steps, and Spark runs them where the data is:

1. **map**: turn each row into `(key, value)`. The key is the "for each" part of the question. The value is the thing you want to combine.
2. **filter**: drop rows you don't need, as early as you can, so less data has to move around.
3. **reduceByKey(f)**: for each key, combine all its values with a function `f(a, b)` that takes two values and gives back one.

For "highest temperature per year" that looks like this:

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

## Why the reduce function has to be associative and commutative

This is where I got caught. `reduceByKey` never sees all the values of a key at once. It first combines values inside each partition. (That's a local "combiner" step, so less data has to travel over the network.) Then it merges those partial results from different partitions, in whatever order they show up. So `f` has to give the same answer no matter how the values are grouped and in what order:

$$
f(f(a, b), c) = f(a, f(b, c)) \quad\text{(associative)}, \qquad f(a, b) = f(b, a) \quad\text{(commutative)}.
$$

`max`, `min` and `+` pass. An average doesn't.

Here's the mistake I actually made. For the daily mean temperature we wanted $(T_{\min} + T_{\max})/2$ per station and day. Our code was:

```python
daily = pairs.reduceByKey(lambda a, b: (min(a, b) + max(a, b)) / 2)   # wrong
```

Look at it again: $(\min(a,b) + \max(a,b))/2$ is just $(a+b)/2$, the average of two numbers. With three readings you get $((a+b)/2 + c)/2$, which gives $c$ half the weight. The answer depends on the order Spark happens to use. It's neither the day's mean nor $(T_{\min}+T_{\max})/2$.

The fix is to carry everything you need through the reduce, and only compute the final number at the end:

```python
# value = (min, max); both parts combine safely
daily = pairs.mapValues(lambda t: (t, t)) \
             .reduceByKey(lambda a, b: (min(a[0], b[0]), max(a[1], b[1]))) \
             .mapValues(lambda mm: (mm[0] + mm[1]) / 2)
```

The same trick gives a true mean. Map each value to `(value, 1)`, reduce with element-wise `+` to get `(sum, count)`, then divide once at the end. The funny part is that we *did* use this `(sum, count)` pattern for the monthly averages. So we knew the idea. We just didn't notice that the daily step broke it.

## Changing the key between steps

Some questions need two levels of grouping. "Maximum daily precipitation per station" really means two things: first sum the readings for each `(station, day)`, then take the max over the days for each `station`. Between those two steps you have to **change the key**:

```python
daily_total = precip.map(lambda r: ((r[0], r[1]), float(r[3]))).reduceByKey(lambda a, b: a + b)
station_max = daily_total.map(lambda kv: (kv[0][0], kv[1])).reduceByKey(max)   # key is now station
```

In our lab code we skipped that middle `map`. So the second `reduceByKey` ran on the same `(station, day)` key and did nothing. Then we joined that RDD with one keyed by station. A join only matches identical keys, and `"12345"` never equals `("12345", "2010-07-01")`, so the join came back empty. Our report said "there is no data under the given conditions". Looking back, the key mismatch alone makes that join empty, whatever the data says. I can't tell from what's left of the lab what the correct answer would have been.

A related slip, from the Spark SQL version of the same lab: we took the max of single precipitation readings instead of daily totals. "Daily" in the question means "sum per day first".

If you need the whole row that holds the max (like *which* station had the hottest reading), that's its own pattern; see [greatest-n-per-group](/notes/greatest-n-per-group/). And when one side of a join is small, there's a better tool than a plain join: [broadcast it](/notes/broadcast-joins-in-spark/).

## Counting distinct things

"How many different stations had a reading above 10 degrees in each month?" The trick is to make the pair itself the thing that should be unique, call `distinct()`, and only then count:

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

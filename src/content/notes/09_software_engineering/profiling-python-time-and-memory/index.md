---
title: "Profiling Python: finding where the time and the memory go"
slug: profiling-python-time-and-memory
date: 2023-05-18
topic: Software Engineering
tags: [python, profiling, performance, cprofile, line-profiler, timeit, memory, tracemalloc]
summary: "How to measure where a Python program spends its time (cProfile, line_profiler, timeit) and its memory (sys.getsizeof, deep sizes, tracemalloc), and the traps in each measurement."
sources:
  - title: "Fast Python: High Performance Techniques for Large Datasets (ch. 2, Extracting maximum performance from built-in features)"
    author: "Tiago Rodrigues Antão"
    url: "https://www.manning.com/books/fast-python"
    type: book
  - title: "Python docs: The Python Profilers (cProfile, pstats), timeit, tracemalloc, sys.getsizeof"
    url: "https://docs.python.org/3/library/profile.html"
    type: docs
  - title: "line_profiler"
    url: "https://github.com/pyutils/line_profiler"
    type: repo
  - title: "SnakeViz"
    url: "https://jiffyclub.github.io/snakeviz/"
    type: repo


status: draft
---

Guessing where a program is slow is usually wrong. A profiler tells you which function, and then which line, takes the time; a memory measurement tells you which object is big. This note covers the tools I use for both, from the standard library first, and the traps that make a measurement lie. All numbers below come from Python 3.13 on a 64-bit Mac.

## Step 1: which function? cProfile

`cProfile` is built in. It records every function call and how long it took. Run a whole script under it and sort by cumulative time:

```bash
python -m cProfile -s cumulative my_script.py | head -20
```

As an example, here is a deliberately slow synthetic script: for 2,000 random points it finds each point's nearest neighbour with a pure-Python loop.

```python
import random

def make_points(n):
    return [(random.random(), random.random()) for _ in range(n)]

def nearest_distance(p, points):
    return min(((p[0]-q[0])**2 + (p[1]-q[1])**2) ** 0.5 for q in points if q is not p)

def mean_nearest(points):
    total = 0.0
    for p in points:
        total += nearest_distance(p, points)
    return total / len(points)

if __name__ == "__main__":
    random.seed(0)
    print(mean_nearest(make_points(2000)))
```

The top of the cProfile report (trimmed):

```text
4009199 function calls in 1.532 seconds
ncalls  tottime  cumtime  filename:lineno(function)
     1    0.001    1.528  slow.py:10(mean_nearest)
  2000    0.001    1.527  slow.py:6(nearest_distance)
  2000    0.413    1.527  {built-in method builtins.min}
4000000   1.114    1.114  slow.py:7(<genexpr>)
```

Two columns matter. **`tottime`** is the time spent inside the function itself. **`cumtime`** also includes everything it calls. Here almost all of the 1.5 s is in the generator expression inside `nearest_distance`, which runs four million times (2,000 × 2,000). That is the place to optimize.

Two practical notes:

- Profiling adds overhead to every call, so a program with many tiny calls looks slower under cProfile than it really is. Use the report for **proportions**, not exact times.
- For larger programs, save the raw data and look at it as a picture: `python -m cProfile -o run.prof my_script.py`, then `snakeviz run.prof` opens an interactive "icicle" chart in the browser where the width of each box is its share of the time.

## Step 2: which line? line_profiler

cProfile works per function. When one function is long, you want per-line times. The `line_profiler` package does that for the functions you mark with `@profile`:

```bash
pip install line_profiler
kernprof -l my_script.py               # runs the script, writes my_script.py.lprof
python -m line_profiler my_script.py.lprof
```

With `@profile` on `mean_nearest`, the report shows that the single line `total += nearest_distance(p, points)` takes 100% of the time, 2,000 hits at about 0.95 ms each. Line profiling has a much larger overhead than cProfile (here the run took 1.9 s instead of 1.5 s), so use it only on the functions cProfile has already pointed to. Remove the decorators afterwards, or the script will fail with a `NameError` when run without `kernprof`.

## Step 3: did my fix help? timeit

Once you know the hot spot, try a faster version and compare the two with `timeit`. It runs a statement many times and avoids common timing mistakes. Take the **minimum** over several repeats: the minimum is the run least disturbed by other processes.

```python
import timeit

setup = ("import random, numpy as np; random.seed(0); "
         "pts = [(random.random(), random.random()) for _ in range(2000)]; P = np.array(pts)")
pure = "min(((pts[0][0]-q[0])**2 + (pts[0][1]-q[1])**2) ** 0.5 for q in pts[1:])"
vec  = "np.sqrt(((P[1:] - P[0])**2).sum(axis=1)).min()"

for name, stmt in [("pure Python", pure), ("NumPy", vec)]:
    best = min(timeit.repeat(stmt, setup, number=200, repeat=5)) / 200
    print(f"{name:12s} {best*1e6:8.1f} us per call")
# pure Python     478.7 us per call
# NumPy            31.5 us per call
```

The vectorized version is about 15 times faster for one query point. (For the full nearest-neighbour problem the real fix is a better algorithm, such as a KD-tree, but the measuring workflow is the same.) In Jupyter, `%timeit` and `%prun` do the same jobs as `timeit` and cProfile.

## Measuring memory: sys.getsizeof is shallow

`sys.getsizeof(obj)` returns the size in bytes of the object itself. My first experiments printed these values:

| Object | `sys.getsizeof` (bytes) |
|---|---|
| `1` | 28 |
| `2**100` | 40 |
| `"1"` | 42 |
| `(1,)` | 48 |
| `[1]` | 64 |
| `np.array([1])` | 120 |
| `{1}` | 216 |
| `{1: 1}` | 224 |
| `range(10**6)` | 48 |
| `list(range(10**6))` | 8,000,056 |

Everything in Python is an object with a header (a reference count and a type pointer), so even the integer 1 takes 28 bytes. Exact numbers change between Python versions: with the older Python I used in 2023, `"1"` was 50 bytes and `{1: 1}` was 232.

The trap is that **for containers, `getsizeof` counts only the container**, not the objects inside it. A list stores 8-byte pointers to its elements. So `list(range(10**6))` reports 8 MB, which is just the pointer array. A list of three strings of 1,000 characters each reports only 80 bytes. To get the real footprint you must follow the references:

```python
import sys

def deep_getsizeof(obj, seen=None):
    """Size of obj plus everything it references (each object counted once)."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_getsizeof(k, seen) + deep_getsizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_getsizeof(x, seen) for x in obj)
    return size

d = dict(zip(range(1_000_000), range(1_000_000)))
print(sys.getsizeof(d), deep_getsizeof(d))   # 41943128  97935932
```

The dictionary itself is 42 MB; with its million key objects it is 98 MB. The `seen` set matters: objects can be shared, and CPython caches the small integers from -5 to 256, so they exist only once however many lists use them. The function above handles only the built-in containers; libraries such as Pympler (`asizeof`) cover more types.

## Memory of NumPy and pandas objects

A NumPy array keeps its numbers in one raw buffer, not as Python objects. `a.nbytes` gives the size of that buffer. For an array that owns its data, `getsizeof` is `nbytes` plus a ~112-byte header (`np.zeros(10**6)`: 8,000,112 vs 8,000,000). For a **view**, such as `a[::2]`, `getsizeof` reports only 112 bytes because the view does not own the buffer, while `nbytes` reports 4,000,000, the size of the data it refers to. Neither number alone tells you whether memory is shared.

For pandas, use `df.memory_usage(deep=True)`. Without `deep=True`, a column of strings is counted as 8-byte pointers. For a 1,000-row frame with an integer column and a short string column, the shallow count was 16 kB and the deep count 62 kB.

## Measuring memory over time: tracemalloc

The size of one object is not the whole story; what often matters is the **peak** memory during a computation. The built-in `tracemalloc` module records every allocation Python makes:

```python
import tracemalloc

tracemalloc.start()
squares = [i * i for i in range(1_000_000)]       # eager: all values stored at once
print(tracemalloc.get_traced_memory())             # current and peak, about 40 MB
del squares

tracemalloc.reset_peak()
total = sum(i * i for i in range(1_000_000))       # lazy: one value at a time
print(tracemalloc.get_traced_memory())             # peak about 1 kB
tracemalloc.stop()
```

The list version needs 40 MB at its peak; the generator version needs about a kilobyte, because each square is created, added and discarded. This is the same idea as `range(10**6)` taking 48 bytes: a `range` stores only start, stop and step and produces numbers on demand. `tracemalloc.take_snapshot()` can also list the source lines that allocated the most memory, which makes it the memory equivalent of cProfile. It sees only allocations made through Python's allocator, so memory used by some C libraries may be missing; for a whole-process view, look at the resident set size (RSS) with `psutil` or a tool such as `memray`.

## Key takeaways

- Profile before optimizing: cProfile to find the function, line_profiler to find the line, timeit to compare fixes.
- Read `tottime` (own time) and `cumtime` (including callees) together; trust proportions more than absolute times under a profiler.
- `sys.getsizeof` is shallow: for containers it counts only pointers, not the elements. Follow references for a deep size, and count shared objects once.
- For arrays use `nbytes`; for DataFrames use `memory_usage(deep=True)`; views share memory with their base.
- Use `tracemalloc` to find peak memory; generators and `range` keep peak memory low by producing values on demand.

## Resources

- Tiago Rodrigues Antão, *Fast Python: High Performance Techniques for Large Datasets*, Manning, 2023, chapter 2 (profiling and the memory cost of Python objects). https://www.manning.com/books/fast-python
- Python documentation: The Python Profilers (`cProfile`, `pstats`), `timeit`, `tracemalloc`, `sys.getsizeof`. https://docs.python.org/3/library/profile.html
- line_profiler (kernprof). https://github.com/pyutils/line_profiler
- SnakeViz, a browser-based viewer for cProfile output. https://jiffyclub.github.io/snakeviz/
- Pympler (`asizeof`) for deep object sizes. https://pympler.readthedocs.io/
- For parallel speed-ups once the hot spot is known, see [Parallel Python: threads, processes, asyncio and the GIL](/notes/python-parallelism-gil/).

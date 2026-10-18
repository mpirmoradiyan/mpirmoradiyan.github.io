---
title: "Choosing a C++ STL container: complexity, memory layout and the usual traps"
slug: choosing-cpp-stl-containers
date: 2025-03-30
topic: Software Engineering
tags: [cpp, stl, data-structures, complexity, vector, map, unordered_map, priority-queue]
summary: "A practical guide to picking between vector, deque, list, set/map, unordered_set/unordered_map and the adapters (stack, queue, priority_queue), based on what each one costs and how it sits in memory."
sources:
  - title: "Data Structures and Algorithms with the C++ STL"
    author: "John Farrier (Packt, 2024)"
    type: book
  - title: "cppreference.com: Containers library"
    url: "https://en.cppreference.com/w/cpp/container"
    type: docs
  - title: "Effective STL"
    author: "Scott Meyers (Addison-Wesley, 2001)"
    type: book


status: draft
---

The C++ standard library offers about a dozen containers, and most of them can hold the same data. The difference is in what each operation costs and how the elements are laid out in memory. Picking well is mostly a matter of asking two questions: *which operations will I do most often?* and *do I need the elements to stay in order?* This note collects the answers in one place, with the corrections I had to make to my own first notes.

## The complexity table

| Container | Structure | Access by index | Find by value / key | Insert / erase at end | Insert / erase at front | Insert / erase in the middle | Ordered? |
|---|---|---|---|---|---|---|---|
| `array<T, N>` | fixed contiguous block | $O(1)$ | $O(n)$ | not possible (fixed size) | not possible | not possible | insertion order |
| `vector<T>` | growable contiguous block | $O(1)$ | $O(n)$ | $O(1)$ amortised | $O(n)$ | $O(n)$ | insertion order |
| `deque<T>` | array of fixed-size blocks | $O(1)$ | $O(n)$ | $O(1)$ | $O(1)$ | $O(n)$ | insertion order |
| `list<T>` | doubly linked nodes | $O(n)$ | $O(n)$ | $O(1)$ | $O(1)$ | $O(1)$ *given an iterator* | insertion order |
| `set` / `map` (and `multi…`) | balanced tree (red–black) | no | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | sorted by key |
| `unordered_set` / `unordered_map` | hash table | no | $O(1)$ average, $O(n)$ worst | $O(1)$ average | n/a | n/a | no order |

The adapters `stack`, `queue` and `priority_queue` are thin wrappers around one of the sequence containers. `stack` and `queue` give $O(1)$ push and pop at the allowed ends. `priority_queue` is a binary heap stored in a `vector`: $O(1)$ to read the top, $O(\log n)$ to push or pop.

Big-O hides the constant factor, and for containers that factor is dominated by **memory layout**. A `vector` keeps its elements side by side, so the CPU loads them a cache line at a time and prefetches the next ones. A `list` or a `map` allocates every element as a separate node somewhere on the heap and follows pointers to reach it. Each step can be a cache miss, which costs far more than the comparison or addition you wanted to do. That is why a linear scan of a `vector` often beats a "faster" structure for small and medium sizes.

## Sequence containers

**`std::vector`** is the default choice. Random access is $O(1)$. `push_back` is $O(1)$ amortised: when capacity runs out, the vector allocates a bigger block (typically 1.5× or 2× larger), moves every element over, and frees the old one. Two habits make it faster:

```cpp
std::vector<Employee> staff;
staff.reserve(1000);               // one allocation up front
staff.emplace_back(1, "Ada");      // constructed in place, no temporary
staff.emplace_back(2, "Alan");

// erase-remove idiom: compact the survivors, then cut off the tail
staff.erase(std::remove_if(staff.begin(), staff.end(),
                           [](const Employee& e) { return e.id == 2; }),
            staff.end());
std::erase_if(staff, [](const Employee& e) { return e.id == 2; });  // C++20 shorthand
```

`std::remove_if` does not delete anything. It moves the elements to keep to the front and returns an iterator to the new logical end; `erase` then shrinks the vector. Since C++20, `std::erase_if` does both in one call. Remember also that any reallocation **invalidates all iterators, pointers and references** into the vector.

**`std::deque`** adds $O(1)$ insertion and removal at the front. It is not one contiguous block: it keeps a small index array (the "map") of pointers to fixed-size blocks. Indexing therefore costs two lookups instead of one, but growing at either end never moves existing elements. My first notes said the blocks are "linked together to form a single contiguous sequence"; that is wrong. Only each block is contiguous. The deque is the default underlying container for `stack` and `queue`, and a natural fit for sliding windows.

**`std::list`** is a doubly linked list. Its famous $O(1)$ insertion in the middle only holds **once you already have an iterator** to the position. Finding that position is a linear walk. Its real strengths are that iterators stay valid when other elements are inserted or removed, and that `splice` can move nodes between lists in $O(1)$ without copying:

```cpp
std::list<int> a = {1, 2, 3}, b = {10, 20};
auto it = std::next(a.begin());    // finding the position is O(n)...
a.insert(it, 99);                  // ...inserting there is O(1)
a.splice(a.end(), b);              // move all of b into a in O(1), no copies
```

Each node carries two pointers, so a `list<int>` uses several times the memory of a `vector<int>`. Use `std::forward_list` if one direction is enough. In practice, a `vector` with occasional $O(n)$ inserts is usually faster than a `list` until the elements are large or the inserts very frequent.

## Ordered associative containers: `set`, `map`, `multiset`, `multimap`

These keep their keys sorted in a balanced binary search tree, usually a red–black tree, so search, insertion and deletion are all $O(\log n)$. Choose them when order matters: iterating in key order, range queries with `lower_bound` / `upper_bound`, or finding the nearest key. The `multi` versions allow repeated keys; `equal_range` returns all entries for one key.

The most common bug is `operator[]` on a map. It **inserts** a default-constructed value when the key is missing, so a lookup quietly grows the map. It also cannot be used on a `const` map.

```cpp
std::map<std::string, int> counts;
if (counts["missing"] == 0) {}                         // inserts "missing" -> 0
if (auto f = counts.find("other"); f != counts.end()) {} // look up without inserting
bool has = counts.contains("other");                   // C++20
```

## Unordered containers: `unordered_set`, `unordered_map`

These are hash tables: an array of buckets, where the key's hash picks the bucket. Lookups are $O(1)$ on average. The worst case is $O(n)$, when many keys land in the same bucket because of a poor hash function or adversarial input. Note that this applies to *access* as well; my first notes listed access as plain $O(1)$.

The **load factor** is the number of elements divided by the number of buckets. When it passes `max_load_factor()` (1.0 by default), the table rehashes: it allocates more buckets and redistributes every element, which is an $O(n)$ pause and invalidates iterators. If you know the final size, call `reserve(n)` once. Use `try_emplace` or `emplace` to insert only when the key is new, and `find` or `contains` to check without inserting.

Choose an unordered container when you only need "is this key present?" or "what is its value?" and never need order. Keys of your own types need a hash function and an equality operator.

## Adapters

- **`std::stack`** (LIFO): backtracking, undo, evaluating expressions in reverse Polish notation.
- **`std::queue`** (FIFO): breadth-first search, task queues.
- **`std::priority_queue`**: always gives the highest-priority element first. It is a *max*-heap by default. For a min-heap, such as in Dijkstra's shortest-path algorithm, pass `std::greater<>` as the comparator:

```cpp
std::priority_queue<std::pair<int, int>,               // (distance, vertex)
                    std::vector<std::pair<int, int>>,
                    std::greater<>> pq;                 // smallest distance on top
```

`priority_queue` has no "decrease key" operation, so Dijkstra simply pushes a vertex again when its distance improves. That leaves stale entries in the heap. When you pop a pair whose distance is larger than the best distance already recorded for that vertex, skip it. My first implementation left this check out. It still gave correct distances, but it re-expanded vertices for nothing.

## A short decision guide

1. Start with `vector`. Switch only when profiling or a clear requirement says so.
2. Need fast insertion at both ends? `deque`.
3. Need stable iterators while inserting and erasing in the middle, or $O(1)$ splicing? `list`.
4. Need keys in sorted order or range queries? `map` / `set`.
5. Need only fast lookup by key? `unordered_map` / `unordered_set`, with `reserve`.
6. Need "always the smallest / largest next"? `priority_queue`.
7. For small, rarely changing lookup tables, a sorted `vector` with `std::lower_bound` is often faster than `map`, thanks to the cache.

## Key takeaways

- Complexity tells you how cost grows; memory layout decides the constant, and contiguous storage usually wins.
- `vector` is the default. Use `reserve`, `emplace_back`, and the erase–remove idiom (`std::erase_if` in C++20).
- `list` inserts in $O(1)$ only once you hold an iterator; finding the spot is $O(n)$.
- `map::operator[]` inserts missing keys; use `find` or `contains` for lookups.
- Hash tables are $O(1)$ on average only; manage the load factor and `reserve` ahead of time.

## Resources

- John Farrier, *Data Structures and Algorithms with the C++ STL*, Packt, 2024. My exercise files follow its chapters on sequence, associative and adapter containers.
- cppreference.com, *Containers library*, including the per-container complexity and iterator-invalidation rules. <https://en.cppreference.com/w/cpp/container>
- Scott Meyers, *Effective STL*, Addison-Wesley, 2001 (Item 32 on the erase–remove idiom, Item 24 on `map::operator[]` versus `insert`).
- Bjarne Stroustrup, keynote at GoingNative 2012 (the "vector vs. list" demonstration of why cache locality beats linked lists).

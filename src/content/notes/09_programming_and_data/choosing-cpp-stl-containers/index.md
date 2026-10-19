---
title: "Choosing a C++ STL container: complexity, memory layout and the usual traps"
slug: choosing-cpp-stl-containers
date: 2025-03-30
topic: Programming & Data
tags: [cpp, stl, data-structures, complexity, vector, map, unordered_map, priority-queue]
summary: "How to pick between vector, deque, list, set/map, unordered_set/unordered_map and the adapters (stack, queue, priority_queue), based on what each one costs and how it sits in memory."
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

When I started working through a book on the C++ STL, the first thing that confused me was the choice. There are about a dozen containers, and most of them can hold the same data. So why so many? The answer turned out to be simple. They differ in how much each operation costs and in how the elements sit in memory. Picking one mostly comes down to two questions: *which operations will I do most?* and *do I need the elements to stay in order?* I'm still learning C++, so these are learning notes. I put the answers in one place here, including a few things I got wrong in my first notes and had to fix.

## The cost table

I made this table first, so I could look things up quickly.

| Container | Structure | Access by index | Find by value / key | Insert / erase at end | Insert / erase at front | Insert / erase in the middle | Ordered? |
|---|---|---|---|---|---|---|---|
| `array<T, N>` | fixed contiguous block | $O(1)$ | $O(n)$ | not possible (fixed size) | not possible | not possible | insertion order |
| `vector<T>` | growable contiguous block | $O(1)$ | $O(n)$ | $O(1)$ amortised | $O(n)$ | $O(n)$ | insertion order |
| `deque<T>` | array of fixed-size blocks | $O(1)$ | $O(n)$ | $O(1)$ | $O(1)$ | $O(n)$ | insertion order |
| `list<T>` | doubly linked nodes | $O(n)$ | $O(n)$ | $O(1)$ | $O(1)$ | $O(1)$ *given an iterator* | insertion order |
| `set` / `map` (and `multi…`) | balanced tree (red–black) | no | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | sorted by key |
| `unordered_set` / `unordered_map` | hash table | no | $O(1)$ average, $O(n)$ worst | $O(1)$ average | n/a | n/a | no order |

The adapters `stack`, `queue` and `priority_queue` are thin wrappers around one of the sequence containers. `stack` and `queue` give $O(1)$ push and pop at the ends they allow. `priority_queue` is a binary heap stored in a `vector`: $O(1)$ to read the top, $O(\log n)$ to push or pop.

But the table isn't the whole story, and this took me a while to get. Big-O hides the constant factor. For containers, that constant is mostly about **memory layout**. Picture it like this. A `vector` keeps its elements side by side, so the CPU loads them a cache line at a time and fetches the next ones early. A `list` or a `map` puts every element in its own node somewhere on the heap, and the program follows pointers to reach it. Each hop can be a cache miss. A cache miss costs way more than the comparison or addition you actually wanted. That's why a plain linear scan over a `vector` often beats a "faster" structure for small and medium sizes.

## Sequence containers

**`std::vector`** is the default. Access by index is $O(1)$. `push_back` is $O(1)$ amortised. That just means cheap on average. When it runs out of room, the vector grabs a bigger block (usually 1.5× or 2× bigger), moves every element over, and frees the old one. Two habits make it faster:

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

This tripped me up at first: `std::remove_if` doesn't actually delete anything. It moves the elements you want to keep to the front and returns an iterator to the new "logical" end. Then `erase` shrinks the vector. Since C++20, `std::erase_if` does both in one call. Also worth remembering: any reallocation **invalidates all iterators, pointers and references** into the vector.

**`std::deque`** adds $O(1)$ insert and remove at the front. It's not one contiguous block. It keeps a small index array (called the "map") of pointers to fixed-size blocks. So indexing costs two lookups instead of one, but growing at either end never moves the existing elements. My first notes said the blocks are "linked together to form a single contiguous sequence". That's wrong: only each block on its own is contiguous. The deque is the default container under `stack` and `queue`, and it's a natural fit for sliding windows.

**`std::list`** is a doubly linked list. Its famous $O(1)$ insert in the middle only holds **once you already have an iterator** to that spot. Finding the spot is a linear walk. Where it really shines is that iterators stay valid when other elements are added or removed, and `splice` can move nodes between lists in $O(1)$ without copying:

```cpp
std::list<int> a = {1, 2, 3}, b = {10, 20};
auto it = std::next(a.begin());    // finding the position is O(n)...
a.insert(it, 99);                  // ...inserting there is O(1)
a.splice(a.end(), b);              // move all of b into a in O(1), no copies
```

Each node carries two pointers, so a `list<int>` uses several times the memory of a `vector<int>`. Use `std::forward_list` if one direction is enough. In practice, a `vector` with the odd $O(n)$ insert is usually faster than a `list`, unless the elements are big or the inserts happen all the time.

## Sorted containers: `set`, `map`, `multiset`, `multimap`

These keep their keys sorted in a balanced binary search tree, usually a red–black tree, so search, insert and delete are all $O(\log n)$. Pick them when order matters: walking through keys in order, range queries with `lower_bound` / `upper_bound`, or finding the nearest key. The `multi` versions allow repeated keys, and `equal_range` gives you all entries for one key.

The bug I'd watch for here is `operator[]` on a map. If the key is missing, it **inserts** a default value, so a simple lookup quietly grows the map. It also can't be used on a `const` map.

```cpp
std::map<std::string, int> counts;
if (counts["missing"] == 0) {}                         // inserts "missing" -> 0
if (auto f = counts.find("other"); f != counts.end()) {} // look up without inserting
bool has = counts.contains("other");                   // C++20
```

## Hash containers: `unordered_set`, `unordered_map`

These are hash tables. There's an array of buckets, and the key's hash decides which bucket it goes in. Lookups are $O(1)$ on average. The worst case is $O(n)$, when lots of keys land in the same bucket because of a bad hash function or nasty input. That worst case applies to *access* too; my first notes said access was plain $O(1)$, which isn't quite right.

The **load factor** is the number of elements divided by the number of buckets. When it goes past `max_load_factor()` (1.0 by default), the table rehashes: it makes more buckets and moves every element. That's an $O(n)$ pause, and it invalidates iterators. If you know the final size, call `reserve(n)` once. Use `try_emplace` or `emplace` to insert only when the key is new, and `find` or `contains` to check without inserting.

So basically: use a hash container when you only care about "is this key here?" or "what's its value?" and never need order. If the keys are your own types, you need to give them a hash function and an equality operator.

## Adapters

- **`std::stack`** (last in, first out): backtracking, undo, evaluating expressions in reverse Polish notation.
- **`std::queue`** (first in, first out): breadth-first search, task queues.
- **`std::priority_queue`**: always hands you the highest-priority element first. By default it's a *max*-heap. For a min-heap, like in Dijkstra's shortest-path algorithm, pass `std::greater<>` as the comparator:

```cpp
std::priority_queue<std::pair<int, int>,               // (distance, vertex)
                    std::vector<std::pair<int, int>>,
                    std::greater<>> pq;                 // smallest distance on top
```

`priority_queue` has no "decrease key" operation, so Dijkstra just pushes a vertex again when its distance gets better. That leaves old, stale entries in the heap. When you pop a pair whose distance is bigger than the best distance already recorded for that vertex, skip it. My first try left this check out. The distances still came out right, but it re-expanded vertices for nothing.

## A quick way to choose

1. Start with `vector`. Only switch when profiling or a clear need says so.
2. Need fast inserts at both ends? `deque`.
3. Need iterators that stay valid while inserting and erasing in the middle, or $O(1)$ splicing? `list`.
4. Need keys in sorted order or range queries? `map` / `set`.
5. Only need fast lookup by key? `unordered_map` / `unordered_set`, with `reserve`.
6. Need "always the smallest / largest next"? `priority_queue`.
7. For small lookup tables that rarely change, a sorted `vector` with `std::lower_bound` is often faster than a `map`, thanks to the cache.

## Key takeaways

- Big-O tells you how the cost grows. Memory layout decides the constant, and side-by-side storage usually wins.
- `vector` is the default. Use `reserve`, `emplace_back`, and the erase–remove idiom (`std::erase_if` in C++20).
- `list` inserts in $O(1)$ only once you have an iterator. Finding the spot is $O(n)$.
- `map::operator[]` inserts missing keys. Use `find` or `contains` to look things up.
- Hash tables are only $O(1)$ on average. Watch the load factor and `reserve` ahead of time.

## Resources

- John Farrier, *Data Structures and Algorithms with the C++ STL*, Packt, 2024. My exercise files follow its chapters on sequence, associative and adapter containers.
- cppreference.com, *Containers library*, including the per-container complexity and iterator-invalidation rules. <https://en.cppreference.com/w/cpp/container>
- Scott Meyers, *Effective STL*, Addison-Wesley, 2001 (Item 32 on the erase–remove idiom, Item 24 on `map::operator[]` versus `insert`).
- Bjarne Stroustrup, keynote at GoingNative 2012 (the "vector vs. list" demonstration of why cache locality beats linked lists).

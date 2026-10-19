---
title: "Inner vs left joins: where rows quietly go missing"
slug: sql-inner-vs-outer-joins
date: 2020-11-11
topic: Programming & Data
tags: [university-lab, sql, joins, databases, aggregation]
summary: "What inner and left outer joins actually return, why groups with no matches vanish from an inner join, and the WHERE-clause trap that turns a left join back into an inner one."
sources:
  - title: "732A57 Database Technology (course lab)"
    author: "Linköping University"
    type: course
  - title: "Fundamentals of Database Systems, 7th ed., Chapters 7-8 (SQL joins and relational algebra)"
    author: "Ramez Elmasri, Shamkant B. Navathe"
    type: book
  - title: "MySQL 8.0 Reference Manual: JOIN Clause"
    author: "Oracle"
    url: "https://dev.mysql.com/doc/refman/8.0/en/join.html"
    type: docs


status: draft
---

Joins look simple until a report is missing rows and nothing errors. The first SQL lab in my database course (pair work, MySQL, a small store database with suppliers, items and sales) had several questions where the only difference between a right and a wrong answer was the join type. I'm a learner with SQL, and this is the part I most want to remember.

## The mental model

Think of a join as: take every pair of rows (one from each table), keep the pairs where the `ON` condition is true.

- **INNER JOIN**: only the matched pairs. A row with no partner on the other side disappears.
- **LEFT (OUTER) JOIN**: all matched pairs, *plus* every row from the left table that matched nothing, with `NULL` filling the right table's columns.
- RIGHT JOIN is the mirror image; FULL OUTER JOIN keeps unmatched rows from both sides (MySQL doesn't have it; you emulate it with a `UNION` of a left and a right join).

The "implicit" syntax `FROM a, b WHERE a.x = b.x` is an inner join written differently. There's no way to write an outer join that way, which is one reason to prefer the explicit `JOIN ... ON` form.

## A small example

Three suppliers. Acme's items have sold, Bolt has an item that never sold, Cogs supplies nothing yet.

```sql
-- supplier(id, name), item(id, name, supplier), sale(id, item, quantity)
SELECT s.name, SUM(sa.quantity) AS sold
FROM supplier s
JOIN item i  ON i.supplier = s.id
JOIN sale sa ON sa.item   = i.id
GROUP BY s.name;
-- Acme | 6           <- Bolt is gone: its item had no sale to pair with
```

If the question is "every supplier that has supplied something, with how much of it has sold", Bolt should be there with 0. The sale table is where rows go missing, so that join must be a LEFT JOIN:

```sql
SELECT s.name,
       COALESCE(SUM(sa.quantity), 0) AS sold,   -- SUM over only NULLs is NULL
       COUNT(sa.id) AS n_sales,                 -- counts non-NULL values: 0 for Bolt
       COUNT(*)     AS n_rows                   -- counts rows: 1 for Bolt (the padded row)
FROM supplier s
JOIN item i       ON i.supplier = s.id          -- inner: only suppliers with items
LEFT JOIN sale sa ON sa.item    = i.id          -- keep items with no sales
GROUP BY s.name;
-- Acme | 6 | 3 | 3
-- Bolt | 0 | 0 | 1
```

Two details: `SUM` of nothing but `NULL`s is `NULL`, hence `COALESCE`; and `COUNT(*)` counts the padded row, so count a column from the right table instead.

The lab had exactly this kind of question (include suppliers whose items haven't sold yet). The course handed out a starting query built with implicit, inner-join syntax, and the point was to see why it drops those suppliers. Our submitted file stopped at that starting query, so that question was left unfinished. The answer is the left-join pattern above.

When is an inner join *correct*? When every row on the left is guaranteed a match. For "total cost of each sale", every sale row references an existing item (a non-null foreign key), so inner and left give the same result and inner says what you mean.

## The WHERE trap

This one is subtle. Suppose you only want to count sales with quantity above 1, and you write:

```sql
... LEFT JOIN sale sa ON sa.item = i.id
WHERE sa.quantity > 1
```

For Bolt's padded row, `sa.quantity` is `NULL`, and `NULL > 1` is not true, so `WHERE` throws the row away. The left join has silently become an inner join. Put conditions on the right-hand table into the `ON` clause instead:

```sql
... LEFT JOIN sale sa ON sa.item = i.id AND sa.quantity > 1
```

Now the filter decides which sales match, and unmatched items still survive with `NULL`s.

In our lab report, one query used a self-join of the parts table (`LEFT JOIN parts t ON t.name = 'card reader' WHERE f.weight > t.weight`) to avoid a subquery. It gave the right answer, but only because the `WHERE` turned it into an inner join, which is what that question needed anyway. Writing `JOIN` would have said that honestly.

## Deleting with foreign keys

A related lesson from the same lab: you can't delete a row that other rows still reference through a foreign key (MySQL reports an integrity constraint error). You have to delete the referencing rows first, children before parents (sales, then items, then the supplier), or define the foreign key with `ON DELETE CASCADE` if that's really the intended behaviour. Joins and foreign keys are two views of the same links between tables.

## Key takeaways

- Inner join keeps matched pairs only; left join also keeps unmatched left rows with `NULL`s.
- If a group "with zero" must appear in the result, the join that might find nothing has to be an outer join.
- With outer joins, use `COALESCE(SUM(...), 0)` and `COUNT(right_table.col)`, not `COUNT(*)`.
- A `WHERE` condition on the right table turns a left join into an inner join. Put it in `ON`.
- Prefer explicit `JOIN ... ON` syntax; implicit comma joins can only express inner joins.

## Resources

- 732A57 Database Technology, Linköping University, SQL lab (2020).
- Ramez Elmasri and Shamkant B. Navathe, *Fundamentals of Database Systems*, 7th ed., Chapter 7 (joined tables and outer joins in SQL) and Chapter 8 (relational algebra joins).
- MySQL 8.0 Reference Manual, [JOIN Clause](https://dev.mysql.com/doc/refman/8.0/en/join.html).

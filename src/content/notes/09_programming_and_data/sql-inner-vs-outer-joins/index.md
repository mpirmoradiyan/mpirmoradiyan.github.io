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

Joins look simple until a report is missing rows and nothing gives an error. The first SQL lab in my database course (pair work, MySQL, a small store database with suppliers, items and sales) had several questions where the only difference between a right and a wrong answer was the join type. I'm a learner with SQL, and this is the part I most want to remember.

## How I picture a join

Take every possible pair of rows, one from each table. Keep the pairs where the `ON` condition is true. That's it.

- **INNER JOIN**: only the matched pairs. A row with no partner on the other side just disappears.
- **LEFT (OUTER) JOIN**: all matched pairs, *plus* every row from the left table that matched nothing. For those rows, the right table's columns are filled with `NULL`.
- RIGHT JOIN is the mirror image. FULL OUTER JOIN keeps unmatched rows from both sides. MySQL doesn't have FULL OUTER JOIN; you build it from a `UNION` of a left and a right join. (PostgreSQL does have it.)

You can also write `FROM a, b WHERE a.x = b.x`. That's an inner join in different clothes. In standard SQL there's no way to write an outer join in that comma style, which is one good reason to always use the explicit `JOIN ... ON` form.

## A small example

Three suppliers. Acme's items have sold. Bolt has one item that never sold. Cogs doesn't supply anything yet.

```sql
-- supplier(id, name), item(id, name, supplier), sale(id, item, quantity)
SELECT s.name, SUM(sa.quantity) AS sold
FROM supplier s
JOIN item i  ON i.supplier = s.id
JOIN sale sa ON sa.item   = i.id
GROUP BY s.name;
-- Acme | 6           <- Bolt is gone: its item had no sale to pair with
```

Say the question is "every supplier that has supplied something, with how much of it has sold". Then Bolt should be there with 0. (Cogs is supposed to be missing, since it has supplied nothing.) So where do rows go missing? At the sale table. That's the join that has to become a LEFT JOIN:

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

Two small details. `SUM` over nothing but `NULL`s gives `NULL`, not 0, so you need `COALESCE`. And `COUNT(*)` counts the padded row too, so count a column from the right table instead.

The lab had exactly this kind of question (include suppliers whose items haven't sold yet). The course gave us a starting query written in the implicit, inner-join style, and the point was to see why it drops those suppliers. Our submitted file stopped at that starting query, so that question was left unfinished. The answer is the left-join pattern above.

So when is an inner join the *right* choice? When every row on the left is sure to have a match. For "total cost of each sale", every sale row points to an existing item (a non-null foreign key). Inner and left give the same result there, and inner says what you mean.

## The WHERE trap

This one is sneaky. Say you only want to count sales with quantity above 1, and you write:

```sql
... LEFT JOIN sale sa ON sa.item = i.id
WHERE sa.quantity > 1
```

Walk through Bolt's padded row. `sa.quantity` is `NULL`, and `NULL > 1` is not true, so `WHERE` throws the row away. Your left join has quietly turned into an inner join. The fix is to put conditions on the right-hand table into the `ON` clause:

```sql
... LEFT JOIN sale sa ON sa.item = i.id AND sa.quantity > 1
```

Now the condition decides which sales count as a match, and items with no match still survive with `NULL`s.

In our lab report, one query used a self-join of the parts table (`LEFT JOIN parts t ON t.name = 'card reader' WHERE f.weight > t.weight`) to avoid a subquery. It gave the right answer, but only because the `WHERE` turned it into an inner join, which is what that question needed anyway. Writing `JOIN` would have said that honestly.

Joins come back in other places too. Finding *which* row holds the max in each group is usually done by joining back to the aggregate; see [greatest-n-per-group](/notes/greatest-n-per-group/).

## Deleting with foreign keys

A related lesson from the same lab: you can't delete a row that other rows still point to through a foreign key. MySQL (with InnoDB, which enforces foreign keys) refuses with a foreign key constraint error. You have to delete the rows that point to it first, children before parents (sales, then items, then the supplier). Or you define the foreign key with `ON DELETE CASCADE`, if that's really what you want. The way I think about it: joins and foreign keys are two views of the same links between tables.

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

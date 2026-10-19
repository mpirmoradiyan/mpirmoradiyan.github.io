---
title: "Functional dependencies, attribute closure and BCNF: normalisation without the hand-waving"
slug: functional-dependencies-and-bcnf
date: 2020-11-23
topic: Programming & Data
tags: [university-lab, databases, normalization, functional-dependencies, bcnf, relational-model]
summary: "What a functional dependency is, how the attribute closure finds every candidate key, what BCNF demands, and what you can lose when you decompose a table to reach it."
sources:
  - title: "732A57 Database Technology (course lab)"
    author: "Linköping University"
    type: course
  - title: "Fundamentals of Database Systems, 7th ed., Chapters 14-15 (Functional dependencies and normalization)"
    author: "Ramez Elmasri, Shamkant B. Navathe"
    type: book


status: draft
---

Normalisation is usually taught as a list of rules ("no repeating groups, no partial dependencies…"), which never stuck for me. What did stick, in the database course I took (this assignment was pair work), was the mechanical version: write down the functional dependencies, compute closures, and the keys and violations fall out. I only know databases at a learner's level, so this note is about the core ideas, not about designing big production schemas.

## Functional dependencies

A functional dependency (FD) $X \to Y$ on a table says: any two rows that agree on the attributes $X$ must also agree on $Y$. "Knowing $X$ determines $Y$." For example, in a table of course enrolments, `course_code → course_name` means one course code can't have two names.

FDs come from the meaning of the data, not from looking at the current rows. A table that happens to have unique names today doesn't make `name → id` a rule.

New FDs follow from known ones with **Armstrong's rules**:

- Reflexivity: if $Y \subseteq X$ then $X \to Y$.
- Augmentation: if $X \to Y$ then $XZ \to YZ$ (add the same attributes on both sides).
- Transitivity: if $X \to Y$ and $Y \to Z$ then $X \to Z$.

From these you get handy shortcuts like decomposition ($X \to YZ$ gives $X \to Y$) and union ($X \to Y$ and $X \to Z$ give $X \to YZ$).

## Attribute closure: the one algorithm you really need

The closure $X^+$ is the set of all attributes determined by $X$:

1. Start with $X^+ = X$.
2. For each FD $L \to R$: if $L \subseteq X^+$, add $R$ to $X^+$.
3. **Repeat step 2 until a full pass adds nothing.**

$X$ is a **superkey** if $X^+$ contains every attribute of the table, and a **candidate key** if it's a minimal superkey (no attribute can be removed and it's still a superkey).

Step 3 is where I slipped in my lab report. I computed a closure, added the attribute that one FD gave me, and stopped, without going back over the list. The newly added attribute made another FD's left side match, which would have pulled in everything else. So I missed one of the candidate keys, and listed a bigger, non-minimal set as a "candidate key" instead. The fix is boring but important: keep looping until nothing changes, and check minimality at the end.

## A worked example

The classic one: `Teaching(student, course, teacher)` with

- FD1: $\{\text{student}, \text{course}\} \to \{\text{teacher}\}$ (each student has one teacher per course),
- FD2: $\{\text{teacher}\} \to \{\text{course}\}$ (each teacher teaches only one course).

Closures:

- $\{\text{student}, \text{course}\}^+ = \{\text{student}, \text{course}, \text{teacher}\}$ by FD1. A key.
- $\{\text{student}, \text{teacher}\}^+$: FD2 adds course, giving $\{\text{student}, \text{teacher}, \text{course}\}$. Also a key!

So there are **two** candidate keys. The second one only shows up because adding `course` from FD2 lets you keep going. A quick hint for finding keys: an attribute that never appears on the right side of any FD (here `student`) must be in every key.

## BCNF

A table is in **Boyce–Codd normal form** if for every non-trivial FD $X \to Y$ that holds on it, $X$ is a superkey. In words: every determinant is a key. If something that isn't a key determines other attributes, those attributes get repeated in every row with the same $X$, and repeated data can go out of sync (update anomalies).

In `Teaching`, FD2 has `teacher` on the left, and $\{\text{teacher}\}^+ = \{\text{teacher}, \text{course}\}$ isn't everything. So FD2 violates BCNF: a teacher's course is stored once per student.

**Decomposition algorithm:** take a violating FD $X \to Y$ and split $R$ into

$$
R_1 = X \cup Y, \qquad R_2 = R \setminus (Y \setminus X).
$$

Then find the FDs and keys of each part and repeat until every part is in BCNF. Here: $R_1 = (\text{teacher}, \text{course})$ with key teacher, and $R_2 = (\text{student}, \text{teacher})$ with key {student, teacher}. Both are in BCNF.

This split is always **lossless**: joining $R_1$ and $R_2$ on $X$ gives back exactly the original rows, because $X$ is a key of $R_1$.

## The price: lost dependencies

What BCNF decomposition does *not* promise is **dependency preservation**. FD1 ({student, course} → teacher) involves attributes that now live in different tables. Neither table alone can enforce "a student has only one teacher per course"; you'd need a join, a trigger or application logic. The same thing happened in one of the lab exercises: after the BCNF split, one of the original FDs no longer fit inside any single table.

That's the trade-off behind the weaker **3NF**, which allows a violating FD if its right side is part of some candidate key (true for FD2 here, since `course` is in a key). 3NF can always be reached while preserving dependencies; BCNF sometimes can't.

## Key takeaways

- $X \to Y$: rows that agree on $X$ agree on $Y$. It's a rule about meaning, not about current data.
- Attribute closure finds superkeys; loop until nothing changes, then check minimality. Tables can have several candidate keys.
- BCNF: every non-trivial determinant is a superkey.
- Decompose on a violating FD into $X \cup Y$ and $R \setminus (Y \setminus X)$. Always lossless, not always dependency-preserving.
- 3NF is the compromise when you need every FD to stay enforceable in one table.

## Resources

- 732A57 Database Technology, Linköping University, assignment on functional dependencies and normalisation (2020).
- Ramez Elmasri and Shamkant B. Navathe, *Fundamentals of Database Systems*, 7th ed., Chapter 14 (FDs, normal forms) and Chapter 15 (closure, decomposition algorithms, lossless join, dependency preservation).

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

Normalisation is usually taught as a list of rules ("no repeating groups, no partial dependencies…"). That list never stuck for me. What did stick, in the database course I took (this assignment was pair work), was the mechanical version. Write down the functional dependencies, compute closures, and the keys and the problems just fall out. I only know databases at a learner's level, so this note is about the core ideas, not about designing big production schemas.

## Functional dependencies

A functional dependency (FD) $X \to Y$ on a table says: if two rows agree on the attributes $X$, they must also agree on $Y$. In plain words, "knowing $X$ tells you $Y$". For example, in a table of course enrolments, `course_code → course_name` means one course code can't have two names.

One thing to keep straight: FDs come from what the data *means*, not from looking at the rows you have today. If every name in the table happens to be unique right now, that doesn't make `name → id` a rule.

You can get new FDs from known ones with **Armstrong's rules**:

- Reflexivity: if $Y \subseteq X$ then $X \to Y$.
- Augmentation: if $X \to Y$ then $XZ \to YZ$ (add the same attributes on both sides).
- Transitivity: if $X \to Y$ and $Y \to Z$ then $X \to Z$.

From these come handy shortcuts like decomposition ($X \to YZ$ gives $X \to Y$) and union ($X \to Y$ and $X \to Z$ give $X \to YZ$).

## Attribute closure: the one algorithm you really need

The closure $X^+$ is the set of all attributes that $X$ determines. You find it like this:

1. Start with $X^+ = X$.
2. For each FD $L \to R$: if $L \subseteq X^+$, add $R$ to $X^+$.
3. **Repeat step 2 until a full pass adds nothing.**

$X$ is a **superkey** if $X^+$ contains every attribute of the table. It's a **candidate key** if it's a minimal superkey, meaning you can't remove any attribute and still have a superkey.

Step 3 is exactly where I slipped in my lab report. I computed a closure, added the attribute one FD gave me, and stopped. I didn't go back over the list. But the attribute I'd just added made another FD's left side match, and that one would have pulled in everything else. So I missed one of the candidate keys, and listed a bigger, non-minimal set as a "candidate key" instead. The fix is boring but important: keep looping until nothing changes, and check minimality at the end.

## A worked example

The classic one is `Teaching(student, course, teacher)` with

- FD1: $\{\text{student}, \text{course}\} \to \{\text{teacher}\}$ (each student has one teacher per course),
- FD2: $\{\text{teacher}\} \to \{\text{course}\}$ (each teacher teaches only one course).

Let's compute closures:

- $\{\text{student}, \text{course}\}^+ = \{\text{student}, \text{course}, \text{teacher}\}$ by FD1. That's a key.
- $\{\text{student}, \text{teacher}\}^+$: FD2 adds course, giving $\{\text{student}, \text{teacher}, \text{course}\}$. Also a key!

So there are **two** candidate keys. The second one only shows up because adding `course` from FD2 lets you keep going, the same kind of step I skipped in my report. A quick hint for finding keys: an attribute that never appears on the right side of any FD (here `student`) has to be in every key.

## BCNF

A table is in **Boyce–Codd normal form** if for every non-trivial FD $X \to Y$ that holds on it, $X$ is a superkey. In words: everything that determines something is a key. Why care? If something that isn't a key determines other attributes, those attributes get repeated in every row with the same $X$. Repeated data can drift out of sync when someone updates one copy and not the others (an update anomaly).

In `Teaching`, FD2 has `teacher` on the left, and $\{\text{teacher}\}^+ = \{\text{teacher}, \text{course}\}$ isn't everything. So FD2 breaks BCNF: a teacher's course is stored again for every student.

**Decomposition algorithm:** take an FD $X \to Y$ that breaks BCNF and split $R$ into

$$
R_1 = X \cup Y, \qquad R_2 = R \setminus (Y \setminus X).
$$

Then find the FDs and keys of each part, and repeat until every part is in BCNF. Here: $R_1 = (\text{teacher}, \text{course})$ with key teacher, and $R_2 = (\text{student}, \text{teacher})$ with key {student, teacher}. Both are in BCNF.

This split is always **lossless**: joining $R_1$ and $R_2$ on $X$ gives back exactly the original rows, because $X$ is a key of $R_1$. (It's an ordinary inner join; every teacher in $R_2$ finds exactly one row in $R_1$. More on join types in [inner vs left joins](/notes/sql-inner-vs-outer-joins/).)

## The price: lost dependencies

What BCNF decomposition does *not* promise is **dependency preservation**. FD1 ({student, course} → teacher) uses attributes that now live in different tables. Neither table alone can enforce "a student has only one teacher per course". You'd need a join, a trigger or application logic. The same thing happened in one of the lab exercises: after the BCNF split, one of the original FDs no longer fit inside any single table.

That trade-off is the reason for the weaker **3NF**. It allows an FD that breaks BCNF if its right side is part of some candidate key (true for FD2 here, since `course` is in a key). You can always reach 3NF while keeping every dependency enforceable; with BCNF you sometimes can't.

## Key takeaways

- $X \to Y$: rows that agree on $X$ agree on $Y$. It's a rule about meaning, not about current data.
- Attribute closure finds superkeys; loop until nothing changes, then check minimality. Tables can have several candidate keys.
- BCNF: every non-trivial determinant is a superkey.
- Decompose on a violating FD into $X \cup Y$ and $R \setminus (Y \setminus X)$. Always lossless, not always dependency-preserving.
- 3NF is the compromise when you need every FD to stay enforceable in one table.

## Resources

- 732A57 Database Technology, Linköping University, assignment on functional dependencies and normalisation (2020).
- Ramez Elmasri and Shamkant B. Navathe, *Fundamentals of Database Systems*, 7th ed., Chapter 14 (FDs, normal forms) and Chapter 15 (closure, decomposition algorithms, lossless join, dependency preservation).

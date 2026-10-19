---
title: "Check-then-act races: why a transaction alone doesn't stop overbooking"
slug: check-then-act-race-conditions-in-sql
date: 2020-12-22
topic: Programming & Data
tags: [university-lab, databases, transactions, concurrency, locking, mysql]
summary: "How two concurrent bookings can both pass a 'seats available?' check and overbook a flight, what isolation and locks actually guarantee, and three ways to make the check and the write one atomic step."
sources:
  - title: "732A57 Database Technology (course lab)"
    author: "Linköping University"
    type: course
  - title: "Fundamentals of Database Systems, 7th ed., Chapters 20-21 (Transaction processing and concurrency control)"
    author: "Ramez Elmasri, Shamkant B. Navathe"
    type: book
  - title: "MySQL 8.0 Reference Manual: InnoDB Locking Reads"
    author: "Oracle"
    url: "https://dev.mysql.com/doc/refman/8.0/en/innodb-locking-reads.html"
    type: docs
  - title: "MySQL 8.0 Reference Manual: Transaction Isolation Levels"
    author: "Oracle"
    url: "https://dev.mysql.com/doc/refman/8.0/en/innodb-transaction-isolation-levels.html"
    type: docs


status: draft
---

The last project in my database course was a small booking database for a made-up airline, written in MySQL with stored procedures (pair work). The schema wasn't the part that stuck with me. The final questions were: what happens if two people book at the same time? The bug behind them shows up everywhere, not just in databases. It's called **check, then act**: you check a condition, and by the time you act on it, someone else has changed it. I only know this at a learner's level, so this is the core idea plus the standard fixes.

## The race

A payment procedure does roughly this:

1. Read how many seats are still free on the flight.
2. If there are enough, insert the booking.

Now run it in two sessions at once. There are 3 seats left, and each session wants 3:

| time | session A | session B |
|---|---|---|
| t1 | reads: 3 free | |
| t2 | | reads: 3 free |
| t3 | inserts booking (3 seats) | |
| t4 | | inserts booking (3 seats) |

Both checks passed. Both inserts happened. The flight now has $-3$ free seats. Each step was fine on its own. The problem is the gap between step 1 and step 2. And like a join that quietly drops rows (see [inner vs left joins](/notes/sql-inner-vs-outer-joins/)), it gives no error at all, just wrong data.

When we ran the two booking scripts at the same time in the lab, nothing went wrong. The first one finished before the second one even checked, so the gap was too short to hit. The question then asked: can you make it happen? Our answer was to put a `SELECT SLEEP(5)` between the check and the insert, to make the gap wider. I like that trick. It shows the race was always possible, just unlikely.

## What a transaction does and doesn't give you

My first thought was: just wrap both steps in `START TRANSACTION ... COMMIT`. That doesn't fix it on its own, which surprised me. Three things matter here.

**Isolation.** Other sessions don't see your changes until you commit (at MySQL's default level, anyway). In the lab, a reservation made inside an open transaction in one terminal was invisible in another terminal until it was committed. That part we understood.

**Plain reads don't lock.** In InnoDB (MySQL's default engine), at the default `REPEATABLE READ` level, a normal `SELECT` is a *consistent read*. It reads from a snapshot and takes no locks. So both sessions can read "3 free" at the same moment, each inside its own transaction, and nothing stops either of them.

**Writes do lock.** When a session changes a row, InnoDB puts an exclusive lock on it until commit or rollback. A second session that tries to write the same row **waits**. In our answer I wrote that the second session got an "error" and that this was "serial scheduling". What really happens is that the second session blocks until the first one commits or rolls back. You only get an error if it waits longer than `innodb_lock_wait_timeout` (50 seconds by default), or if the database finds a deadlock. Also, locks only line up the *conflicting* operations, not the whole transactions.

## Fix 1: lock the row you check (`SELECT ... FOR UPDATE`)

A *locking read* puts an exclusive lock on the rows it reads. So a second booking for the same flight has to wait right at the check:

```sql
START TRANSACTION;
SELECT free_seats INTO @free FROM flight WHERE flight_id = 42 FOR UPDATE;  -- B waits here
-- (inside a procedure) IF @free >= 3 THEN
INSERT INTO booking(flight_id, seats) VALUES (42, 3);
UPDATE flight SET free_seats = free_seats - 3 WHERE flight_id = 42;
COMMIT;                                                                     -- B continues, sees 0
```

When B finally gets the lock, it sees 0. That's because a locking read reads the latest committed value, not the old snapshot a plain `SELECT` would use.

This needs one row that every booking for that flight touches (here, a `free_seats` counter on the flight). If free seats are computed by counting bookings, there's no single row to lock. Then you'd lock the flight row anyway, just as a "mutex".

## Fix 2: make the check part of the write

Let one statement do the check and the act together, then look at whether it did anything:

```sql
UPDATE flight
SET free_seats = free_seats - 3
WHERE flight_id = 42 AND free_seats >= 3;
-- ROW_COUNT() = 1: seats taken, go ahead and insert the booking
-- ROW_COUNT() = 0: not enough seats, roll back
```

The `UPDATE` locks the row while it checks the condition, so two of these can't both win the last 3 seats. A `CHECK (free_seats >= 0)` constraint is a good safety net on top. (MySQL only enforces `CHECK` from 8.0.16; older versions accept the syntax and ignore it.)

## Fix 3: lock whole tables

Our lab answer was `LOCK TABLES ... WRITE` before the booking and `UNLOCK TABLES` after. That works: only one session at a time can use the locked tables. (I wrote that "the table will be used by other tables each at a time"; I meant other *sessions*.) But it's the most heavy-handed option. It blocks bookings on every flight, not just the one being booked. In MySQL you also have to lock every table the procedure touches (including tables read inside functions it calls), and `LOCK TABLES` quietly commits any open transaction. Row-level locking (fix 1 or 2) is what you'd normally reach for.

## Key takeaways

- Check-then-act with a gap in between is a race, even inside a transaction.
- Isolation hides uncommitted changes; it doesn't stop two sessions reading the same old value.
- In InnoDB, plain `SELECT`s don't lock; writes take row locks, and a conflicting writer waits rather than failing (until a timeout or deadlock).
- Fixes: `SELECT ... FOR UPDATE` on the row you check, or a conditional `UPDATE ... WHERE enough` and check the affected row count. Table locks work but block everything.
- Inject a delay (like `SLEEP`) between check and act to make a hidden race show up in testing.

## Resources

- 732A57 Database Technology, Linköping University, lab 4 project (2020).
- Ramez Elmasri and Shamkant B. Navathe, *Fundamentals of Database Systems*, 7th ed., Chapter 20 (transactions, schedules, serializability) and Chapter 21 (two-phase locking).
- MySQL 8.0 Reference Manual, [Locking Reads](https://dev.mysql.com/doc/refman/8.0/en/innodb-locking-reads.html) and [Transaction Isolation Levels](https://dev.mysql.com/doc/refman/8.0/en/innodb-transaction-isolation-levels.html).

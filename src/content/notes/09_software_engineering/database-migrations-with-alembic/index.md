---
title: "Database migrations with Alembic: version control for your schema"
slug: database-migrations-with-alembic
date: 2024-04-26
topic: Software Engineering
tags: [databases, migrations, alembic, sqlalchemy, sql, schema]
summary: "Why table changes should be written as small, ordered, reversible migration scripts, how Alembic chains and applies them, what autogenerate can and cannot see, and how to add a NOT NULL column to a table that already has rows."
sources:
  - title: "Alembic documentation: Tutorial, Auto Generating Migrations, Batch Migrations for SQLite"
    author: "Mike Bayer"
    url: "https://alembic.sqlalchemy.org/en/latest/"
    type: docs
  - title: "Python API Development – Comprehensive Course for Beginners"
    author: "Sanjeev Thiyagarajan (freeCodeCamp)"
    url: "https://www.youtube.com/watch?v=0sOvCWFmrtA"
    type: course
  - title: "SQLAlchemy 2.0 documentation: ORM Declarative Mapping"
    url: "https://docs.sqlalchemy.org/en/20/orm/declarative_mapping.html"
    type: docs


status: draft
---

The first version of an app often creates its tables with one call such as `Base.metadata.create_all(engine)`. That works until the second version. `create_all` only creates tables that are missing; it never adds a column to an existing table, never changes a constraint, and never removes anything. The database on your laptop, the test server and production then drift apart. **Migrations** fix this by treating the schema like code: every change is a small script, the scripts are ordered and committed to git, and each database records which ones it has already run. Alembic is the migration tool for SQLAlchemy; Django migrations, Flyway and Liquibase follow the same idea.

## A chain of revisions

Each migration is a Python file with a unique **revision** ID, a pointer to its parent (**down_revision**), and two functions:

```python
revision = "0f8542fa1ff0"
down_revision = "b9bd290a46fb"      # the migration this one builds on

def upgrade() -> None:
    op.add_column("posts", sa.Column("owner_id", sa.Integer, nullable=False))
    op.create_foreign_key("posts_owner_id_fkey", "posts", "users",
                          ["owner_id"], ["id"], ondelete="CASCADE")

def downgrade() -> None:
    op.drop_constraint("posts_owner_id_fkey", "posts")
    op.drop_column("posts", "owner_id")
```

The `down_revision` links form a linked list from the first migration (`down_revision = None`) to the newest one, the **head**. In the course project the chain was: create posts → create users → add owner_id to posts → add phone_number to users → create votes → add content to posts. Alembic stores the current revision in a one-row table, `alembic_version`, inside the database itself. That is how it knows what still has to run.

`downgrade()` must undo `upgrade()` exactly, in reverse order (drop the foreign key before the column it uses). Write it even if you rarely use it: it is what lets you back out of a bad deploy.

## The everyday commands

```bash
alembic init migrations                          # once: creates alembic.ini and migrations/
alembic revision -m "add content to posts"       # new empty script, you fill in upgrade/downgrade
alembic revision --autogenerate -m "add votes"   # script drafted by comparing models to the DB
alembic upgrade head                             # run every pending migration
alembic downgrade -1                             # undo the latest one
alembic current                                  # which revision is this database at?
alembic history                                  # the whole chain
alembic upgrade head --sql                       # print the SQL instead of running it
```

The last one is useful when a DBA must review the SQL before it touches production.

## Wiring Alembic to your models and settings

Two lines in `migrations/env.py` matter. First, give Alembic your models' metadata so autogenerate can compare them with the database. Second, read the connection URL from the environment:

```python
import os
from app.models.base import Base

target_metadata = Base.metadata
config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
```

The generated `alembic.ini` has a `sqlalchemy.url = ...` line. Leave it empty. `alembic.ini` is committed to git, so a URL with a username and password there ends up in your history forever. The same settings object (or environment variables) that the app uses should feed Alembic.

## Autogenerate: a helpful draft, not the truth

`--autogenerate` compares `target_metadata` with the live database and writes the operations it thinks are needed. In my test project it correctly produced the whole `votes` table, including its composite primary key `(user_id, post_id)` and both `ON DELETE CASCADE` foreign keys. It is good at added and removed tables and columns, nullability, and most indexes and constraints.

It cannot read your intent, though. I renamed a column `phone_number` to `phone` in the model and asked for a migration:

```
Detected added column 'users.phone'
Detected removed column 'users.phone_number'
```

Run as generated, that drops the column and all its data, then adds an empty one. A rename must be written by hand with `op.alter_column("users", "phone_number", new_column_name="phone")`. Autogenerate also misses some type changes and server defaults depending on the backend, and knows nothing about data. **Always read the generated script before running it.**

## Pitfall: a NOT NULL column on a table that has rows

Adding a required column is the most common migration that works on an empty dev database and fails in production. With two existing users, autogenerate wrote `op.add_column('users', sa.Column('phone_number', sa.String(), nullable=False))`, and the upgrade failed:

```
sqlite3.OperationalError: Cannot add a NOT NULL column with default value NULL
```

PostgreSQL refuses for the same reason: the existing rows would break the constraint the moment the column appears. There are two clean fixes. If one constant is a sensible value, give the column a `server_default`. Otherwise use three steps in one migration:

```python
def upgrade() -> None:
    # 1) add the column as nullable so existing rows are allowed
    op.add_column("users", sa.Column("phone_number", sa.String(), nullable=True))
    # 2) backfill the existing rows
    op.execute("UPDATE users SET phone_number = 'unknown' WHERE phone_number IS NULL")
    # 3) tighten the constraint (batch mode lets SQLite rebuild the table)
    with op.batch_alter_table("users") as batch:
        batch.alter_column("phone_number", existing_type=sa.String(), nullable=False)

def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_column("phone_number")
```

After `alembic upgrade head` both old rows have `'unknown'` and the column is `NOT NULL`; `alembic downgrade -1` removes it again. `batch_alter_table` is needed only on SQLite, whose `ALTER TABLE` cannot change constraints; on PostgreSQL a plain `op.alter_column` works.

## Pitfall: the model and the migrations disagree

In my course project the model said `phone_number` was `nullable=False`, but the hand-written migration had made the column `nullable=True` to get past the error above, and the backfill-and-tighten steps were never added. Nothing failed: the app ran, and the database silently accepted users without a phone number, while the code assumed there would always be one. This is **schema drift**. Alembic can detect it: `alembic check` compares the models with an up-to-date database and fails if a migration is still needed. In my demo it reported:

```
New upgrade operations detected: [('modify_nullable', ..., 'users', 'phone_number', ..., True, False)]
```

Running `alembic check` in CI after `alembic upgrade head` on a fresh database catches drift before it reaches production.

## Habits that keep migrations safe

- One logical change per migration, with a descriptive message.
- Never edit a migration that has already run somewhere else; write a new one instead.
- Separate schema changes from large data changes when the table is big, so locks stay short.
- Test both directions: `upgrade head`, `downgrade base`, `upgrade head` on a scratch database.
- After switching to migrations, remove `create_all` from the app so there is only one source of schema changes.

These pair naturally with the API side covered in [building a REST API](/notes/rest-api-with-fastapi/): the ORM models define what the code expects, and the migrations make the database match.

## Key takeaways

- `create_all` creates missing tables but never alters existing ones; real projects need migrations.
- Alembic migrations form a chain through `revision` / `down_revision`; the database records its position in `alembic_version`.
- Autogenerate drafts a migration from model metadata, but it sees a rename as drop plus add and knows nothing about data: always review it.
- Adding a NOT NULL column to a populated table needs a `server_default` or the add-nullable, backfill, tighten sequence.
- Keep credentials out of `alembic.ini` and use `alembic check` to catch drift between models and migrations.

## Resources

- Alembic documentation: *Tutorial*, *Auto Generating Migrations* (including "what autogenerate does not detect"), *Running "Batch" Migrations for SQLite*: https://alembic.sqlalchemy.org/en/latest/
- Sanjeev Thiyagarajan, *Python API Development – Comprehensive Course for Beginners*, freeCodeCamp (Alembic section): https://www.youtube.com/watch?v=0sOvCWFmrtA
- SQLAlchemy 2.0 documentation, *ORM Declarative Mapping*: https://docs.sqlalchemy.org/en/20/orm/declarative_mapping.html

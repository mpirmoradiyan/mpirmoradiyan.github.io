---
title: "Database migrations with Alembic: version control for your schema"
slug: database-migrations-with-alembic
date: 2024-04-26
topic: Algorithms
tags: [databases, migrations, alembic, sqlalchemy, sql, schema]
summary: "Why table changes should be small, ordered, reversible migration scripts, how Alembic chains and runs them, what autogenerate can and can't see, and how to add a NOT NULL column to a table that already has rows."
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

When you first build an app, you often create the tables with one call like `Base.metadata.create_all(engine)`. That's fine until you need to change something. `create_all` only creates tables that don't exist yet. It never adds a column to an existing table, never changes a constraint, and never removes anything. So the database on your laptop, a test server and a real one slowly drift apart. **Migrations** fix this by treating the schema (the shape of your tables) like code: every change is a small script, the scripts are ordered and committed to git, and each database remembers which ones it has already run. Alembic is the migration tool for SQLAlchemy. I learned it while following a beginner course on building a Python API. Django migrations, Flyway and Liquibase work on the same idea.

## A chain of revisions

Each migration is a Python file with a unique **revision** ID, a pointer to the migration before it (**down_revision**), and two functions:

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

The way I think about it: the `down_revision` links make a linked list, from the first migration (`down_revision = None`) to the newest one, called the **head**. In the course project the chain went: create posts → create users → add owner_id to posts → add phone_number to users → create votes → add content to posts. Alembic keeps the current revision in a one-row table, `alembic_version`, inside the database itself. That's how it knows what still needs to run.

`downgrade()` has to undo `upgrade()` exactly, in reverse order (drop the foreign key before the column it uses). It's worth writing even if you rarely use it, because it's how you back out of a bad change.

## The commands you use all the time

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

The last one is handy when someone needs to read the SQL before it runs on an important database.

## Connecting Alembic to your models and settings

Two lines in `migrations/env.py` matter. First, hand Alembic your models' metadata so autogenerate can compare them with the database. Second, read the connection URL from the environment:

```python
import os
from app.models.base import Base

target_metadata = Base.metadata
config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
```

The generated `alembic.ini` has a `sqlalchemy.url = ...` line. Leave it empty. `alembic.ini` gets committed to git, so a URL with a username and password in it stays in your history forever. Alembic should get its settings from the same place the app does (a settings object or environment variables).

## Autogenerate is a draft, not the truth

`--autogenerate` compares `target_metadata` with the live database and writes the operations it thinks are needed. In my test project it got the whole `votes` table right, including its composite primary key `(user_id, post_id)` and both `ON DELETE CASCADE` foreign keys. It's good at spotting added and removed tables and columns, nullability, and most indexes and constraints.

But it can't read your mind. I renamed a column `phone_number` to `phone` in the model and asked for a migration:

```
Detected added column 'users.phone'
Detected removed column 'users.phone_number'
```

If you run that as is, it drops the column with all its data and then adds an empty one. A rename has to be written by hand with `op.alter_column("users", "phone_number", new_column_name="phone")`. Depending on the database, autogenerate can also miss some type changes and server defaults, and it knows nothing about the data itself. **Always read the generated script before you run it.**

## Trap: a NOT NULL column on a table that already has rows

Adding a required column is the classic migration that works fine on an empty dev database and then fails on a real one. With two users already in the table, autogenerate wrote `op.add_column('users', sa.Column('phone_number', sa.String(), nullable=False))`, and the upgrade failed:

```
sqlite3.OperationalError: Cannot add a NOT NULL column with default value NULL
```

It makes sense once you think about it: the existing rows would break the NOT NULL rule the moment the column appears. PostgreSQL refuses for the same reason. There are two clean fixes. If one fixed value makes sense for everyone, give the column a `server_default`. Otherwise do it in three steps inside one migration:

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

After `alembic upgrade head` both old rows have `'unknown'` and the column is `NOT NULL`. `alembic downgrade -1` removes it again. You only need `batch_alter_table` on SQLite, because its `ALTER TABLE` can't change constraints. On PostgreSQL a plain `op.alter_column` works.

## Trap: the model and the migrations don't agree

In my course project the model said `phone_number` was `nullable=False`. But the hand-written migration had made the column `nullable=True` to get past the error above, and I never added the backfill-and-tighten steps. Nothing broke. The app ran, and the database quietly accepted users with no phone number, while the code assumed there would always be one. This mismatch is called **schema drift**. Alembic can catch it: `alembic check` compares the models with an up-to-date database and fails if a migration is still needed. In my demo it said:

```
New upgrade operations detected: [('modify_nullable', ..., 'users', 'phone_number', ..., True, False)]
```

If you run `alembic check` in CI right after `alembic upgrade head` on a fresh database, drift gets caught early instead of slipping through.

## Habits that keep migrations safe

- One logical change per migration, with a clear message.
- Never edit a migration that has already run somewhere else. Write a new one instead.
- On big tables, keep schema changes and large data changes in separate steps, so tables aren't locked for long.
- Test both directions on a scratch database: `upgrade head`, `downgrade base`, `upgrade head`.
- Once you switch to migrations, remove `create_all` from the app, so there's only one place schema changes come from.

This goes hand in hand with the API side in [building a REST API](/notes/rest-api-with-fastapi/): the ORM models say what the code expects, and the migrations make the database match.

## Key takeaways

- `create_all` creates missing tables but never changes existing ones, so real projects need migrations.
- Alembic migrations form a chain through `revision` / `down_revision`, and the database records where it is in `alembic_version`.
- Autogenerate drafts a migration from the model metadata, but it sees a rename as drop plus add and knows nothing about data. Always review it.
- Adding a NOT NULL column to a table with rows needs a `server_default`, or the add-nullable, backfill, tighten steps.
- Keep passwords out of `alembic.ini`, and use `alembic check` to catch drift between models and migrations.

## Resources

- Alembic documentation: *Tutorial*, *Auto Generating Migrations* (including "what autogenerate does not detect"), *Running "Batch" Migrations for SQLite*: https://alembic.sqlalchemy.org/en/latest/
- Sanjeev Thiyagarajan, *Python API Development – Comprehensive Course for Beginners*, freeCodeCamp (Alembic section): https://www.youtube.com/watch?v=0sOvCWFmrtA
- SQLAlchemy 2.0 documentation, *ORM Declarative Mapping*: https://docs.sqlalchemy.org/en/20/orm/declarative_mapping.html

---
title: "Building a REST API with FastAPI: routes, schemas and dependency injection"
slug: rest-api-with-fastapi
date: 2024-04-26
topic: Software Engineering
tags: [rest, api, fastapi, pydantic, dependency-injection, http, python]
summary: "How a small REST API is organised: resources and HTTP verbs, path and query parameters, separate input and output schemas, a per-request database session injected as a dependency, and honest status codes."
sources:
  - title: "FastAPI documentation: Tutorial – User Guide"
    author: "Sebastián Ramírez"
    url: "https://fastapi.tiangolo.com/tutorial/"
    type: docs
  - title: "Python API Development – Comprehensive Course for Beginners"
    author: "Sanjeev Thiyagarajan (freeCodeCamp)"
    url: "https://www.youtube.com/watch?v=0sOvCWFmrtA"
    type: course
  - title: "RFC 9110: HTTP Semantics (methods and status codes)"
    url: "https://www.rfc-editor.org/rfc/rfc9110"
    type: docs
  - title: "Pydantic documentation: Models"
    url: "https://docs.pydantic.dev/latest/concepts/models/"
    type: docs


status: draft
---

Most machine learning work ends up behind an HTTP endpoint: a model server, a labelling tool, a small internal service. A REST API is the usual shape for that. This note walks through the parts every such API needs, using FastAPI because it makes each part explicit: routes, validated request and response schemas, shared resources injected per request, and correct status codes. The ideas carry over to Flask, Django REST Framework or any other framework.

## Resources and verbs

REST treats the server as a set of **resources** addressed by URLs, and uses the HTTP method to say what to do with them. For a blog-like app with posts:

| Method | URL | Meaning | Success code |
|---|---|---|---|
| `GET` | `/posts/` | list posts | 200 OK |
| `GET` | `/posts/{id}` | read one post | 200 OK |
| `POST` | `/posts/` | create a post | 201 Created |
| `PUT` | `/posts/{id}` | replace a post | 200 OK |
| `DELETE` | `/posts/{id}` | delete a post | 204 No Content |

`GET`, `PUT` and `DELETE` are **idempotent**: sending the same request twice leaves the server in the same state as sending it once. `POST` is not, which is why retrying a failed `POST` can create duplicates.

## Path and query parameters

Data reaches an endpoint in three places. The **path** identifies the resource (`/posts/7`). The **query string** tunes a request (`/posts/?limit=10&skip=20&search=cat`). The **body** carries the content of a create or update. In FastAPI each one is just a typed function argument: a name that appears in the route template is a path parameter, a simple type that does not is a query parameter, and a Pydantic model is the body. The type hints do double work: they convert the incoming strings and they reject bad input with a `422 Unprocessable Entity` before your code runs.

```python
from typing import Annotated
from fastapi import APIRouter, Query

router = APIRouter(prefix="/posts", tags=["Posts"])

@router.get("/", response_model=list[PostOut])
def list_posts(db: DB, limit: Annotated[int, Query(ge=1, le=100)] = 10,
               skip: int = 0, search: str = ""):
    posts = [p for p in db["posts"].values() if search in p["title"]]
    return posts[skip: skip + limit]          # simple pagination
```

`GET /posts/abc` returns 422 because `abc` is not an integer, and `?limit=0` returns 422 because of the `ge=1` bound. You get this validation for free.

## Separate schemas for input and output

A common beginner mistake is to use one model for everything. The client should send only what it is allowed to set, and the server should return only what the client is allowed to see. So define (at least) two schemas per resource:

```python
from pydantic import BaseModel

class PostIn(BaseModel):          # what the client sends
    title: str
    content: str
    published: bool = True

class PostOut(PostIn):            # what the server returns
    id: int                       # set by the database
    owner_id: int                 # set from the logged-in user

class UserIn(BaseModel):
    email: str
    password: str

class UserOut(BaseModel):
    id: int
    email: str                    # no password field at all
```

The `response_model` of a route filters the return value through the output schema. Even if the handler returns the full database row, including the password hash, the client only sees `id` and `email`. Making leaks impossible by construction is far safer than remembering to delete fields by hand.

When the handler returns ORM objects instead of dicts, Pydantic must be told to read attributes: in Pydantic v2 this is `model_config = ConfigDict(from_attributes=True)` (the v1 name was `orm_mode = True`).

## Dependency injection: one database session per request

Many endpoints need the same things: a database session, the current user, settings. Instead of creating them inside every handler, declare them as **dependencies**. FastAPI calls the dependency, passes its result into the handler, and, if the dependency is a generator, runs the code after `yield` once the response is done:

```python
from fastapi import Depends

def get_db():
    db = SessionLocal()       # open a session for this request
    try:
        yield db              # the endpoint runs here
    finally:
        db.close()            # always runs, even if the endpoint raised

DB = Annotated[Session, Depends(get_db)]
```

This is the same pattern as a context manager. Every request gets its own session, and it is always closed. Dependencies can depend on other dependencies: in the [authentication note](/notes/oauth2-password-flow-and-jwt/) a `get_current_user` dependency itself depends on the token and on `get_db`. In tests you can swap a dependency for a fake with `app.dependency_overrides[get_db] = fake_db`, which is the main practical win of the pattern.

## Status codes and errors

Return the code that matches what happened, and raise an exception for failures:

```python
from fastapi import HTTPException, Response, status

@router.get("/{post_id}", response_model=PostOut)
def get_post(post_id: int, db: DB):
    if post_id not in db["posts"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Post {post_id} not found")
    return db["posts"][post_id]

@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_post(post_id: int, db: DB):
    if db["posts"].pop(post_id, None) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Post {post_id} not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)   # empty body
```

Two mistakes from my first version are worth naming. I wrote `raise status.HTTP_404_NOT_FOUND`, but that constant is just the integer 404; raising an integer is a `TypeError`, so the client got a 500 instead of a 404. You must raise an exception object such as `HTTPException`. And I declared a delete route as 204 but returned a 200 response with a text body. A 204 response must have no body; either return an empty 204 or use 200 with a message, not a mix.

A useful rule of thumb for the 4xx family: **400** malformed request, **401** "I don't know who you are", **403** "I know who you are, and you may not do this", **404** no such resource, **409** conflict with current state (for example voting twice on the same post), **422** valid JSON that fails the schema.

## Routers keep the app organised

As the API grows, group routes by resource into `APIRouter` objects, each in its own module, and include them in the app:

```python
app = FastAPI()
app.include_router(posts_router)     # /posts/...
app.include_router(users_router)     # /users/...
app.include_router(auth_router)      # /login
```

A layout that worked well for me: `models/` for database tables, `schemas/` for Pydantic models, `routers/` for endpoints, plus `database.py` (engine and `get_db`) and `config.py` (settings read from environment variables, never hard-coded). FastAPI also generates interactive OpenAPI documentation at `/docs` from these type hints, which is the quickest way to try every endpoint by hand.

## Testing without a server

FastAPI's `TestClient` sends requests to the app in-process, so endpoints can be checked with ordinary asserts:

```python
from fastapi.testclient import TestClient

c = TestClient(app)
r = c.post("/posts/", json={"title": "Hello", "content": "First post"})
assert r.status_code == 201 and r.json()["id"] == 1
assert c.get("/posts/7").status_code == 404
assert c.post("/posts/", json={"title": "No content"}).status_code == 422
assert c.delete("/posts/1").status_code == 204
assert "password" not in c.post("/users/", json={"email": "a@b.c", "password": "x"}).json()
```

Once the tables live in a real database, their schema changes need to be versioned too; that is the subject of [database migrations](/notes/database-migrations-with-alembic/).

## Key takeaways

- Model the API as resources plus HTTP verbs; know which verbs are idempotent.
- Use typed path, query and body parameters so bad input is rejected with 422 before your code runs.
- Keep separate input and output schemas; let `response_model` make it impossible to leak fields like password hashes.
- Inject shared resources (database session, current user) as dependencies; a `yield` dependency guarantees cleanup and is easy to override in tests.
- Raise `HTTPException` for errors and return codes that mean what happened; a 204 has no body.

## Resources

- FastAPI documentation, *Tutorial – User Guide* (path and query parameters, request body, response model, dependencies, bigger applications): https://fastapi.tiangolo.com/tutorial/
- Sanjeev Thiyagarajan, *Python API Development – Comprehensive Course for Beginners*, freeCodeCamp: https://www.youtube.com/watch?v=0sOvCWFmrtA
- RFC 9110, *HTTP Semantics*, sections on methods and status codes: https://www.rfc-editor.org/rfc/rfc9110
- Pydantic documentation, *Models*: https://docs.pydantic.dev/latest/concepts/models/

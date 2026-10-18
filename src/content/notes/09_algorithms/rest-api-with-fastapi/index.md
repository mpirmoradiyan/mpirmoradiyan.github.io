---
title: "Building a REST API with FastAPI: routes, schemas and dependency injection"
slug: rest-api-with-fastapi
date: 2024-04-26
topic: Algorithms
tags: [rest, api, fastapi, pydantic, dependency-injection, http, python]
summary: "How a small REST API fits together: resources and HTTP verbs, path and query parameters, separate input and output schemas, a database session per request passed in as a dependency, and honest status codes."
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

A lot of machine learning code ends up behind an HTTP endpoint sooner or later: a model you can send requests to, a small labelling tool, that kind of thing. A REST API is the usual way to do that. I wanted to understand the basics, so I followed a beginner course and built a small blog-style practice API with FastAPI. FastAPI makes each piece easy to see: routes, checked request and response schemas, shared things passed into each request, and the right status codes. The ideas carry over to Flask, Django REST Framework or any other framework.

## Resources and verbs

REST treats the server as a set of **resources** (things like posts or users), each with its own URL. The HTTP method says what you want to do with it. For a blog-like app with posts:

| Method | URL | Meaning | Success code |
|---|---|---|---|
| `GET` | `/posts/` | list posts | 200 OK |
| `GET` | `/posts/{id}` | read one post | 200 OK |
| `POST` | `/posts/` | create a post | 201 Created |
| `PUT` | `/posts/{id}` | replace a post | 200 OK |
| `DELETE` | `/posts/{id}` | delete a post | 204 No Content |

`GET`, `PUT` and `DELETE` are **idempotent**, a fancy word that just means sending the same request twice leaves the server in the same state as sending it once. `POST` isn't, which is why retrying a failed `POST` can create duplicates.

## Path and query parameters

Data reaches an endpoint in three places. The **path** says which resource (`/posts/7`). The **query string** tweaks the request (`/posts/?limit=10&skip=20&search=cat`). The **body** carries the content for a create or update. In FastAPI each one is just a typed function argument. If the name appears in the route, it's a path parameter. If it's a simple type that doesn't appear in the route, it's a query parameter. If it's a Pydantic model, it's the body. The nice part is that the type hints do two jobs: they convert the incoming strings, and they reject bad input with a `422 Unprocessable Entity` before your code even runs.

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

`GET /posts/abc` returns 422 because `abc` isn't an integer, and `?limit=0` returns 422 because of the `ge=1` bound. You get this checking for free.

## Separate schemas for input and output

A common beginner mistake is using one model for everything. The client should only send what it's allowed to set, and the server should only send back what the client is allowed to see. So you define (at least) two schemas per resource:

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

A route's `response_model` filters whatever it returns through the output schema. So even if the handler returns the full database row, password hash and all, the client only sees `id` and `email`. I like this: it makes the leak impossible by design, which is much safer than remembering to delete fields by hand.

If the handler returns ORM objects instead of dicts, you have to tell Pydantic to read attributes. In Pydantic v2 that's `model_config = ConfigDict(from_attributes=True)` (in v1 it was called `orm_mode = True`).

## Dependency injection: one database session per request

"Dependency injection" sounded scary to me, but the idea is simple. Lots of endpoints need the same things: a database session, the current user, settings. Instead of creating them inside every handler, you declare them as **dependencies**. FastAPI calls the dependency, passes the result into the handler, and if the dependency is a generator, it runs the code after `yield` once the response is done:

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

So basically it works like a context manager (a `with` block). Every request gets its own session, and it always gets closed. Dependencies can depend on other dependencies: in the [authentication note](/notes/oauth2-password-flow-and-jwt/), a `get_current_user` dependency itself depends on the token and on `get_db`. In tests you can swap a dependency for a fake one with `app.dependency_overrides[get_db] = fake_db`, and that's the biggest practical win.

## Status codes and errors

Return the code that matches what happened, and raise an exception when something fails:

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

I made two mistakes in my first version that are worth writing down. First, I wrote `raise status.HTTP_404_NOT_FOUND`. But that constant is just the integer 404, and raising an integer is a `TypeError`, so the client got a 500 instead of a 404. You have to raise an exception object like `HTTPException`. Second, I declared a delete route as 204 but returned a 200 response with a text body. A 204 response must have no body. Either return an empty 204 or use 200 with a message, not a mix of both.

A rule of thumb for the 4xx codes: **400** the request is malformed, **401** "I don't know who you are", **403** "I know who you are, and you can't do this", **404** no such resource, **409** clashes with the current state (like voting twice on the same post), **422** valid JSON that doesn't match the schema.

## Routers keep things tidy

As the API grows, you group routes by resource into `APIRouter` objects, each in its own file, and plug them into the app:

```python
app = FastAPI()
app.include_router(posts_router)     # /posts/...
app.include_router(users_router)     # /users/...
app.include_router(auth_router)      # /login
```

The layout I used in the course project: `models/` for database tables, `schemas/` for Pydantic models, `routers/` for endpoints, plus `database.py` (the engine and `get_db`) and `config.py` (settings read from environment variables, never hard-coded). FastAPI also builds interactive OpenAPI docs at `/docs` from these type hints, which is the quickest way to try every endpoint by hand.

## Testing without running a server

FastAPI's `TestClient` sends requests to the app inside the same Python process, so you can check endpoints with plain asserts:

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

Once the tables live in a real database, changes to their structure need to be tracked too. That's what [database migrations](/notes/database-migrations-with-alembic/) are about.

## Key takeaways

- Think of the API as resources plus HTTP verbs, and know which verbs are idempotent.
- Use typed path, query and body parameters, so bad input gets rejected with 422 before your code runs.
- Keep separate input and output schemas, and let `response_model` make it impossible to leak fields like password hashes.
- Pass shared things (database session, current user) in as dependencies. A `yield` dependency always cleans up and is easy to swap out in tests.
- Raise `HTTPException` for errors and return codes that match what happened. A 204 has no body.

## Resources

- FastAPI documentation, *Tutorial – User Guide* (path and query parameters, request body, response model, dependencies, bigger applications): https://fastapi.tiangolo.com/tutorial/
- Sanjeev Thiyagarajan, *Python API Development – Comprehensive Course for Beginners*, freeCodeCamp: https://www.youtube.com/watch?v=0sOvCWFmrtA
- RFC 9110, *HTTP Semantics*, sections on methods and status codes: https://www.rfc-editor.org/rfc/rfc9110
- Pydantic documentation, *Models*: https://docs.pydantic.dev/latest/concepts/models/

---
title: "Logging users in: password hashing, the OAuth2 password flow and JWTs"
slug: oauth2-password-flow-and-jwt
date: 2024-04-26
topic: Software Engineering
tags: [authentication, oauth2, jwt, bcrypt, security, api, fastapi]
summary: "How an API stores passwords safely, swaps a username and password for a signed token, and checks that token on every request, with a JWT built by hand to show there is no magic."
sources:
  - title: "FastAPI documentation: Security – OAuth2 with Password (and hashing), Bearer with JWT tokens"
    url: "https://fastapi.tiangolo.com/tutorial/security/"
    type: docs
  - title: "Python API Development – Comprehensive Course for Beginners"
    author: "Sanjeev Thiyagarajan (freeCodeCamp)"
    url: "https://www.youtube.com/watch?v=0sOvCWFmrtA"
    type: course
  - title: "RFC 7519: JSON Web Token (JWT)"
    url: "https://www.rfc-editor.org/rfc/rfc7519"
    type: docs
  - title: "RFC 6749: The OAuth 2.0 Authorization Framework (section 4.3, Resource Owner Password Credentials)"
    url: "https://www.rfc-editor.org/rfc/rfc6749"
    type: docs
  - title: "OWASP Password Storage Cheat Sheet"
    url: "https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html"
    type: docs


status: draft
---

An API that lets users create and delete their own data must answer two questions on every request: who is calling, and are they allowed to do this? HTTP is stateless, so the usual answer is a **token**: the user proves their identity once with a password, receives a signed token, and sends it with every later request. This note covers the three pieces: storing passwords, issuing tokens, and checking them.

## Never store passwords, store slow salted hashes

If the database leaks, plain-text passwords are game over, and people reuse passwords across sites. So store a **hash** instead and compare hashes at login. Not any hash, though. A fast hash like SHA-256 lets an attacker try billions of guesses per second against a stolen table. Password hashes such as **bcrypt**, scrypt or Argon2 are built to be slow, and they add a random **salt** to each password so that two users with the same password get different hashes and precomputed tables are useless.

```python
import bcrypt

pw = b"correct horse battery staple"
h1 = bcrypt.hashpw(pw, bcrypt.gensalt())     # cost factor 12 by default
h2 = bcrypt.hashpw(pw, bcrypt.gensalt())
print(h1)   # b'$2b$12$3s5OaZuUIbMr...'  algorithm, cost, salt and hash in one string
print(h1 != h2)                                         # True: different salts
print(bcrypt.checkpw(pw, h1), bcrypt.checkpw(b"wrong", h1))   # True False
```

On my laptop one check takes about 0.3 s. That is nothing for a login and a disaster for a brute-force attack. The cost factor is stored inside the hash, so you can raise it later without breaking old hashes. (In Python, `passlib.CryptContext(schemes=["bcrypt"])` wraps the same thing and adds hash upgrading.)

## The OAuth2 password flow

OAuth2 defines several ways to get a token. The simplest, the **password grant**, fits a first-party API where the client is your own front end:

1. The client sends `username` and `password` as a form to a token endpoint, for example `POST /login`.
2. The server looks up the user, checks the password hash, and returns `{"access_token": "...", "token_type": "bearer"}`.
3. The client sends `Authorization: Bearer <token>` on every protected request.

The OAuth2 spec now discourages the password grant for third-party clients (they should never see the user's password; the authorization-code flow exists for that), but the token-on-every-request part is the same in all flows.

The token can be anything the server can check. My first version simply returned the **username as the token**. That works mechanically, but it gives no security at all: anyone who types `Bearer alice` becomes Alice. The token must be something a client cannot forge. One option is a random string stored in a sessions table. The other is a self-contained signed token: a JWT.

## What a JWT is

A JSON Web Token is three base64url-encoded parts joined by dots: `header.payload.signature`.

- **Header**: which algorithm signed it, e.g. `{"alg": "HS256", "typ": "JWT"}`.
- **Payload**: the **claims**, e.g. `{"user_id": 42, "exp": 1792322566}`. `exp` is the expiry time in Unix seconds.
- **Signature**: for HS256, $\mathrm{HMAC\text{-}SHA256}(K,\ \texttt{header} \,\|\, \texttt{"."} \,\|\, \texttt{payload})$, where $K$ is a secret key known only to the server.

Writing it with the standard library shows there is no magic:

```python
import base64, hashlib, hmac, json, time

def b64url(b): return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
def unb64url(s): return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

def sign(payload: dict, key: bytes) -> str:
    h = b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = b64url(json.dumps(payload).encode())
    sig = hmac.new(key, f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{b64url(sig)}"

def verify(token: str, key: bytes) -> dict:
    h, p, s = token.split(".")
    expected = hmac.new(key, f"{h}.{p}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, unb64url(s)):   # constant-time compare
        raise ValueError("bad signature")
    payload = json.loads(unb64url(p))
    if payload["exp"] < time.time():
        raise ValueError("token expired")
    return payload

key = b"long-random-secret-from-an-environment-variable"
token = sign({"user_id": 42, "exp": int(time.time()) + 30 * 60}, key)
print(verify(token, key))                     # {'user_id': 42, 'exp': ...}
print(json.loads(unb64url(token.split(".")[1])))   # readable without the key!
```

Two facts follow directly from the code:

- **Signed, not encrypted.** Anyone can decode the payload. Never put secrets in it; a user ID and an expiry are enough.
- **Tamper-evident.** Changing `user_id` to 1 changes the payload, so the old signature no longer matches and `verify` raises "bad signature". Without the key, nobody can produce a valid signature.

In real code use a maintained library (PyJWT or python-jose) and always pass the list of accepted algorithms when decoding, so a token cannot choose its own algorithm (the classic `alg: none` attack). The secret key, algorithm and expiry belong in environment variables, not in the source.

Because a JWT is self-contained, the server does not need a database lookup to check it, which scales well. The price is that you cannot easily revoke a token before it expires. Keep access tokens short-lived (minutes to an hour) and use refresh tokens or a deny-list if you need logout.

## Checking the token in a dependency

In FastAPI the check is a [dependency](/notes/rest-api-with-fastapi/) that every protected route asks for. `OAuth2PasswordBearer` reads the `Authorization` header (and returns 401 if it is missing); our function verifies the token:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

@app.post("/login")
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = USERS.get(form.username)
    if not user or not bcrypt.checkpw(form.password.encode(), user["hash"]):
        raise HTTPException(401, "Invalid credentials")   # same message for both cases
    token = sign({"user_id": user["id"], "exp": int(time.time()) + 1800}, key)
    return {"access_token": token, "token_type": "bearer"}

def get_current_user_id(token: Annotated[str, Depends(oauth2_scheme)]) -> int:
    try:
        return verify(token, key)["user_id"]
    except (ValueError, KeyError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Could not validate credentials",
                            headers={"WWW-Authenticate": "Bearer"})

@app.delete("/posts/{post_id}", status_code=204)
def delete_post(post_id: int, user_id: Annotated[int, Depends(get_current_user_id)]):
    if POSTS[post_id]["owner_id"] != user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your post")
```

Testing this with a `TestClient` gives: wrong password → 401, no token → 401, garbage token → 401, valid token for someone else's post → 403. The login error deliberately does not say whether the user exists, so it cannot be used to discover registered emails.

## 401 versus 403

These two are often mixed up:

- **401 Unauthorized** really means *unauthenticated*: no credentials, or credentials that do not check out. Send `WWW-Authenticate: Bearer` so the client knows how to authenticate.
- **403 Forbidden** means *authenticated but not allowed*: we know you are user 42, but this post belongs to user 7.

Authorization (the 403 check) usually happens in the route, because it depends on the resource; authentication (the 401 check) is shared and lives in the dependency.

## Key takeaways

- Store passwords only as slow, salted hashes (bcrypt, scrypt, Argon2), never plain or with a fast hash.
- The password flow exchanges a username and password for a token once; every later request carries `Authorization: Bearer <token>`.
- A JWT is `header.payload.signature`, base64url-encoded; HS256 signs it with an HMAC over a server secret. It is signed, not encrypted.
- Verify signature, algorithm and `exp` on every request, keep tokens short-lived, and keep the key out of the code.
- 401 means "who are you?", 403 means "not allowed".

## Resources

- FastAPI documentation, *Security*: *OAuth2 with Password (and hashing), Bearer with JWT tokens*: https://fastapi.tiangolo.com/tutorial/security/
- Sanjeev Thiyagarajan, *Python API Development – Comprehensive Course for Beginners*, freeCodeCamp (authentication and JWT sections): https://www.youtube.com/watch?v=0sOvCWFmrtA
- RFC 7519, *JSON Web Token (JWT)*: https://www.rfc-editor.org/rfc/rfc7519
- RFC 6749, *The OAuth 2.0 Authorization Framework*, section 4.3: https://www.rfc-editor.org/rfc/rfc6749
- OWASP, *Password Storage Cheat Sheet*: https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html

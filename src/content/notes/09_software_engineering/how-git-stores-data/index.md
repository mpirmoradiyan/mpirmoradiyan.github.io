---
title: "How Git stores your work: blobs, trees, commits and refs"
slug: how-git-stores-data
date: 2024-04-09
topic: Software Engineering
tags: [git, version-control, content-addressing, sha1, branching, merging]
summary: "Git is a small key-value store of four object types plus a few pointer files; once you see that, branches, merges, cherry-picks and why a deleted secret stays in history all make sense."
sources:
  - title: "Pro Git (2nd ed.), Chapter 10: Git Internals"
    author: "Scott Chacon, Ben Straub"
    url: "https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain"
    type: book
  - title: "Git documentation: gitglossary, git-cat-file, git-hash-object, git-filter-repo"
    url: "https://git-scm.com/docs"
    type: docs


status: draft
---

Most Git confusion comes from thinking of commits as "diffs" and branches as "folders". Neither is true. Underneath the commands, Git is a simple database. It stores every version of every file as an object named by the hash of its contents, links those objects into snapshots, and keeps a few small text files that point at the latest snapshot. I built a throw-away practice repository to try branches, merges, pull requests and cherry-picks. This note explains what each of those actions does to that database.

## Content addressing: an object's name is its hash

Git stores data as **objects** in `.git/objects/`. The name of each object is the SHA-1 hash of a short header plus its content. The header is the type, a space, the size in bytes, and a zero byte. You can reproduce Git's hash in a few lines of Python:

```python
import hashlib

data = b"Hello, World!\n"
store = b"blob %d\x00" % len(data) + data   # header + content
print(hashlib.sha1(store).hexdigest())      # 8ab686eafeb1f44702738c8b0f24f2567c36da6d
```

`git hash-object greeting.md` prints the same 40 hex characters for a file containing that line. On disk the object lives at `.git/objects/8a/b686ea…`: the first two characters name a subfolder, and the content is compressed with zlib. Decompressing the file with `zlib.decompress` gives back `b'blob 14\x00Hello, World!\n'`.

Three consequences follow from naming objects by their content:

- **Deduplication.** Two identical files, in any folder and any commit, are stored once.
- **Integrity.** If a single bit changes, the name no longer matches, so corruption is detected.
- **Immutability.** You cannot edit an object. "Changing" something always means creating a new object with a new name.

(Newer Git versions can use SHA-256 instead of SHA-1, but the idea is the same.)

## The four object types

1. **Blob**: the raw content of one file. No file name, no permissions, just bytes.
2. **Tree**: one directory listing. Each line holds a mode, a type, a hash and a name, for example `100644 blob 8ab686ea… greeting.md`. Subfolders are trees inside trees.
3. **Commit**: a pointer to one root tree (the whole project snapshot), plus zero or more parent commits, the author, the committer, timestamps and the message.
4. **Tag** (annotated): a named, optionally signed pointer to another object, usually a commit.

After `git add greeting.md`, the blob already exists in the object store, because staging writes it. `git commit` then adds a tree and a commit. `git cat-file -p` prints any object:

```text
$ git cat-file -p HEAD
tree 89b6e34971467c6d834d06e64754939a2c5dd043
author Demo <demo@example.com> 1711616400 +0100
committer Demo <demo@example.com> 1711616400 +0100

add greeting

$ git cat-file -p HEAD^{tree}
100644 blob 8ab686eafeb1f44702738c8b0f24f2567c36da6d	greeting.md
```

A commit therefore stores a **full snapshot**, not a diff. That sounds wasteful, but unchanged files point to blobs that already exist, so a new commit only adds objects for what actually changed. Diffs are computed on demand by comparing two trees. For storage efficiency, `git gc` later packs objects into a *packfile* and stores similar objects as deltas of each other. In my demo, ten loose objects became one pack. That compression is a storage detail and does not change the model.

## History is a chain of parent pointers

Each commit names its parent, and the parent's hash covers the parent's own parent, and so on back to the first commit. The second commit in the demo shows this:

```text
tree e6331b9d4678d8dfccdfe0160ddcedaf8c009cbc
parent 5c330db73028a5f6c837ef743f174f5533ce5195
...
extend greeting
```

So a commit's hash depends on its whole history. This is the same structure as a Merkle tree or a blockchain: you cannot alter an old commit without changing the hash of every commit after it.

A **merge commit** is simply a commit with two (or more) parent lines:

```text
tree 2d0eb2635f2ec2c62cd86b576d04a945ce44a31a
parent 0e8571558bf1fffe4e8a2d3a85a06b45a04def42
parent b8e41fe6e9a9f60fac0f3d82155561ed96d5c392
...
Merge branch 'fix_1'
```

If the branch being merged already contains the current branch's tip, Git does a **fast-forward** instead. It just moves the branch pointer forward and creates no merge commit. Use `--no-ff` when you want the merge recorded anyway.

## Refs: branches are 41-byte files

A **branch** is a file under `.git/refs/heads/` that contains one commit hash and a newline, 41 bytes in total. Creating a branch writes one small file, which is why branching in Git is instant. Committing on a branch writes the new commit and then overwrites that file with the new hash.

**HEAD** is the file `.git/HEAD`. Normally it holds a symbolic reference such as `ref: refs/heads/main`, meaning "the current branch is main". If it holds a raw commit hash instead, you are in **detached HEAD** state: new commits are not attached to any branch and can be lost when you switch away. (`git reflog` can still find them for a while.)

Remote-tracking branches such as `origin/main` are the same kind of file under `refs/remotes/`, updated by `git fetch`. A pull request on GitHub is just a proposal to merge one ref into another. Tags are refs under `refs/tags/`. Big repositories collect many refs into a single `packed-refs` file, but the meaning is the same.

## What common commands actually do

- **`git add`**: writes blobs for the changed files and records them in the **index** (`.git/index`), the staging area that will become the next tree.
- **`git commit`**: writes trees from the index, writes a commit pointing to them and to the current commit, then moves the current branch ref.
- **`git switch` / `git checkout`**: points HEAD at another branch and rewrites the working folder to match that commit's tree.
- **`git cherry-pick`**: computes the change a commit made relative to its parent and replays it on the current branch as a *new* commit. The content and author date may be identical, but the parent and committer time differ, so the hash differs. In my practice history, the same message "added to greeting" appears twice with two different hashes for exactly this reason.
- **`git rebase`**: a series of cherry-picks followed by moving the branch ref. The old commits still exist until garbage collection removes them; rebasing never edits a commit, it makes new ones.
- **`git reset --hard <commit>`**: moves the branch ref (and resets the index and the working folder). The commits that are "gone" are only unreferenced.

## Deleted is not gone: secrets in history

As part of the practice I committed a file named `secrets` and removed it in a later commit. The removal only creates a new tree without that entry. The old commit, its tree and the blob are still in the object store, and anyone who clones the repository gets them:

```text
$ git log --all --oneline -- secrets.txt
e3f8843 delete secrets
47ac70c add secrets
$ git show 47ac70c:secrets.txt       # the content is still there
```

If a real credential ever reaches a pushed commit:

1. **Revoke or rotate the credential first.** Assume it has been copied.
2. Then rewrite history with `git filter-repo` (or BFG Repo-Cleaner) to drop the file from every commit, and force-push. Every commit hash after that point changes, so everyone else must re-clone.
3. Prevent a repeat: list secret files such as `.env` in `.gitignore` from the start, keep credentials in environment variables or a secret manager, and consider a pre-commit secret scanner.

Removing the file in a new commit is not enough, because content-addressed objects are never edited, only left unreferenced.

## Key takeaways

- Git is a key-value store: the key is the hash of the content, the value is a blob, tree, commit or tag.
- A commit is a full snapshot (root tree) plus parent links; diffs are computed, not stored.
- Branches, tags and HEAD are tiny files holding a hash, which is why branching is cheap.
- Cherry-pick and rebase create new commits with new hashes; the originals remain until garbage collection.
- Deleting a file in a later commit does not remove it from history. Rotate leaked secrets, then rewrite history.

## Resources

- Scott Chacon and Ben Straub, *Pro Git*, 2nd ed., Chapter 10 "Git Internals" (objects, references, packfiles). <https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain>
- Git reference documentation: `git cat-file`, `git hash-object`, `gitglossary`, `git reflog`. <https://git-scm.com/docs>
- `git filter-repo` documentation (removing files from history). <https://github.com/newren/git-filter-repo>
- GitHub Docs, "Removing sensitive data from a repository". <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository>

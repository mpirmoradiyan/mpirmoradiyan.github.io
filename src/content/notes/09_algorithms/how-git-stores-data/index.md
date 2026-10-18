---
title: "How Git stores your work: blobs, trees, commits and refs"
slug: how-git-stores-data
date: 2024-04-09
topic: Algorithms
tags: [git, version-control, content-addressing, sha1, branching, merging]
summary: "Git is just a small database of four kinds of objects plus a few pointer files, and once you see that, branches, merges, cherry-picks and why a deleted secret stays in history all make sense."
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

For a long time I thought of commits as "diffs" and branches as "folders". Both are wrong, and that's where most of my Git confusion came from. Under all the commands, Git is a simple database. It saves every version of every file as an object named after the hash of its content. It links those objects into snapshots. And it keeps a few tiny text files that point at the latest snapshot. To get a feel for it, I made a throw-away practice repo and tried branches, merges, pull requests and cherry-picks. These notes are about what each of those actually does to that database.

## An object's name is its hash

Git keeps its data as **objects** in `.git/objects/`. Each object's name is the SHA-1 hash of a short header plus the content. The header is just the type, a space, the size in bytes, and a zero byte. You can get the same hash as Git with a few lines of Python:

```python
import hashlib

data = b"Hello, World!\n"
store = b"blob %d\x00" % len(data) + data   # header + content
print(hashlib.sha1(store).hexdigest())      # 8ab686eafeb1f44702738c8b0f24f2567c36da6d
```

`git hash-object greeting.md` prints the same 40 hex characters for a file with that line in it. On disk the object sits at `.git/objects/8a/b686ea…`. The first two characters are a subfolder, and the content is compressed with zlib. If you decompress the file with `zlib.decompress` you get back `b'blob 14\x00Hello, World!\n'`.

This "name = hash of content" idea (people call it *content addressing*) has three nice side effects:

- **No duplicates.** Two identical files, in any folder and any commit, are stored only once.
- **Corruption shows up.** Change a single bit and the name no longer matches.
- **Nothing gets edited.** You can't change an object. "Changing" something always means making a new object with a new name.

(Newer Git versions can use SHA-256 instead of SHA-1, but the idea doesn't change.)

## The four kinds of objects

1. **Blob**: the raw content of one file. No file name, no permissions, just bytes.
2. **Tree**: one folder listing. Each line has a mode, a type, a hash and a name, like `100644 blob 8ab686ea… greeting.md`. Subfolders are just trees inside trees.
3. **Commit**: points to one root tree (the snapshot of the whole project), plus zero or more parent commits, the author, the committer, timestamps and the message.
4. **Tag** (the annotated kind): a named pointer to another object, usually a commit. It can be signed.

Something that surprised me: after `git add greeting.md`, the blob is already in the object store, because staging writes it. `git commit` then only adds a tree and a commit. You can print any object with `git cat-file -p`:

```text
$ git cat-file -p HEAD
tree 89b6e34971467c6d834d06e64754939a2c5dd043
author Demo <demo@example.com> 1711616400 +0100
committer Demo <demo@example.com> 1711616400 +0100

add greeting

$ git cat-file -p HEAD^{tree}
100644 blob 8ab686eafeb1f44702738c8b0f24f2567c36da6d	greeting.md
```

So a commit stores a **full snapshot**, not a diff. That sounded wasteful to me at first. But files that didn't change just point to blobs that already exist, so a new commit only adds objects for what really changed. Diffs are worked out when you ask for them, by comparing two trees. To save space, `git gc` later packs objects into a *packfile* and stores similar objects as deltas of each other. In my practice repo, ten loose objects turned into one pack. That's only a storage trick, though. It doesn't change the model.

## History is a chain of parent pointers

Each commit names its parent. The parent's hash covers *its* parent, and so on, all the way back to the first commit. You can see it in the second commit of the demo:

```text
tree e6331b9d4678d8dfccdfe0160ddcedaf8c009cbc
parent 5c330db73028a5f6c837ef743f174f5533ce5195
...
extend greeting
```

So a commit's hash depends on its whole history. It's the same structure as a Merkle tree or a blockchain: you can't change an old commit without changing the hash of every commit after it.

A **merge commit** is just a commit with two (or more) parent lines:

```text
tree 2d0eb2635f2ec2c62cd86b576d04a945ce44a31a
parent 0e8571558bf1fffe4e8a2d3a85a06b45a04def42
parent b8e41fe6e9a9f60fac0f3d82155561ed96d5c392
...
Merge branch 'fix_1'
```

If the branch you're merging already contains the tip of your current branch, Git does a **fast-forward** instead. It just slides the branch pointer forward and makes no merge commit. Use `--no-ff` if you want the merge recorded anyway.

## Branches are 41-byte files

A **branch** (Git calls these *refs*) is a file under `.git/refs/heads/` holding one commit hash and a newline. That's 41 bytes in total. Making a branch means writing one tiny file, which is why branching in Git is instant. When you commit on a branch, Git writes the new commit and then overwrites that file with the new hash.

**HEAD** is the file `.git/HEAD`. Usually it holds something like `ref: refs/heads/main`, which just means "the current branch is main". If it holds a raw commit hash instead, you're in **detached HEAD** state. New commits then don't belong to any branch, and you can lose them when you switch away. (`git reflog` can still find them for a while.)

Remote-tracking branches like `origin/main` are the same kind of file, under `refs/remotes/`, and `git fetch` updates them. A pull request on GitHub is just a proposal to merge one ref into another. Tags are refs under `refs/tags/`. Big repos bundle lots of refs into one `packed-refs` file, but it means the same thing.

## What the everyday commands really do

- **`git add`**: writes blobs for the changed files and records them in the **index** (`.git/index`). The index is the staging area that turns into the next tree.
- **`git commit`**: writes trees from the index, writes a commit that points to them and to the current commit, then moves the current branch ref.
- **`git switch` / `git checkout`**: points HEAD at another branch and rewrites your working folder to match that commit's tree.
- **`git cherry-pick`**: works out the change a commit made compared to its parent and replays it on the current branch as a *new* commit. The content and author date can be identical, but the parent and committer time are different, so the hash is different. This confused me at first: in my practice history the same message "added to greeting" shows up twice with two different hashes, and this is exactly why.
- **`git rebase`**: basically a series of cherry-picks, then moving the branch ref. The old commits are still there until garbage collection cleans them up. A rebase never edits a commit, it makes new ones.
- **`git reset --hard <commit>`**: moves the branch ref (and resets the index and working folder). The commits that look "gone" are just no longer pointed to.

## Deleted doesn't mean gone: secrets in history

As part of the practice I committed a file called `secrets` and then removed it in a later commit. The removal only makes a new tree without that entry. The old commit, its tree and the blob are all still in the object store, and anyone who clones the repo gets them:

```text
$ git log --all --oneline -- secrets.txt
e3f8843 delete secrets
47ac70c add secrets
$ git show 47ac70c:secrets.txt       # the content is still there
```

If a real password or key ever ends up in a pushed commit:

1. **Revoke or rotate it first.** Assume someone already copied it.
2. Then rewrite history with `git filter-repo` (or BFG Repo-Cleaner) to drop the file from every commit, and force-push. Every commit hash after that point changes, so everyone else has to re-clone.
3. Stop it happening again: put secret files like `.env` in `.gitignore` from day one, keep credentials in environment variables or a secret manager, and think about a pre-commit secret scanner.

Just deleting the file in a new commit isn't enough. Objects are never edited, they're only left unreferenced.

## Key takeaways

- Git is a key-value store: the key is the hash of the content, the value is a blob, tree, commit or tag.
- A commit is a full snapshot (a root tree) plus links to its parents. Diffs are computed, not stored.
- Branches, tags and HEAD are tiny files holding a hash, so branching is cheap.
- Cherry-pick and rebase make new commits with new hashes. The originals stay until garbage collection.
- Deleting a file in a later commit doesn't remove it from history. Rotate a leaked secret first, then rewrite history.

## Resources

- Scott Chacon and Ben Straub, *Pro Git*, 2nd ed., Chapter 10 "Git Internals" (objects, references, packfiles). <https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain>
- Git reference documentation: `git cat-file`, `git hash-object`, `gitglossary`, `git reflog`. <https://git-scm.com/docs>
- `git filter-repo` documentation (removing files from history). <https://github.com/newren/git-filter-repo>
- GitHub Docs, "Removing sensitive data from a repository". <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository>

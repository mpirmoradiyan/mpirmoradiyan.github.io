// Copies the learning notes from ../100_my_writings_for_learning/notes into src/content/notes,
// strips private front-matter fields and rewrites links between notes to site URLs.
import fs from "node:fs";
import path from "node:path";

const SRC = path.resolve("../100_my_writings_for_learning/notes");
const DST = path.resolve("src/content/notes");
const PRIVATE_FIELDS = ["original", "original_language"];

if (!fs.existsSync(SRC)) {
  console.log(`sync-notes: ${SRC} not found, keeping existing src/content/notes`);
  process.exit(0);
}
fs.rmSync(DST, { recursive: true, force: true });

let count = 0;
for (const topic of fs.readdirSync(SRC).sort()) {
  const tdir = path.join(SRC, topic);
  if (!fs.statSync(tdir).isDirectory()) continue;
  for (const slug of fs.readdirSync(tdir).sort()) {
    const ndir = path.join(tdir, slug);
    const md = path.join(ndir, "index.md");
    if (!fs.existsSync(md)) continue;
    const out = path.join(DST, topic, slug);
    fs.mkdirSync(out, { recursive: true });
    let text = fs.readFileSync(md, "utf8");
    // drop private front-matter fields (single line or indented block that follows)
    const end = text.indexOf("\n---", 3);
    let fm = text.slice(0, end);
    const body = text.slice(end);
    for (const f of PRIVATE_FIELDS) {
      fm = fm.replace(new RegExp(`^${f}:.*(\\n[ \\t]+.*)*`, "m"), "");
    }
    // ../../<topic>/<slug>/ , ../<slug>/ , ../<slug>/index.md  ->  /notes/<slug>/
    let b = body.replace(/\]\((?:\.\.\/\.\.\/[0-9a-z_]+\/|\.\.\/)([a-z0-9-]+)\/?(?:index\.md)?(#[^)]*)?\)/g,
      (_m, s, hash) => `](/notes/${s}/${hash ?? ""})`);
    // single-line display math "$$ ... $$" -> block form so it renders as display math
    b = b.replace(/^([ \t]*)\$\$(.+?)\$\$[ \t]*$/gm, (_m, ind, eq) => `${ind}$$\n${ind}${eq.trim()}\n${ind}$$`);
    fs.writeFileSync(path.join(out, "index.md"), fm + b);
    for (const f of fs.readdirSync(ndir)) {
      if (f !== "index.md" && /\.(png|jpe?g|svg|gif|webp)$/i.test(f)) fs.copyFileSync(path.join(ndir, f), path.join(out, f));
    }
    count++;
  }
}
console.log(`sync-notes: ${count} notes synced`);

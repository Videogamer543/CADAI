# Putting StressViz somewhere other people can reach it

Everything in here is off by default. On your own laptop StressViz has no
password, no rate limits and no upload cap, which is exactly right for a tool
only you can open — and exactly wrong the moment it has an address.

Work through this in order. The first three sections are the ones that cost you
real money or real trust if skipped; the rest is polish.

---

## 1. Rotate the keys — before anything else

Any key that has ever been pasted into a chat window, committed, screenshotted
or shared should be treated as public, because it is. Rotating one takes about
a minute and there is no downside to doing it when you didn't need to.

- **Onshape** — delete the existing API keys in the developer portal and issue
  new ones. Both key pairs discussed while building this are burned.
- **Groq** — console.groq.com/keys → revoke, create, paste the new one into
  `SET_API_KEY.bat`.
- **Tavily** — same, at tavily.com.

Never paste a key into a chat, an issue, or a commit message. `SET_API_KEY.bat`
writes them straight into `.env`, which is where they belong and nowhere else.

**Onshape specifically:** a shared API key on a public app means every visitor
is browsing *your* Onshape documents under your name. If StressViz is going to
read Onshape at all in public, it has to be OAuth, so each visitor authorises
their own account, and the scope has to be **read-documents** and nothing more.

## 2. Publishing the source safely

`.gitignore` already excludes `.env`, `data/kb.json`, `outputs/` and the
virtualenv. The trap is the *order* of your first two commands:

```
git init
git add .gitignore && git commit -m "ignore rules"   <-- this one FIRST
git add . && git commit -m "StressViz"
```

If `.env` lands in the first commit, deleting it later does not remove it — it
stays in the history for anyone who clones the repo. If that has already
happened, rotate the keys again (see §1) and start the repo over; rewriting
history is more trouble than it is worth for a project this size.

Check before you push:

```
git status --porcelain --ignored | findstr /C:".env"
```

`.env` should appear with a `!!` (ignored). If it shows `A ` it is staged, and
you should `git rm --cached .env` immediately.

## 3. Turn the limits on

Open `.env` and add:

```
STRESSVIZ_PUBLIC=1
STRESSVIZ_PASSWORD=something-long-and-not-guessable
```

That single flag switches on every default in `app/limits.py`: 8 chat questions
per visitor per minute and 60 per hour, 400 questions per day across everyone,
6 analyses per visitor per minute and 60 per hour, 200 analyses per day across
everyone, 3,600 seconds of analysis time per day across everyone, 2 solves
running at once, and a 12 MB upload ceiling. Each is individually overridable —
every knob is listed with an explanation in `.env.example`.

Restart, then confirm at `http://localhost:8000/api/health`. The `limits` block
should read `"public": true, "password": true`.

**Two numbers are worth thinking about, and they guard two different bills.**

`STRESSVIZ_CHAT_PER_DAY` guards the API bill. Every question is one Groq call
plus up to four Tavily searches at advanced depth. Per-visitor limits stop one
person hammering the box; they do nothing about two hundred people each politely
asking three questions. The daily ceiling is what turns "unbounded" into a
number you chose. When it is reached the assistant says so plainly and the rest
of StressViz keeps working.

`STRESSVIZ_COMPUTE_SECONDS_PER_DAY` guards the hosting bill, and it only matters
once StressViz is somewhere other than your own laptop. A solve is the one
genuinely expensive thing in this app — tens of seconds of CPU — and CPU is the
unit hosts meter in. Capping the *number* of analyses is a proxy, because a
plate outline is two seconds and a tetrahedral mesh is ninety; capping the
seconds caps the thing itself. `STRESSVIZ_SOLVE_PER_DAY` sits alongside it as a
cruder count. Both are global, both reset at UTC midnight, and both are reported
as `compute_seconds_used_today` and `solves_used_today` on `/api/health` so you
can see how close a busy day came.

Each solve is charged when it finishes, including one that failed — a solve that
crashed after eighty seconds still cost eighty seconds, and not charging for
failures would make a crash loop the cheapest way to exhaust the budget.

The password is HTTP Basic: the browser shows its own login box, there is no
login page to build and no session to store. It is only as private as the
connection carrying it, so it is worth having exactly as much as §4 is.

## 4. Hosting

`START_APP.bat` runs uvicorn with no `--host`, so it listens on localhost only —
today, nothing outside your machine can reach StressViz at all. That is a real
protection and you are about to remove it deliberately.

To serve it beyond your own machine you need `--host 0.0.0.0`, and you need
HTTPS in front of it. Do not skip the second part: without TLS the Basic
password and every part you upload cross the network in the clear.

The `Dockerfile` in this folder already binds `0.0.0.0:8000` and is the easiest
route — most hosts (Fly, Render, Railway, a small VPS with Caddy) will build it
and terminate TLS for you. If you go that way, set `STRESSVIZ_TRUST_PROXY=1`,
because behind a proxy every request otherwise appears to come from the proxy's
own address and all your per-visitor limits become one shared limit for the
entire internet. Set it **only** behind a proxy: with nothing in front, that
flag lets any visitor invent a new identity per request with one header.

Two more things about the container: the knowledge base is not copied into the
image (`data/` is not in the `COPY` lines and `data/kb.json` is git-ignored), so
mount it as a volume or the deployed assistant will have no library at all. And
`outputs/` is written to the container's own filesystem, which most hosts throw
away on restart — mount that too if the saved images matter.

## 5. Say what it is

The sidebar now carries a permanent line: linear elastic FEM on a simplified
shape, no welds, fasteners, impacts, fatigue life or safety factors, check
anything load-bearing yourself. Leave it there. Someone will otherwise decide a
green picture means a bracket is fine, and a green picture means the model of
the bracket is fine, which is not the same claim.

While you are being honest about limits, the **thickness disagreement** is worth
resolving before strangers meet it: the load-context box accepts "1/4 in 7075
plate" while a STEP file measures its own thickness and silently wins, and the
material dropdown can disagree with both. StressViz reports which source it
used, but a visitor who types a number and gets a different one is a visitor who
stops trusting the whole tool.

## 6. What is in the library

Whatever is in `data/kb.json` becomes something a public bot quotes. Two
questions before that is fine:

- Do the sources allow it? Ingesting frcdesign.org, FIRST material or Onshape's
  documentation for your own reference is one thing; republishing it through a
  bot anybody can query is another. Keep the source links visible — StressViz
  already cites every chunk it uses, and that citation is what makes this
  quoting rather than passing off.
- Is anything in there yours and private? Team notes, a design review, an
  unreleased robot. Option **7** in `ADD_TO_LIBRARY.bat` lists everything
  indexed; option **9** removes a document.

## 7. Before you hand out the link

- [ ] Keys rotated, `.env` not in git, `.env.example` still placeholders only
- [ ] `STRESSVIZ_PUBLIC=1` and a password set; `/api/health` confirms both
- [ ] HTTPS in front; `STRESSVIZ_TRUST_PROXY=1` **only** if there is a proxy
- [ ] `STRESSVIZ_CHAT_PER_DAY` set to a number you would happily pay for
- [ ] `STRESSVIZ_COMPUTE_SECONDS_PER_DAY` set to fit whatever the host gives you
      free — 3,600/day keeps a month inside Cloud Run's 180,000 vCPU-seconds
- [ ] Uploaded a 50 MB file and been refused; asked ten questions fast and been
      told to slow down
- [ ] Library reviewed for anything private
- [ ] Deleted the stray duplicate tree on the Desktop so you ship one copy

# Putting StressViz on a free Hugging Face Space

The result is a permanent HTTPS address — `https://<your-username>-stressviz.hf.space`
— on hardware with 2 vCPU and 16 GB of RAM, at no cost. The RAM is the reason
this host and not a cheaper-looking one: a gmsh mesh plus a 3D solve will not
fit in the 512 MB most free tiers give you, and what you get there is a killed
process rather than an error message.

Read [GOING_PUBLIC.md](GOING_PUBLIC.md) first if you haven't. Everything in it
applies the moment this Space is live.

---

## 1. Create the Space

At <https://huggingface.co/new-space>:

| Field | Value |
|---|---|
| Owner | your account |
| Space name | `stressviz` |
| License | your choice |
| SDK | **Docker** → **Blank** |
| Hardware | CPU basic · 2 vCPU · 16 GB · FREE |
| Visibility | Public *(see §5 before choosing)* |

The URL it gives you is `https://huggingface.co/spaces/<user>/stressviz`. The
app itself serves from `https://<user>-stressviz.hf.space` — that second one is
what goes in the browser and what Onshape needs. Hyphens, not slashes.

## 2. Push the code

```
cd C:\Users\jcshu\OneDrive\Desktop\Claude\stressviz-py

git init
git add .gitignore && git commit -m "ignore rules"     <-- FIRST, always
git add . && git commit -m "StressViz"

git remote add space https://huggingface.co/spaces/<user>/stressviz
git push space main
```

That ordering is not a style preference. If `.env` lands in the first commit,
deleting it later does not remove it from the history, and the keys are public
to anyone who clones.

Check before you push:

```
git status --porcelain --ignored | findstr /C:".env"
```

`.env` must appear with `!!` (ignored). If it shows `A ` it is staged — run
`git rm --cached .env` immediately.

**The knowledge base needs one extra step.** `data/kb.json` is git-ignored on
purpose, and the Dockerfile copies `data/`, so without this the deployed
assistant starts with an empty library:

```
git add -f data/kb.json data/pocket_cal.json data/Onshape_Material_Library.csv
git commit -m "ship the library"
git push space main
```

Force-adding it for the Space is fine; leaving it ignored for a public GitHub
mirror is still right, because it is large, regenerable, and it is *your*
documents.

## 3. Set the secrets

Space → **Settings** → **Variables and secrets**. Add each as a **Secret**, not
a variable — variables are visible to anyone looking at the Space.

```
GROQ_API_KEY              your rotated Groq key
TAVILY_API_KEY            your rotated Tavily key
STRESSVIZ_PUBLIC          1
STRESSVIZ_PASSWORD        something long and not guessable
STRESSVIZ_TRUST_PROXY     1
```

`STRESSVIZ_TRUST_PROXY=1` matters here. A Space sits behind Hugging Face's
router, so without it every visitor arrives wearing the router's IP address and
your per-visitor rate limits collapse into one shared limit for the entire
internet. It is correct here for exactly the reason it would be dangerous on a
machine with nothing in front of it.

`.env` is never uploaded. `app/limits.py` and `app/chat.py` read environment
variables, and Space secrets arrive as environment variables, so nothing in the
code changes between your laptop and here.

Adding a secret restarts the Space. Give it a few minutes on the first build —
scipy, opencv and gmsh are not small.

## 4. Point Onshape at it

Back on the Onshape application you created, press **+** under Redirect URLs and
add the second entry, keeping the localhost one so you can still develop:

```
http://localhost:8000/api/onshape/callback
https://<user>-stressviz.hf.space/api/onshape/callback
```

Then add three more Space secrets:

```
ONSHAPE_OAUTH_CLIENT_ID       on_...
ONSHAPE_OAUTH_CLIENT_SECRET   the one shown once at creation
ONSHAPE_OAUTH_REDIRECT        https://<user>-stressviz.hf.space/api/onshape/callback
```

`ONSHAPE_OAUTH_REDIRECT` must match the registered URL character for character.
A trailing slash, `http` where you registered `https`, or `127.0.0.1` where you
registered `localhost` are all different URLs, and Onshape's refusal does not
say which of them is wrong. Locally, keep `ONSHAPE_OAUTH_REDIRECT` set to the
localhost form in `.env`; the deployed Space uses its own.

Do **not** set `ONSHAPE_ACCESS_KEY` or `ONSHAPE_SECRET_KEY` as Space secrets.
Those are your personal key pair; on a public deployment they would mean every
visitor reads *your* documents under *your* name. They exist for local
development and nowhere else.

## 5. Public or private

A public Space means the code, the `data/` you shipped, and the app itself are
all visible. That is usually what you want — but the library is worth a look
first, because whatever is in `data/kb.json` becomes something a bot anybody can
query will quote back. `ADD_TO_LIBRARY.bat` option **7** lists everything
indexed and option **9** removes a document.

A private Space is only reachable while signed in as you, which makes
`STRESSVIZ_PASSWORD` redundant and makes sharing the link impossible. Public
plus a password is the combination that lets you hand the URL to one person.

One caveat worth testing rather than trusting: HTTP Basic auth travels through
Hugging Face's router, and if the password prompt does not appear when you open
the Space, fall back to a private Space or say so and I'll switch the password
check to a cookie-based one.

## 6. Check it worked

Open `https://<user>-stressviz.hf.space/api/health`. You want:

```json
{
  "ok": true,
  "chat_ready": true,
  "limits": { "public": true, "password": true, "trust_proxy": true },
  "onshape": {
    "oauth_configured": true,
    "redirect_uri": "https://<user>-stressviz.hf.space/api/onshape/callback"
  }
}
```

`/api/health` is deliberately outside the password wall so an uptime check needs
no credentials and you can still see the server is alive if you lock yourself
out. It reports configuration only, never a secret.

Then, in order:

- `/api/kb` — confirms the library shipped. `docs: 0` means step 2's force-add
  did not happen.
- Upload a part and run a 3D solve. This is the one that would have failed on a
  512 MB host.
- Visit `/api/onshape/connect`. Onshape should show a consent screen listing
  *only* document read access. If it asks for write or delete, the wrong boxes
  are ticked on the application — fix it there, not here.
- After connecting, `POST /api/onshape/massprops` with `{"url": "<a part studio
  link>"}` should return real mass and volume. That single call proves the
  application, the redirect, the token exchange and the read permission are all
  correct.

## 7. What this host is not

Free Spaces sleep when idle and take a moment to wake, so the first visitor
after a quiet spell waits. The disk is not persistent: `outputs/` survives until
the next restart and no longer, and anything a visitor saves is gone after a
rebuild. Onshape connections live in memory, so a restart signs everyone out —
one click to reconnect, no data lost.

None of that matters for showing the tool to a team. All of it matters if it
becomes something people depend on, and that is the point to pay for hardware
rather than to re-architect.

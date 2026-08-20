# Putting StressViz on Google Cloud Run

The result is a permanent HTTPS address — `https://stressviz-<numbers>.us-central1.run.app`
— that sleeps when nobody is using it and costs nothing while asleep. Cloud Run
runs the same `Dockerfile` you already have, so there is no second version of
the app to maintain.

Read [GOING_PUBLIC.md](GOING_PUBLIC.md) first if you haven't. Everything in it
applies the moment this is live.

---

## 0. Where you are now

You are in the **90-day free trial** with $300 of credit. Two things follow from
that, and they are the whole answer to "will I get charged":

**During the trial you cannot be charged.** Google will not take money from your
card until you press Upgrade, and the trial does not upgrade itself. If the
credit runs out or the 90 days expire, your resources stop and the app goes
down. That is the failure mode — an outage, not a bill.

**The Always Free tier is already running, right now.** 180,000 vCPU-seconds,
360,000 GiB-seconds and 2 million requests every month, on the trial account as
well as on a paid one. Free-tier usage does not touch the $300 — the credit only
pays for what spills past the allowance. At the 1 vCPU / 2 GiB size below, both
limits bind at roughly **50 hours of actual request-handling per month** — about
6,000 thirty-second solves. A tool you show a team will not approach that, which
means in practice the credit is untouched and the balance stays at $300.

So: the next 90 days are free and riskless. Section 5 is what to do before you
upgrade.

## 0b. Staying up past 4 November

The trial and the Always Free tier are two different things, and the difference
is not the one most people assume. **The free tier is not a reward for
upgrading — it is running already, on the trial account, today.** Every month
your usage is settled in this order: the free allowance covers what it covers,
then the $300 credit pays for the overflow, then your card. At this app's size
the first line covers everything, which is why the credit balance is likely to
still read $300 in November.

What expires on **4 November 2026** is not the allowance. It is the *account*.
An unupgraded trial billing account is closed at the end of the 90 days, its
projects are stopped, and after a 30-day grace period the resources are deleted
outright. That is the failure — the app goes down not because the free tier ran
out but because the account it lives in was closed.

**So the fix is to press Upgrade, and pressing Upgrade does not charge you
anything.** It converts the account from trial to paid, which changes nothing
about the monthly settlement above except that there is no longer an expiry date
attached to it. Your unspent credit carries over (still usable within the
original 90 days). Concretely: Billing → **Upgrade**, or the *Activate full
account* banner across the top of the console. Do it any time before
4 November — early costs nothing and late costs an outage.

**What keeps you inside the allowance is already in the deploy command in
section 3**, and it is worth knowing which flags are doing that work:

`--region us-central1` — the free tier is applied as a discount on Tier 1
pricing only. Deploy to London or Sydney and there is no allowance to be inside
of; you bill from the first second.

**Scale to zero** — the default `--min-instances 0`, which is why there is no
flag for it. You are billed for startup, request handling and shutdown, and for
nothing in between. An idle app consumes none of the 180,000 vCPU-seconds. The
moment you set `--min-instances 1` to kill cold starts, you are buying a
container that runs 2.6 million seconds a month against a 180,000-second
allowance, and that is a real bill.

`--max-instances 1` and `--cpu 1 --memory 2Gi` — these fix the *rate* at which
the allowance can drain. One instance at 1 vCPU / 2 GiB burns 1 vCPU-second and
2 GiB-seconds per second of work, so both limits bind at the same place: about
50 hours of active request-handling a month, roughly 6,000 thirty-second solves.
Going to 2 vCPU halves that. Going to 4 instances quarters it.

`STRESSVIZ_PUBLIC=1` and `STRESSVIZ_PASSWORD` from section 4 — these do not
touch Google's meter directly, but they are what stops someone discovering the
URL and spending your 50 hours in an afternoon.

Two things are genuinely not free even inside Always Free, and both are pennies:
outbound data past 1 GiB/month, and Artifact Registry storage past 0.5 GiB at
about $0.10/GiB/month — this image is a few GiB, so expect fifteen to
twenty-five cents a month once the credit is gone. To keep it there, prune old
revisions after a few deploys:

```
gcloud artifacts docker images list us-central1-docker.pkg.dev/project-c5816071-45b5-41b8-bf1/cloud-run-source-deploy --include-tags
gcloud artifacts docker images delete <image>@<digest> --delete-tags
```

If you would rather the card never be reachable at all, the alternative is to
not upgrade and instead redeploy the same `Dockerfile` to a host with a
permanent no-card free tier — Azure Container Apps grants the same
180,000 vCPU-seconds and 360,000 GiB-seconds monthly with no trial clock. That
is a real option, but it is a migration; upgrading is a button.

## 1. Turn on the three services

In the console, click the **`>_`** icon in the top bar to open Cloud Shell — a
Linux terminal in the browser, already signed in as you, nothing to install.
Paste:

```
gcloud config set project project-c5816071-45b5-41b8-bf1
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

That takes a minute and only has to happen once. `run` serves the app,
`cloudbuild` turns your Dockerfile into an image, `artifactregistry` stores it.

### 1b. Let the builder read your upload

Enabling the services is not enough, and the way you find that out is a deploy
that fails several minutes in. Run this too:

```
gcloud projects add-iam-policy-binding project-c5816071-45b5-41b8-bf1 \
  --member=serviceAccount:282761536722-compute@developer.gserviceaccount.com \
  --role=roles/run.builder
```

(`282761536722` is this project's *number*, not its ID. `gcloud projects
describe project-c5816071-45b5-41b8-bf1 --format="value(projectNumber)"` prints
it for any other project.)

**Why this is necessary.** `gcloud run deploy --source .` does not build in
Cloud Shell. It zips the folder, uploads it to a Cloud Storage bucket named
`run-sources-<project>-<region>`, and hands the job to Cloud Build, which runs
as the *Compute Engine default service account* rather than as you. Until May
2024 that account was given the Editor role at creation and could read anything.
The `iam.automaticIamGrantsForDefaultServiceAccounts` policy is now enforced by
default, so on a project created since then the builder starts with no
permissions at all — including no permission to read the source you just
uploaded on its behalf.

The failure is confusing because it names *storage* when the thing missing is a
*role*:

```
Uploading sources...failed
ERROR: (gcloud.run.deploy) INVALID_ARGUMENT: Invalid build request. could not
resolve source: googleapi: Error 403: <number>-compute@developer.gserviceaccount.com
does not have storage.objects.get access to the Google Cloud Storage object.
```

`roles/run.builder` is the bundle Cloud Run's own documentation asks for here:
reading the source bucket, writing the image to Artifact Registry, and writing
build logs. Grant it, give IAM a minute or two to propagate, then re-run the
section 3 deploy command unchanged. The failed attempt created nothing except
the Artifact Registry repository, which you wanted anyway — so you will not be
asked the `(Y/n)` repository question a second time.

## 2. Get the code to Google

Cloud Shell has its own little filesystem, and your code is on your Windows
machine, so it has to travel.

**Do not use Upload → Folder on `stressviz-py` directly.** It takes everything,
and `stressviz-py\.venv` is several hundred megabytes of Windows virtualenv that
Cloud Shell cannot run and that the image rebuilds from `requirements.txt`
anyway. It would also carry `.env` — your keys — into a cloud filesystem you
will forget you left them in.

**Instead, double-click `MAKE_CLOUD_ZIP.bat`.** It builds `stressviz-cloud.zip`
next to itself containing `app`, `static`, `tools`, `data`, `requirements.txt`,
`Dockerfile` and `.gcloudignore` — and nothing else. It warns you if
`data\kb.json` is missing, since that is the difference between a deployed
assistant with a library and one that cites nothing.

Then in Cloud Shell: **⋮** menu → **Upload** → **File**, pick the zip, and run

```
unzip -o stressviz-cloud.zip
cd stressviz-py
```

**The other way — install the CLI locally** from
<https://cloud.google.com/sdk/docs/install>, then run `gcloud init` and deploy
straight from your own folder. Better if you plan to redeploy often, since it
skips the upload every time. `.gcloudignore` does the excluding for you in that
case, so `.venv` and `.env` still stay behind.

### Why `.gcloudignore` has to exist

This one is a trap rather than an optimisation, and it is worth knowing about
because it fails *silently*. When there is no `.gcloudignore`, `gcloud run
deploy --source .` falls back to reading `.gitignore` — and `.gitignore`
excludes `data/kb.json` on purpose, because a 900 KB knowledge base does not
belong in a git repo. It very much does belong in the image. Without the
`.gcloudignore` file the build succeeds, the app starts, and the assistant comes
up with an empty library. That is what `/api/kb` reporting `docs: 0` in section
7 means. The file is already in your folder; just don't delete it.

## 3. Deploy

```
cd ~/stressviz-py

gcloud run deploy stressviz \
  --source . \
  --region us-central1 \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 1 \
  --concurrency 20 \
  --timeout 600 \
  --allow-unauthenticated
```

The first build takes ten to fifteen minutes — scipy, opencv and gmsh are not
small — and prints a URL at the end. That URL is yours permanently. Later
deploys reuse the cached layers and take two or three minutes.

**The URL does not exist until this command finishes**, which is why every
example in this file writes it as a placeholder. Google builds it as
`https://stressviz-<project-number>.us-central1.run.app` — the service name you
chose, then your project *number*, which is a different thing from the project
*ID* you set in section 1. If you want to know it in advance:

```
gcloud projects describe project-c5816071-45b5-41b8-bf1 --format="value(projectNumber)"
```

Either way, copy the URL the deploy prints and use that, rather than assembling
it yourself — a small number of services get a hash-based address instead, and
the printed one is always right. To get it back later:

```
gcloud run services describe stressviz --region us-central1 --format="value(status.url)"
```

Every flag there is load-bearing:

`--region us-central1` because **the free tier only applies in Tier 1 regions**.
London or Sydney would work and would bill you from the first second.

`--memory 2Gi` because Cloud Run's default is 512 MiB, and a gmsh mesh plus a 3D
solve in 512 MiB is a killed process rather than an error message. This is
exactly the wall you would have hit on Render's free tier.

`--max-instances 1` for two reasons, and the first one is not about money.
**StressViz keeps state in the process's own memory** — your visitors' Onshape
sessions, and the rate-limit counters. Cloud Run's instinct is to run several
copies of the container at once, and if it does, a visitor who starts an Onshape
login on one copy and comes back to another gets told *"that sign-in did not
start here"*, unreproducibly. Worse, per-visitor rate limits become per-copy, so
five copies means five times the spending limit you thought you set. One
instance keeps both honest. It also caps your compute spend at "one container",
which is the only cap here that is actually enforced rather than merely
reported.

`--concurrency 20` lets that single instance handle twenty requests at once —
plenty, because chat requests spend their time waiting on Groq rather than on
CPU. Simultaneous *solves* are separately capped at 2 by
`STRESSVIZ_SOLVE_CONCURRENT`, so heavy requests queue instead of fighting.

`--timeout 600` gives a large 3D solve ten minutes. The default 300 seconds
would cut the biggest parts off mid-solve.

`--allow-unauthenticated` is what makes it a public app rather than one only you
can open. Your password (next section) is what stands in the gap.

## 4. Set the environment

```
gcloud run services update stressviz --region us-central1 \
  --set-env-vars STRESSVIZ_PUBLIC=1,STRESSVIZ_TRUST_PROXY=1,\
STRESSVIZ_PASSWORD=something-long-and-not-guessable,\
GROQ_API_KEY=your-rotated-groq-key,\
TAVILY_API_KEY=your-rotated-tavily-key
```

**This command only works after section 3 has actually succeeded.** `update`
changes an existing service; if the build failed, there is no service to change
and you get `ERROR: (gcloud.run.services.update) Service [stressviz] could not
be found.` That message means the deploy never finished — go back and fix that
first. It does not mean anything is wrong with the variables.

Better still, fold the two together and pass `--set-env-vars` on the *deploy*
command itself. Then the very first revision comes up already password-protected
and rate-limited. Doing it in two steps leaves a window — usually a minute or
two, but it only takes one — where the service is live, unauthenticated and has
no password and no limits at all:

```
gcloud run deploy stressviz --source . --region us-central1 \
  --memory 2Gi --cpu 1 --max-instances 1 --concurrency 20 --timeout 600 \
  --allow-unauthenticated \
  --set-env-vars STRESSVIZ_PUBLIC=1,STRESSVIZ_TRUST_PROXY=1,\
STRESSVIZ_PASSWORD=<your-password>,\
GROQ_API_KEY=<your-groq-key>,\
TAVILY_API_KEY=<your-tavily-key>
```

`STRESSVIZ_PUBLIC=1` is the master switch that turns on every limit in
`app/limits.py` at once. Without it StressViz has no rate limits at all, which
is correct for a tool only you can reach and ruinous for one anybody can.

`STRESSVIZ_TRUST_PROXY=1` is correct here and would be dangerous without a proxy
in front. Cloud Run puts Google's front end between the internet and your
container, so without this flag every visitor arrives wearing Google's IP and
your per-visitor limits collapse into one shared limit for the whole internet.

`.env` is never uploaded and never read here. `app/limits.py` and `app/chat.py`
read environment variables, and that is what these become. The values are the
same ones sitting in your local `.env` — open it in Notepad to read them off,
and note the keys go into this command, never into a chat window or a
screenshot.

Rotate the Groq and Tavily keys before you paste them, if you haven't. Any key
that has been in a chat window should be considered spent.

**`--set-env-vars` versus `--update-env-vars`, because the difference will bite
you.** `--set-env-vars` replaces the *entire* environment with what you list, so
running it a second time for the Onshape variables in section 6 would silently
delete your password, your API keys and `STRESSVIZ_PUBLIC` — and the first sign
would be a public app with no rate limits and a broken assistant. Use
`--set-env-vars` once, here, to establish the set. Every command after this one
uses `--update-env-vars`, which adds and changes without removing. If you are
ever unsure what is actually set:

```
gcloud run services describe stressviz --region us-central1 \
  --format="value(spec.template.spec.containers[0].env)"
```

## 5. Not getting charged

Layers, strongest first. The first three actually stop money moving; the rest
reduce how much there is to stop.

**The trial itself.** Until you press Upgrade, no charge is possible. This is
the strongest protection you will ever have and it is already on. Put a note in
your calendar for **1 November 2026**, three days before it expires — and read
section 0b before that date, because letting the trial simply lapse takes the
app offline rather than keeping it free.

**`--max-instances 1`.** A hard technical ceiling on compute. One container can
only burn one container's worth of CPU no matter how many people arrive; the
rest queue. This is the cap that is enforced.

**The app's own limits**, already written and tested, switched on by
`STRESSVIZ_PUBLIC=1`. Two different bills are being defended here and it is
worth keeping them apart.

*Chat costs Groq and Tavily* — every message is a Groq call plus up to four
searches, and Google has nothing to do with it. Defaults: 8/minute and 60/hour
per visitor, **400/day across everyone**.

*Solves cost Google* — a tetrahedral solve is the only genuinely expensive thing
here in CPU, and CPU is what the free tier is measured in. Defaults: 6/minute
and 60/hour per visitor, **200/day across everyone**, **3,600 seconds of solve
time per day across everyone**, 2 running at once.

That last one is the important one, and it is the only limit in the app that is
denominated in the same unit Google bills in. Capping *analyses* is a proxy — a
plate outline is two seconds and a tetrahedral mesh is ninety, so "200 a day"
is anywhere between six minutes and five hours of machine time depending on what
people upload. `STRESSVIZ_COMPUTE_SECONDS_PER_DAY` caps the machine time itself.
At the default 3,600 s/day that is about 111,000 vCPU-seconds a month against a
180,000 allowance, leaving room for chat requests and cold starts. Each solve is
charged when it finishes, including one that crashed — otherwise a crash loop
would be the cheapest way to burn your allowance — so the budget can be overrun
by at most one analysis.

The per-visitor limits cannot see a hundred well-behaved strangers; these global
ones can. Set them to what you are willing to pay:

```
gcloud run services update stressviz --region us-central1 \
  --update-env-vars STRESSVIZ_CHAT_PER_DAY=150,STRESSVIZ_CHAT_PER_MIN=4,\
STRESSVIZ_SOLVE_PER_DAY=100,STRESSVIZ_COMPUTE_SECONDS_PER_DAY=1800
```

`/api/health` reports `solves_used_today` and `compute_seconds_used_today`
alongside the caps, so you can see how close a busy day actually got without
opening the Google console. Both reset at UTC midnight, and both reset on a
restart — deliberately, because a state file that goes stale or is lost with the
container can lock you out of your own tool.

**The password.** By far the cheapest control there is: an address nobody can
use is an address nobody can run your bill up on. `STRESSVIZ_PASSWORD` gives
you a link you can hand to one team and nobody else.

**A spend cap budget** — Billing → **Budgets & alerts** → **Create budget** →
choose **Spend cap enforcement** rather than the alerts-only kind. This is new
as of July 2026 and it is the thing that did not used to exist: a budget that
**actually pauses the service** instead of only emailing you about it. Scope it
to this project and to **Cloud Run** (one project and one service per cap, and
the period is fixed at monthly), set the amount to something like $5, and it
will notify at 50%, 80% and 100% and stop new usage at the cap. Four things to
know before you rely on it: it is in **Preview**, so it carries pre-GA terms;
enforcement is *not* instantaneous, because it rides on billing data that lags,
so treat it as a backstop and not as a fence; requests already in flight finish
and still bill; and once it fires you have to **lift it by hand**, after which
the service takes up to an hour to come back. Make a second, ordinary **alerts**
budget at $1 alongside it, so you hear about money moving long before the cap is
anywhere near.

**A plain budget alert** at $1, at the same screen, choosing the alerts-only
kind. Be clear about what this one does: **it emails you, it does not stop
anything.** It is the early-warning half of the pair above.

What is *not* free, so you know where the pennies come from: outbound data
transfer past 1 GiB/month (your stress-map PNGs and 3D mesh data), and image
storage past 0.5 GiB in Artifact Registry at about $0.10/GiB/month. This image
is a few GiB, so budget fifteen to twenty-five cents a month, and prune old
revisions to keep it there. Those cents come out of the $300 while the credit
lasts, which is why a $5 spend cap will sit untouched for a long time.

## 5b. Knowing where you stand

Three places, in increasing order of how much they tell you.

**What you would be paying if the credit vanished tomorrow.** This is the number
that actually matters and almost nobody finds it. Billing → **Reports**, filter
to this project, then in the right-hand panel under **Savings** or **Credits**,
**untick "Promotional credits"** and leave the free-tier credits ticked. The
chart redraws as your real ongoing cost — what a fully paid account would owe,
with the $300 taken out of the picture. If that line is flat at zero or a few
cents, upgrading changes nothing about your bill. Check it once a month.

**How much of the allowance you have burned.** Cloud Run → **stressviz** →
**Metrics** → *Billable container instance time*. That graph is the one being
counted against 180,000 vCPU-seconds. Because you are at 1 vCPU, one second on
that graph is one vCPU-second, so the month's total in seconds is directly
comparable to the allowance. Fifty hours is your ceiling; a graph that idles
near zero between short spikes means you are nowhere near it.

**Whether anything moved at all.** The billing account's main page shows the
credit balance. As long as it reads $300, nothing has ever spilled past the free
tier, and no arithmetic is needed.

The honest short version: with `--max-instances 1`, scale-to-zero, and a
password on the door, the first charge you could plausibly see is a fraction of
a dollar of image storage, and it will be paid out of the credit for years
before it reaches your card.

## 6. Point Onshape at it

Take the URL from step 3 and add a second Redirect URL on your Onshape
application, keeping the localhost one so you can still develop:

```
http://localhost:8000/api/onshape/callback
https://stressviz-<numbers>.us-central1.run.app/api/onshape/callback
```

Then tell the server the same thing, character for character:

```
gcloud run services update stressviz --region us-central1 \
  --update-env-vars ONSHAPE_OAUTH_CLIENT_ID=on_...,\
ONSHAPE_OAUTH_CLIENT_SECRET=the-one-shown-once-at-creation,\
ONSHAPE_OAUTH_REDIRECT=https://stressviz-<project-number>.us-central1.run.app/api/onshape/callback
```

A trailing slash, `http` where you registered `https`, or a different subdomain
are all different URLs, and Onshape's refusal does not say which part is wrong.
Keep `.env` on your laptop pointing at the localhost form; the deployed service
has its own.

Do **not** set `ONSHAPE_ACCESS_KEY` or `ONSHAPE_SECRET_KEY` here. Those are your
personal key pair, and on a public deployment they would mean every visitor
reads *your* documents under *your* name. The code refuses to use them whenever
`STRESSVIZ_PUBLIC=1`, so setting them would achieve nothing except leaving them
somewhere they don't belong.

## 7. Check it worked

Open `https://<your-url>/api/health`. You want:

```json
{
  "ok": true,
  "chat_ready": true,
  "limits": { "public": true, "password": true, "trust_proxy": true },
  "onshape": {
    "oauth_configured": true,
    "redirect_uri": "https://<your-url>/api/onshape/callback"
  }
}
```

`/api/health` sits outside the password wall deliberately, so an uptime check
needs no credentials and you can still see the server is alive if you lock
yourself out. It reports configuration only, never a secret.

Then, in order:

- `/api/kb` — confirms the library shipped. `docs: 0` means `data/kb.json` was
  missing from the upload; it is git-ignored, so check it came across.
- Upload a part and run a 3D solve. This is the one that would have died in
  512 MiB.
- Visit `/api/onshape/connect`. Onshape should show a consent screen listing
  *only* document read access. If it asks for write or delete, the wrong boxes
  are ticked on the application — fix it there, not here.
- After connecting, `POST /api/onshape/massprops` with `{"url": "<a part studio
  link>"}` should return real mass and volume. That one call proves the
  application, the redirect, the token exchange and the read permission are all
  correct.

## 8. What this host is not

It scales to zero, so after a quiet spell the first visitor waits through a cold
start while a few gigabytes of scientific Python load. Memory goes with it:
Onshape connections are dropped (one click to reconnect, no data lost) and the
rate-limit counters reset. The disk is not persistent either — `outputs/` lasts
until the instance stops.

None of that matters for showing the tool to a team. All of it matters if it
becomes something people depend on, and that is the point to pay for a minimum
instance rather than to re-architect.

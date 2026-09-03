# Deploying the interview server

## Vercel cannot host this, and it is not a configuration problem

Worth stating first, with the specifics, because the answer is not "try
harder with a different `vercel.json`". Five independent blockers, any one of
which is fatal on its own:

| Blocker | Where it lives | Why Vercel cannot |
|---|---|---|
| **Five WebSockets** | `web/server.py` — `/ws/candidate`, `/ws/audio`, `/ws/presence` ×2, `/ws/interviewer` | Vercel Functions do not support WebSocket servers. The interview *is* these sockets. |
| **In-process session state** | `web.server.SESSIONS`, `web.live.HUB` | Functions are stateless and per-invocation. The session would not exist on the next request. |
| **Continuous filesystem writes** | 18 `.save()` calls on the request path | The filesystem is read-only apart from `/tmp`, which is per-invocation and shared with nothing. |
| **~700 MB of native wheels + 61 MB of weights** | mediapipe, opencv, scipy, onnxruntime, pyarrow, ctranslate2 | The bundle limit is 250 MB unzipped. |
| **Long-lived CPU work** | MediaPipe per frame, Whisper per chunk, a 23 s Gemini call | Functions have a wall-clock ceiling and no concept of a call that stays open. |

This is not a Vercel deficiency. Vercel is built for stateless request/response
work, and this app is a stateful long-lived process holding a video call open.
It needs a **container that keeps running**.

## What to deploy on instead

**Render** is the closest thing to Vercel's experience — connect the repo,
push to deploy, automatic HTTPS — and it supports WebSockets, persistent
disks and long-running processes. This guide uses it. Fly.io and Railway work
identically in shape; see the end.

---

## Before you start

Two things from this project's own documentation, not from me:

1. **All three notices are marked DRAFT and unreviewed** (`docs/WP7a`,
   `docs/WP7b`, `docs/WP7c`). Each says it must not be shown to a real
   participant or candidate until counsel has signed it off.
2. **The subgroup audit (WP8a) has not been run.** `README.md` lists it as one
   of three things to change before this goes near a real hire, because rPPG
   error and AU accuracy both vary with skin tone.

Neither blocks a deployment for a demo, an internal pilot, or testing with
colleagues who know what they are looking at. Both block interviewing a real
candidate whose job depends on it. Deploy accordingly.

---

## Step by step, on Render

### 1. Commit the deployment files

Already in the repo: `Dockerfile`, `.dockerignore`, `render.yaml`. Also
`requirements.txt` — which until now did not list `fastapi`, `uvicorn` or
`python-multipart`, so a fresh machine could not start the server at all.

```bash
git add Dockerfile .dockerignore render.yaml requirements.txt DEPLOY.md
git commit -m "Deployment: container image, Render blueprint, declare web deps"
git push
```

### 2. Check the image builds locally first

Optional but it saves a slow round trip through Render's build queue.

```bash
docker build -t interview-signals .
docker run --rm -p 8000:8000 -e GEMINI_API_KEY="$(grep GEMINI_API_KEY .env | cut -d= -f2)" interview-signals
```

Open http://127.0.0.1:8000 — you should get the session-creation page.

On an Apple Silicon Mac this build runs under emulation and is slow, because
the Dockerfile pins `linux/amd64`. That pin is deliberate: mediapipe ships
x86_64 wheels only, and on arm64 the newest available is 0.10.18, which
cannot satisfy the `>=0.10.30` pin. Render builds natively and is fast.

### 3. Create the service

1. Go to <https://dashboard.render.com> → **New +** → **Blueprint**.
2. Connect your GitHub account and pick `Surajkumarmachani/HR_Hiring_Agents`.
3. Render reads `render.yaml` and shows one web service, `interview-signals`,
   with a 10 GB disk. Approve it.

If you would rather not use the blueprint: **New +** → **Web Service** →
select the repo → Language **Docker** → Instance type **Standard** → then add
the disk and the environment variable by hand as below.

### 4. Set the API key

In the service → **Environment** → **Add Environment Variable**:

```
GEMINI_API_KEY = <your key>
```

`render.yaml` declares it with `sync: false`, which means Render prompts for
it and never stores it in the repo. Do not commit `.env` — it is gitignored
and should stay that way.

Without the key everything works except question generation and the answer
read; the server says so at startup rather than failing mid-interview.

### 5. Pick the instance size

**Standard (2 GB RAM) is the realistic minimum.** In one process you have
MediaPipe's face and pose graphs, a Whisper `base.en` model, an ONNX
sentence embedder and the rolling signal buffers for the live session.
Starter's 512 MB will OOM as soon as a candidate joins.

Free tier will not work at all: no persistent disk, and it spins down after
15 minutes of inactivity — which is a dead link at the exact moment a
candidate clicks it.

### 6. Deploy, and watch the first build

The first build takes 10–20 minutes: it installs ~700 MB of wheels and bakes
the Whisper and embedder models into the image so that no download happens
during a live interview. Later builds reuse the dependency layer and are much
faster.

Watch for `Application startup complete` in the logs.

### 7. Check it works

You get a URL like `https://interview-signals.onrender.com`.

- Open it. You should see **Start an interview**.
- Create a session. Open the candidate link in one tab, the interviewer link
  in another.
- **The camera must work.** `getUserMedia` needs a secure context; Render
  gives you HTTPS automatically, which is exactly why this works deployed
  when it would not over plain HTTP on a LAN.
- Check the transcript appears and the live read populates. If frames arrive
  but nothing is measured, the logs will name the missing piece.

---

## Things that will bite you

**Never raise the instance count above 1.** `SESSIONS` and `HUB` are
process-local. With two replicas the candidate's frames land on one instance
and the panel's socket on the other, and neither can see the interview. It
does not degrade gracefully — it simply does not work. `render.yaml` pins
`numInstances: 1`. Scale by making the box bigger until WP6 replaces the
store.

**A deploy kills any interview in progress.** Render stops the old instance
before starting the new one when a disk is attached, and the sessions live in
that process's memory. Do not push while someone is being interviewed.

**The disk is the consent record.** `/app/out` holds consent records, the
audit log, interview records, uploaded CVs and recordings. Without the
mounted disk every deploy erases the evidence that a candidate agreed to
anything. It is in `render.yaml`; do not remove it.

**Personal data is now on someone else's server.** CVs, recordings and
consent records were on a laptop and are now in a datacentre in whichever
region you picked. `render.yaml` defaults to `singapore`; change `region:` if
you want it elsewhere. Whichever you pick is a fact your candidate notice
should state, since it is where their data actually lives.

**There is no jurisdictional gate any more.** The EU/EEA region gate was
removed: it inferred a legal position from a CDN header that most hosts do
not set, and its `strict` default then refused capture on every one of them.
Consent is the control — itemised, asked of the candidate, enforced in the
pipeline. If a deployment needs a country restriction on top of that, put it
in the CDN or load balancer in front of this app, which actually knows where
the request came from.

---

## Other hosts

Same container, same constraints:

- **Fly.io** — `fly launch` reads the Dockerfile. Use a volume for `/app/out`
  and `min_machines_running = 1`. Best latency control, most config.
- **Railway** — connect the repo, it detects the Dockerfile. Add a volume at
  `/app/out`. Simplest of the three.
- **A plain VM** (Hetzner, DigitalOcean, EC2) — `docker run` with `-v` for
  `out/` and Caddy or nginx in front for TLS. Cheapest at this size, and you
  do the TLS and updates yourself.

**Do not use:** Vercel, Netlify, Cloudflare Workers, AWS Lambda, Google Cloud
Functions. All are stateless request/response platforms and all fail on the
five blockers in the first table.

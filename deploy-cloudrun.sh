#!/usr/bin/env bash
# Deploy the interview server to Google Cloud Run.
#
# WHY CLOUD RUN AND NOT FIREBASE HOSTING OR CLOUD FUNCTIONS
# ---------------------------------------------------------
# Firebase Hosting serves static files; it cannot run this. Cloud Functions
# are stateless and cannot hold a WebSocket. Cloud Run runs the container as
# a long-lived process and DOES support WebSockets, which is what makes it
# the one Google option that fits.
#
# Firebase Hosting can sit in front of Cloud Run -- but its CDN does not
# proxy WebSockets, and this app builds every socket URL from
# `location.host`. Put Hosting in front and the pages load while all five
# sockets fail. So: serve directly from the Cloud Run URL. That needs no code
# change at all.
set -euo pipefail
cd "$(dirname "$0")"

PROJECT="${PROJECT:-}"                       # required
REGION="${REGION:-asia-south1}"              # Mumbai. Nearest to India.
SERVICE="${SERVICE:-interview-signals}"
BUCKET="${BUCKET:-${PROJECT}-interview-out}"

if [[ -z "$PROJECT" ]]; then
  echo "Set PROJECT first:  PROJECT=my-gcp-project ./deploy-cloudrun.sh" >&2
  exit 2
fi

echo "==> project $PROJECT / region $REGION / service $SERVICE"
gcloud config set project "$PROJECT" >/dev/null

echo "==> enabling the APIs this needs"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com

# --- the API key, in Secret Manager rather than an env var ---------------
# An env var is visible to anyone with viewer on the project and shows up in
# the service description. A secret is access-controlled and versioned.
if ! gcloud secrets describe gemini-api-key >/dev/null 2>&1; then
  echo "==> creating the gemini-api-key secret"
  if [[ -f .env ]] && grep -q GEMINI_API_KEY .env; then
    grep GEMINI_API_KEY .env | head -1 | cut -d= -f2- | tr -d '"'"'" \
      | gcloud secrets create gemini-api-key --data-file=- --replication-policy=automatic
    echo "    read from .env"
  else
    echo "    paste your Gemini API key, then press Ctrl-D:"
    gcloud secrets create gemini-api-key --data-file=- --replication-policy=automatic
  fi
else
  echo "==> gemini-api-key secret already exists (leaving it alone)"
fi

# --- persistence ---------------------------------------------------------
# Cloud Run has no disk: the filesystem is in-memory and dies with the
# instance. `out/` holds consent records, the audit log, interview records,
# uploaded CVs and recordings -- the consent records in particular have to
# outlive any container, because they are the evidence the candidate agreed.
# So a Cloud Storage bucket is mounted there.
if ! gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
  echo "==> creating gs://$BUCKET for the out/ directory"
  gcloud storage buckets create "gs://$BUCKET" --location="$REGION" \
    --uniform-bucket-level-access
else
  echo "==> gs://$BUCKET already exists"
fi

# --- build on Google's machines, not on a laptop ------------------------
# The Dockerfile pins linux/amd64 because mediapipe ships x86_64 wheels only.
# Cloud Build is natively amd64, so this is fast -- the same build under
# emulation on an Apple Silicon Mac is not. The longer timeout and bigger
# machine are for the ~700 MB of wheels plus the baked-in Whisper and
# embedder models.
IMAGE="$REGION-docker.pkg.dev/$PROJECT/cloud-run-source-deploy/$SERVICE"
echo "==> building $IMAGE (10-20 min the first time)"
gcloud builds submit --tag "$IMAGE" --timeout=2400 \
  --machine-type=e2-highcpu-8

# --- deploy --------------------------------------------------------------
# Every flag below is load-bearing. Read the comments before changing one.
echo "==> deploying"
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  \
  `# The session lives in this process's memory (web.server.SESSIONS and` \
  `# web.live.HUB). A second instance cannot see the interview the first is` \
  `# holding, so the candidate's frames and the panel's socket would land on` \
  `# different machines and neither would work. ONE. Not a performance` \
  `# setting -- a correctness one, until WP6 replaces the store.` \
  --max-instances=1 \
  \
  `# Kept warm. A cold start reloads MediaPipe's graphs and the Whisper` \
  `# model, which is ~10 s that a candidate would spend staring at a dead` \
  `# page. This is also what most of the monthly cost is.` \
  --min-instances=1 \
  \
  `# THE ONE THAT WILL CATCH YOU. A WebSocket is a single long-lived` \
  `# request, and Cloud Run's default request timeout is 300 s. Leave it at` \
  `# the default and every interview dies, silently, at five minutes in.` \
  `# 3600 is the maximum and fits any interview.` \
  --timeout=3600 \
  \
  `# CPU allocated even between requests. The measurement worker and the` \
  `# transcriber are asyncio tasks that do work while no HTTP request is in` \
  `# flight; throttled, they stall mid-call.` \
  --no-cpu-throttling \
  \
  `# MediaPipe's two graphs, a Whisper model, an ONNX embedder and the` \
  `# rolling signal buffers all live in one process. 2 GB will OOM as soon` \
  `# as a candidate joins.` \
  --cpu=2 --memory=4Gi \
  \
  --set-secrets=GEMINI_API_KEY=gemini-api-key:latest \
  --add-volume=name=out,type=cloud-storage,bucket="$BUCKET" \
  --add-volume-mount=volume=out,mount-path=/app/out

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" \
        --format='value(status.url)')"
echo
echo "==> live at $URL"
echo "    HTTPS is automatic, which is what getUserMedia needs."
echo
echo "    Logs:   gcloud run services logs tail $SERVICE --region $REGION"
echo "    Records: gcloud storage ls gs://$BUCKET/"

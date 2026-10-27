#!/usr/bin/env bash
# Check that the host ports the stack wants are free, then start it with Docker.
#
#   ./scripts/docker-up.sh                 check ports, build, start, wait for health
#   ./scripts/docker-up.sh --check-only    only report port status; start nothing
#   ./scripts/docker-up.sh --strict        stop with an error instead of picking another port
#   ./scripts/docker-up.sh --kafka         also start Kafka + ZooKeeper (async ingestion)
#   ./scripts/docker-up.sh --dev-auth      local login without Okta (dev tokens; APP_ENV=dev only)
#   ./scripts/docker-up.sh --no-build      reuse existing images
#
# Written for bash 3.2 (macOS default): no associative arrays, no mapfile.
set -u
cd "$(dirname "$0")/.." || exit 1

PROJECT="production-rag"
STRICT=0; CHECK_ONLY=0; KAFKA=0; DEV_AUTH=0; BUILD=1
for arg in "$@"; do
  case "$arg" in
    --strict) STRICT=1 ;;
    --check-only) CHECK_ONLY=1 ;;
    --kafka) KAFKA=1 ;;
    --dev-auth) DEV_AUTH=1 ;;
    --no-build) BUILD=0 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# service | env var | default host port
SERVICES="api|API_PORT|8000
frontend|FRONTEND_PORT|8080
qdrant|QDRANT_PORT|6333
qdrant-grpc|QDRANT_GRPC_PORT|6334
redis|REDIS_PORT|6379"
if [ "$KAFKA" = 1 ]; then
  SERVICES="$SERVICES
kafka|KAFKA_PORT|9092
zookeeper|ZOOKEEPER_PORT|2181"
fi

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }

# ── helpers ──────────────────────────────────────────────────────────────────

# Read one KEY=VALUE from .env without sourcing it (it holds secrets).
env_file_value() {
  [ -f .env ] || return 0
  grep -E "^$1=" .env | tail -n1 | cut -d= -f2- | sed -e 's/[[:space:]]*#.*$//' -e 's/^"//' -e 's/"$//' | tr -d '[:space:]'
}

# Is anything listening on this TCP port on this machine?
port_in_use() {
  local p="$1"
  if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then return 0; fi
  # connect test catches listeners lsof cannot see (other users, some VMs)
  (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null && return 0
  return 1
}

# Human-readable owner of a port: a Docker container, or a local process.
port_owner() {
  local p="$1" name
  name=$(docker ps --format '{{.Names}}|{{.Ports}}' 2>/dev/null | grep -E "[:.]$p->" | head -n1 | cut -d'|' -f1)
  if [ -n "$name" ]; then echo "docker container '$name'"; return; fi
  if command -v lsof >/dev/null 2>&1; then
    name=$(lsof -nP -iTCP:"$p" -sTCP:LISTEN 2>/dev/null | awk 'NR==2{print $1" (pid "$2")"}')
    [ -n "$name" ] && { echo "process $name"; return; }
  fi
  echo "an unknown process"
}

# Host ports already published by THIS project's running containers: those are
# not conflicts, they are the stack we are about to (re)start.
OUR_PORTS=" "
collect_our_ports() {
  local line
  line=$(docker ps --filter "label=com.docker.compose.project=$PROJECT" --format '{{.Ports}}' 2>/dev/null \
    | grep -oE '[0-9]+->' | tr -d '>-' | sort -u | tr '\n' ' ')
  OUR_PORTS=" $line"
}

CHOSEN=" "   # ports already assigned in this run
next_free_port() {
  local p="$1"
  while :; do
    p=$((p + 1))
    [ "$p" -gt 65000 ] && { echo 0; return; }
    case "$CHOSEN" in *" $p "*) continue ;; esac
    case "$OUR_PORTS" in *" $p "*) continue ;; esac
    port_in_use "$p" || { echo "$p"; return; }
  done
}

# ── preflight ────────────────────────────────────────────────────────────────

if ! command -v docker >/dev/null 2>&1; then red "docker is not installed or not on PATH."; exit 1; fi
if ! docker info >/dev/null 2>&1; then red "The Docker daemon is not running. Start Docker Desktop and retry."; exit 1; fi

collect_our_ports
echo "Checking host ports for project '$PROJECT'..."
printf '  %-12s %-18s %-8s %s\n' SERVICE VARIABLE PORT STATUS

CONFLICTS=0
EXPORTS=""
OLDIFS="$IFS"; IFS='
'
for row in $SERVICES; do
  IFS="$OLDIFS"
  svc=${row%%|*}; rest=${row#*|}; var=${rest%%|*}; default=${rest#*|}
  # precedence: real environment, then .env, then the default
  eval "wanted=\${$var:-}"
  [ -z "$wanted" ] && wanted=$(env_file_value "$var")
  [ -z "$wanted" ] && wanted=$default
  case "$wanted" in ''|*[!0-9]*) red "  $var must be a number, got '$wanted'"; exit 2 ;; esac

  case "$OUR_PORTS" in
    *" $wanted "*) status="in use by this stack (will be reused)"; chosen=$wanted ;;
    *)
      if port_in_use "$wanted"; then
        owner=$(port_owner "$wanted")
        if [ "$STRICT" = 1 ]; then
          printf '  %-12s %-18s %-8s ' "$svc" "$var" "$wanted"; red "BUSY - $owner"
          CONFLICTS=$((CONFLICTS + 1)); chosen=$wanted; status=""
        else
          chosen=$(next_free_port "$wanted")
          if [ "$chosen" = 0 ]; then red "No free port found near $wanted for $svc."; exit 1; fi
          status="busy ($owner) -> using $chosen instead"
        fi
      else
        status="free"; chosen=$wanted
      fi ;;
  esac
  CHOSEN="$CHOSEN$chosen "
  [ -n "$status" ] && printf '  %-12s %-18s %-8s %s\n' "$svc" "$var" "$chosen" "$status"
  EXPORTS="$EXPORTS $var=$chosen"
  IFS='
'
done
IFS="$OLDIFS"

if [ "$CONFLICTS" -gt 0 ]; then
  red "$CONFLICTS port conflict(s). Free them, set different ports in .env (e.g. API_PORT=8001), or drop --strict."
  exit 1
fi
green "Ports OK."

if [ "$CHECK_ONLY" = 1 ]; then
  echo "Would start with:$EXPORTS"
  exit 0
fi

# ── configuration files ──────────────────────────────────────────────────────

if [ ! -f .env ]; then
  cp .env.example .env
  yellow "Created .env from .env.example. Add your LLM API keys to it."
fi

if [ "$DEV_AUTH" = 1 ]; then
  if [ ! -f .env.dev-auth ]; then
    secret=$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 48)
    {
      echo "# Generated by scripts/docker-up.sh --dev-auth. Local development only; not committed."
      echo "APP_ENV=dev"
      echo "AUTH_MODE=dev_token"
      echo "DEV_JWT_SECRET=$secret"
      echo "TENANT_POLICIES_PATH=config/tenant_policies.dev.yaml"
    } > .env.dev-auth
    chmod 600 .env.dev-auth
    yellow "Created .env.dev-auth (dev tokens, permissive 'dev' tenant policy)."
  fi
elif grep -qE '^AUTH_MODE=okta' .env 2>/dev/null && grep -q 'YOUR-ORG' .env 2>/dev/null; then
  yellow "WARNING: .env still has the placeholder OKTA_ISSUER. The API refuses to start without a real"
  yellow "         issuer. Set OKTA_ISSUER, or run:  ./scripts/docker-up.sh --dev-auth"
fi

# ── start ────────────────────────────────────────────────────────────────────

COMPOSE="docker compose"
[ "$KAFKA" = 1 ] && COMPOSE="$COMPOSE --profile kafka"
UP="up -d"
[ "$BUILD" = 1 ] && UP="up -d --build"

echo "Starting:$EXPORTS"
# shellcheck disable=SC2086
env $EXPORTS $COMPOSE $UP || { red "docker compose failed."; exit 1; }

API_PORT_CHOSEN=$(echo "$EXPORTS" | tr ' ' '\n' | grep '^API_PORT=' | cut -d= -f2)
FRONTEND_PORT_CHOSEN=$(echo "$EXPORTS" | tr ' ' '\n' | grep '^FRONTEND_PORT=' | cut -d= -f2)
QDRANT_PORT_CHOSEN=$(echo "$EXPORTS" | tr ' ' '\n' | grep '^QDRANT_PORT=' | cut -d= -f2)

echo "Waiting for the API to become healthy (the first start loads the models)..."
for i in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$API_PORT_CHOSEN/health" >/dev/null 2>&1; then
    green "API is healthy."
    echo
    echo "  UI          http://localhost:$FRONTEND_PORT_CHOSEN"
    echo "  API docs    http://localhost:$API_PORT_CHOSEN/docs"
    echo "  Qdrant UI   http://localhost:$QDRANT_PORT_CHOSEN/dashboard"
    if [ "$DEV_AUTH" = 1 ]; then
      echo
      echo "  Mint a dev token and paste it into the UI's 'Access token' box:"
      echo "    docker compose exec api python scripts/make_dev_token.py --tenant dev \\"
      echo "      --groups finance --clearance 3 --scopes rag:query,rag:ingest,rag:web,rag:admin"
    fi
    echo
    echo "  Logs   docker compose logs -f api      Stop   docker compose down"
    exit 0
  fi
  sleep 2
done
red "The API did not become healthy in time. Inspect:  docker compose logs api"
exit 1

"""scripts/docker-up.sh: the port check, run against a fake `docker` and `lsof`.

The shims let every scenario be staged exactly (who holds a port, what compose publishes) while
the real script runs unmodified. Nothing here starts a container.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "docker-up.sh"

# Ports well away from anything real, so the script's own connect test sees them as free.
PORTS = {"API_PORT": "47001", "FRONTEND_PORT": "47002", "QDRANT_PORT": "47003",
         "QDRANT_GRPC_PORT": "47004", "REDIS_PORT": "47005", "KAFKA_PORT": "47006", "ZOOKEEPER_PORT": "47007"}

DOCKER_SHIM = textwrap.dedent("""\
    #!/bin/bash
    echo "$*" >> "$FAKE_LOG"
    case "$1" in
      info) [ -n "$FAKE_DOCKER_DOWN" ] && exit 1; exit 0 ;;
      ps)   if echo "$*" | grep -q -- "--filter"; then printf '%s\\n' "$FAKE_OURS"; else printf '%s\\n' "$FAKE_ALL"; fi ;;
      compose)
        if [ "$2" = "up" ] && [ -n "$FAKE_COMPOSE_FAIL" ]; then exit 1; fi
        if [ "$2" = "port" ]; then
          var="FAKE_PORT_${3}_${4}"
          [ -n "${!var}" ] && { echo "${!var}"; exit 0; }
          exit 1
        fi
        exit 0 ;;
    esac
""")

LSOF_SHIM = textwrap.dedent("""\
    #!/bin/bash
    port=""
    for arg in "$@"; do case "$arg" in -iTCP:*) port="${arg#-iTCP:}" ;; esac; done
    for busy in $FAKE_BUSY; do
      if [ "$busy" = "$port" ]; then printf 'COMMAND PID\\nfakeproc 4242\\n'; exit 0; fi
    done
    exit 1
""")


@pytest.fixture
def run(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", DOCKER_SHIM), ("lsof", LSOF_SHIM)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "docker.log"

    def invoke(*args, busy=(), ours="", all_containers="", ports=None, docker_down=False, current=None,
               compose_fail=False, extra_env=None):
        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "FAKE_LOG": str(log),
            "FAKE_BUSY": " ".join(map(str, busy)), "FAKE_OURS": ours, "FAKE_ALL": all_containers,
            **(PORTS if ports is None else ports),
        }
        if docker_down:
            env["FAKE_DOCKER_DOWN"] = "1"
        if compose_fail:
            env["FAKE_COMPOSE_FAIL"] = "1"
        env.update(extra_env or {})
        for (service, cport), value in (current or {}).items():
            env[f"FAKE_PORT_{service}_{cport}"] = value
        result = subprocess.run(["/bin/bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60)
        return SimpleResult(result, log.read_text() if log.exists() else "")

    return invoke


class SimpleResult:
    def __init__(self, proc, docker_log):
        self.code, self.out, self.err, self.docker_log = proc.returncode, proc.stdout, proc.stderr, docker_log
        self.text = proc.stdout + proc.stderr

    def row(self, variable: str) -> str:
        return next(line for line in self.out.splitlines() if variable in line and "VARIABLE" not in line)


class TestPortCheck:
    def test_all_free(self, run):
        result = run("--check-only")
        assert result.code == 0 and "Ports OK" in result.out
        assert all("free" in result.row(v) for v in ("API_PORT", "FRONTEND_PORT", "QDRANT_PORT", "REDIS_PORT"))
        assert "API_PORT=47001 FRONTEND_PORT=47002 QDRANT_PORT=47003 QDRANT_GRPC_PORT=47004 REDIS_PORT=47005" in result.out

    def test_a_busy_port_is_replaced_and_its_holder_is_named(self, run):
        result = run("--check-only", busy=[47001], all_containers="job-pipeline-worker-1|127.0.0.1:47001->8000/tcp")
        row = result.row("API_PORT")
        assert "busy (docker container 'job-pipeline-worker-1')" in row and "using" in row
        assert "API_PORT=47001" not in result.out and result.code == 0

    def test_a_local_process_is_named_when_no_container_holds_the_port(self, run):
        row = run("--check-only", busy=[47005]).row("REDIS_PORT")
        assert "process fakeproc (pid 4242)" in row

    def test_strict_mode_fails_instead_of_moving(self, run):
        result = run("--check-only", "--strict", busy=[47001])
        assert result.code == 1 and "BUSY" in result.out and "1 port conflict" in result.text
        assert "Would start" not in result.out

    def test_docker_port_ranges_are_understood_for_this_stacks_own_containers(self, run):
        # Regression: docker prints qdrant's two ports as one range, and parsing only "NNNN->"
        # lost the first port, so the stack reported its own container as a conflict.
        ours = "0.0.0.0:47003-47004->47003-47004/tcp, [::]:47003-47004->47003-47004/tcp, 0.0.0.0:47005->6379/tcp"
        result = run("--check-only", "--strict", busy=[47003, 47004, 47005], ours=ours)
        assert result.code == 0, result.text
        for variable in ("QDRANT_PORT", "QDRANT_GRPC_PORT", "REDIS_PORT"):
            assert "in use by this stack" in result.row(variable)
        assert "BUSY" not in result.out

    def test_the_owner_of_a_busy_port_is_found_when_docker_shows_a_range(self, run):
        other = "otherproj-qdrant-1|0.0.0.0:47003-47004->6333-6334/tcp, [::]:47003-47004->6333-6334/tcp"
        assert "docker container 'otherproj-qdrant-1'" in run("--check-only", busy=[47003], all_containers=other).row("QDRANT_PORT")
        assert "docker container 'otherproj-qdrant-1'" in run("--check-only", busy=[47004], all_containers=other).row("QDRANT_GRPC_PORT")

    def test_a_restart_keeps_the_port_this_service_already_uses(self, run):
        ours = "0.0.0.0:47099->8000/tcp"
        result = run("--check-only", busy=[47001, 47099], ours=ours, current={("api", "8000"): "0.0.0.0:47099"})
        assert "keeping this stack's current port 47099" in result.row("API_PORT")
        assert "API_PORT=47099" in result.out

    def test_a_fallback_never_collides_with_a_later_services_wanted_port(self, run):
        # API (47001) is busy; its first free neighbour is 47002, which is also the UI's wanted port.
        result = run("--check-only", busy=[47001])
        chosen = [p.split("=")[1] for p in result.out.split("Would start with:")[1].split()]
        assert len(chosen) == len(set(chosen)), f"duplicate ports handed out: {chosen}"
        assert "assigned to another service" in result.row("FRONTEND_PORT")

    def test_chains_of_busy_ports_still_yield_distinct_ports(self, run):
        result = run("--check-only", busy=[47001, 47002, 47003])
        chosen = [p.split("=")[1] for p in result.out.split("Would start with:")[1].split()]
        assert len(chosen) == len(set(chosen)) == 5

    def test_the_kafka_profile_adds_its_two_ports(self, run):
        result = run("--check-only", "--kafka")
        assert "47006" in result.row("KAFKA_PORT") and "47007" in result.row("ZOOKEEPER_PORT")
        assert "KAFKA_PORT=47006" in result.out
        assert "KAFKA_PORT" not in run("--check-only").out


class TestSafety:
    def test_check_only_never_starts_anything(self, run):
        result = run("--check-only", busy=[47001])
        compose_calls = [l for l in result.docker_log.splitlines() if l.startswith("compose")]
        assert all(l.startswith("compose port ") for l in compose_calls)   # read-only lookups only
        assert not any(" up" in l or "build" in l for l in compose_calls)

    def test_a_non_numeric_port_is_rejected(self, run):
        result = run("--check-only", ports={**PORTS, "API_PORT": "80; echo pwned"})
        assert result.code == 2 and "must be a number" in result.text and "pwned" not in result.out.replace("echo pwned", "")

    def test_unknown_options_are_rejected(self, run):
        result = run("--nope")
        assert result.code == 2 and "Unknown option" in result.text

    def test_a_stopped_docker_daemon_is_reported_clearly(self, run):
        result = run("--check-only", docker_down=True)
        assert result.code == 1 and "daemon is not running" in result.text

    def test_help_prints_usage_without_touching_docker(self, run):
        result = run("--help")
        assert result.code == 0 and "--check-only" in result.out and result.docker_log == ""

    def test_it_never_sources_the_env_file(self):
        text = SCRIPT.read_text()
        assert "source .env" not in text and ". .env" not in text   # .env holds secrets; it is parsed, not executed


class TestLlmKeyNote:
    """Uploading works without a model key but questions do not; say so before the user finds out."""

    def start(self, run, **env):
        return run(ports=PORTS, compose_fail=True, extra_env=env)   # `compose up` fails fast: nothing starts

    def test_a_missing_key_produces_a_clear_note(self, run, monkeypatch):
        for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY", "MISTRAL_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        env_file = SCRIPT.parent.parent / ".env"
        existed = env_file.exists()
        original = env_file.read_text() if existed else None
        try:
            env_file.write_text("AUTH_MODE=dev_token\nGROQ_API_KEY=\nOPENAI_API_KEY=\n")
            result = self.start(run)
            assert "no LLM API key is set" in result.text and "503" in result.text
            with_key = self.start(run, GROQ_API_KEY="gsk_test")
            assert "no LLM API key" not in with_key.text
            env_file.write_text("GEMINI_API_KEY=abc123\n")                     # a key in .env counts too
            assert "no LLM API key" not in self.start(run).text
        finally:
            if existed:
                env_file.write_text(original)
            else:
                env_file.unlink()

    def test_the_note_never_blocks_the_start(self, run):
        result = self.start(run)
        assert "docker compose failed" in result.text        # reached the (stubbed) start: the note is advice only

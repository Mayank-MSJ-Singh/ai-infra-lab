# ──────────────────────────────────────────────────────────────
# Simulation Configuration
# ──────────────────────────────────────────────────────────────
# Edit these values to control the simulation behaviour.
# No need to touch any other file.
# ──────────────────────────────────────────────────────────────

# ── Server ───────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"

# ── Load ─────────────────────────────────────────────────────
TOTAL_USERS = 100           # Total simulated users
MAX_CONCURRENCY = 50        # Max simultaneous in-flight requests

# ── Timeouts (seconds) ──────────────────────────────────────
REQUEST_TIMEOUT = 300.0     # Per-request timeout
CONNECT_TIMEOUT = 30.0      # TCP connect timeout

# ── Behaviour ────────────────────────────────────────────────
DELAY_BETWEEN_MSGS = 0.5    # Seconds between msg1 and msg2 per user

# ── Dashboard ────────────────────────────────────────────────
DASHBOARD_REFRESH_RATE = 4  # How many times per second the dashboard updates

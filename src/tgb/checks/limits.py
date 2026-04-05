"""Centralized field length and value limits for checks.

All check modules should import limits from here rather than
hard-coding magic numbers.
"""

# ── Reasoning / narration ───────────────────────────────────────────

REASONING_MAX_CHARS = 1200
NARRATION_MAX_CHARS = 1800
NARRATION_MIN_CHARS = 10
NARRATION_MAX_WORDS = 300
NARRATION_MIN_WORDS = 5

# ── XP ──────────────────────────────────────────────────────────────

XP_MIN = 0
XP_MAX = 10

# ── Visibility ──────────────────────────────────────────────────────

VISIBILITY_REASON_MAX_CHARS = 240

# ── SMS ─────────────────────────────────────────────────────────────

SMS_FIELD_MAX_CHARS = 80       # thread, from, to
SMS_MESSAGE_MAX_CHARS = 500
SMS_DELAY_MIN = 0
SMS_DELAY_MAX = 86400
SMS_MAX_THREADS = 24
SMS_MAX_MESSAGES_PER_THREAD = 40

# ── Subplot / planning ──────────────────────────────────────────────

SUBPLOT_DESCRIPTION_MAX_CHARS = 240

# ── Calendar ────────────────────────────────────────────────────────

CALENDAR_NAME_MAX_CHARS = 80
CALENDAR_TARGET_MAX_CHARS = 160

# ── Timers ──────────────────────────────────────────────────────────

TIMER_DELAY_MIN = 30
TIMER_DELAY_MAX = 300

# ── Inventory ──────────────────────────────────────────────────

MAX_INVENTORY_CHANGES_PER_TURN = 10

# ── Autobiography ──────────────────────────────────────────────

AUTOBIOGRAPHY_FIELD_MAX_CHARS = 600       # a, b, c, text fields
AUTOBIOGRAPHY_TRIGGER_MAX_CHARS = 80
AUTOBIOGRAPHY_IMPORTANCE_MAX_CHARS = 40
AUTOBIOGRAPHY_MAX_ENTRIES_PER_CALL = 16

# ── Memory ─────────────────────────────────────────────────────

MEMORY_SEARCH_MAX_QUERIES = 4
MEMORY_SEARCH_CONTEXT_LINES_MAX = 50
MEMORY_STORE_MEMORY_MAX_CHARS = 1600
MEMORY_TERMS_MAX_RESULTS = 40
MEMORY_TURN_ID_MIN = 1

# ── SMS read/list ──────────────────────────────────────────────

SMS_READ_LIMIT_MIN = 1
SMS_READ_LIMIT_MAX = 40

import re
from html.parser import HTMLParser

DOCS_PAGE_URL = "https://opencode.ai/docs/go/"

CURRENCY = "USD"

PLAN_KEY_FALLBACK = ("go", "go_plus")

# Variants inherit these base fields when absent — pricing is never inherited
# (variant economics come only from variant rows).
VARIANT_INHERITED_FIELDS = (
    "endpoint",
    "ai_sdk_package",
    "model_training",
    "data_retention",
    "rate_limits",
)

# Header-signature contract: table identity is declared as a set of required
# normalized header texts per table family. Columns are looked up by header
# text — never by position.
TABLE_CONTRACTS = {
    "pricing": {
        "required": (
            "input",
            "output",
            "cached read",
            "cached write",
            "monthly limit",
        )
    },
    "rate_limits": {
        "required": (
            "requests per 5 hours",
            "requests per week",
            "requests per month",
        )
    },
    "endpoints": {"required": ("model id", "endpoint", "ai sdk package")},
    "policies": {"required": ("model training", "data retention")},
}

# Verbatim header text → logical field. The row-name column is "model".
_COLUMN_FIELDS = {
    "model": "model",
    "input": "input",
    "output": "output",
    "cached read": "cached_read",
    "cached write": "cached_write",
    "monthly limit": "monthly_limit",
    "requests per 5 hours": "per_5_hours",
    "requests per week": "per_week",
    "requests per month": "per_month",
    "model id": "model_id",
    "endpoint": "endpoint",
    "ai sdk package": "ai_sdk_package",
    "model training": "model_training",
    "data retention": "data_retention",
}

# Tables on the page that are recognized but carry no model data: tolerated
# silently (the privacy facts live in the mapped policies contract).
UNMAPPED_HEADER_SHAPES = {
    frozenset(("plan", "price", "included usage")),
    frozenset(("client", "session support")),
    frozenset(("client", "status and tracking")),
}


class DocsScrapeError(Exception):
    pass


class _DocsTableParser(HTMLParser):
    """Extract DOM facts: tables, per-cell text segments and plan panels.

    Each table is emitted as::

        {"headers": [[segment, ...], ...],
         "rows": [[[segment, ...], ...], ...],
         "plan_panel": <nearest ancestor data-plan-panel value or None>}

    Cell text is kept as a list of segments so nested-element boundaries
    (e.g. ``<td><strong>$60</strong><br/><small>limited time</small></td>``)
    survive; the parser, not the extractor, decides how to join them.
    """

    def __init__(self):
        super().__init__()
        self.tables = []
        self._table_stack = []
        self._row = None
        self._cell = None
        self._in_headers = False
        self._panel_stack = []

    def _current_panel(self):
        return self._panel_stack[-1][1] if self._panel_stack else None

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "data-plan-panel":
                self._panel_stack.append((tag, value))
        if tag == "table":
            self._table_stack.append(
                {"headers": [], "rows": [], "plan_panel": self._current_panel()}
            )
        elif tag == "thead":
            self._in_headers = True
        elif tag == "tbody":
            self._in_headers = False
        elif tag == "tr" and self._table_stack and self._row is None:
            self._row = []
        elif tag in ("th", "td") and self._row is not None and self._cell is None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "table":
            if self._table_stack:
                self.tables.append(self._table_stack.pop())
        elif tag == "thead":
            self._in_headers = False
        elif tag == "tr":
            if self._row is not None and self._table_stack:
                table = self._table_stack[-1]
                if self._in_headers:
                    table["headers"].extend(self._row)
                else:
                    table["rows"].append(self._row)
            self._row = None
        elif tag in ("th", "td") and self._cell is not None:
            if self._row is not None:
                self._row.append(self._cell)
            self._cell = None
        if self._panel_stack and self._panel_stack[-1][0] == tag:
            self._panel_stack.pop()

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_html_tables(html):
    """Parse HTML into tables with per-cell text segments and plan panels."""
    parser = _DocsTableParser()
    parser.feed(html)
    parser.close()
    return parser.tables


def _join_cell(cell):
    """Join a cell's text segments the way a browser textContent would."""
    return re.sub(r"\s+", " ", "".join(cell).replace("\xa0", " ")).strip()


def _normalize_name(name):
    """Normalize a display name (or header) for matching."""
    return re.sub(r"[\s_-]+", " ", name.lower().strip())


def _split_promo(cell):
    """Split a name cell into (name, promo) at a ``·`` segment boundary."""
    for index, segment in enumerate(cell):
        if "·" in segment:
            return _join_cell(cell[:index]), _join_cell(cell[index:]) or None
    return _join_cell(cell), None


_VARIANT_RE = re.compile(r"^(.*?)\s*\(([^()]*)\)$")


def _split_variant(name):
    """Split a tier-suffixed display name into (base name, variant label)."""
    match = _VARIANT_RE.match(name)
    if not match:
        return name, None
    base, label = match.group(1).strip(), match.group(2).strip()
    if not base or not label:
        return name, None
    return base, label


def _slug(label):
    """Slugify a variant label, keeping bucket distinctions unique."""
    text = label.strip().lower()
    for char, word in (("≤", "le"), ("≥", "ge"), (">", "gt"), ("<", "lt")):
        text = text.replace(char, f"{word} ")
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return slug or "variant"


_PRICE_TEXT_RE = re.compile(r"^\$?[0-9]+(?:[.,][0-9]+)*\.?$")


def _parse_price(text):
    if not text or text.strip() == "-":
        return None
    cleaned = re.sub(r"[^\d.]", "", text)
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _parse_price_cell(cell):
    """Parse a pricing cell into (value, note, is_free).

    Segments are the DOM facts: the last numeric segment wins (a superseded
    ``<del>`` value never does), and segments after it become a note (e.g.
    ``limited time`` markup under a ``<small>``). ``Free`` parses to ``0.0``
    with the free flag set so the parser can preserve provenance.
    """
    parts = [segment.strip() for segment in cell if segment.strip()]
    if not parts:
        return None, None, False
    free_index = next(
        (i for i, part in enumerate(parts) if part.lower() == "free"), None
    )
    if free_index is not None:
        note = _join_note(parts[free_index + 1 :])
        return 0.0, note, True
    numeric = [i for i, part in enumerate(parts) if _PRICE_TEXT_RE.match(part)]
    if numeric:
        last = numeric[-1]
        note = _join_note(parts[last + 1 :])
        return _parse_price(parts[last]), note, False
    if all(part == "-" for part in parts):
        return None, None, False
    return None, _join_note(parts), False


def _join_note(parts):
    if not parts:
        return None
    return " ".join(parts).strip() or None


def _parse_rate_cell(cell):
    """Parse a rate-limit cell into a string; the newest numeric segment wins."""
    parts = [segment.strip() for segment in cell if segment.strip()]
    if not parts:
        return ""
    numeric = [i for i, part in enumerate(parts) if _PRICE_TEXT_RE.match(part)]
    if numeric:
        return parts[numeric[-1]].replace(",", "")
    return parts[0]


def _plan_key(panel, family_count):
    """Plan key from the data-plan-panel attr, falling back to table order
    (first pricing/rates table = Go, second = Go Plus)."""
    if panel:
        return re.sub(r"[^a-z0-9]+", "_", panel.strip().lower())
    return PLAN_KEY_FALLBACK[family_count % len(PLAN_KEY_FALLBACK)]


def _identify_table(header_set):
    """Match a header set against the declared contract.

    Returns (family, warning): family when a contract's required headers are
    all present; otherwise the closest-family drift warning, or silence for
    recognized-but-unmapped shapes. Extra columns beyond the contract (plus
    the common ``model`` name column) are genuine header drift: they warn.
    """
    for family, contract in TABLE_CONTRACTS.items():
        required = set(contract["required"])
        if required <= header_set:
            extras = sorted(header_set - required - {"model"})
            if extras:
                return family, f"{family}: extra column(s) {extras}"
            return family, None
    best_family, best_overlap = None, 0
    for family, contract in TABLE_CONTRACTS.items():
        overlap = len(header_set & set(contract["required"]))
        if overlap > best_overlap:
            best_family, best_overlap = family, overlap
    if best_family:
        missing = sorted(set(TABLE_CONTRACTS[best_family]["required"]) - header_set)
        return None, (
            f"{best_family}: header drift (missing {', '.join(missing)}); "
            f"found {sorted(header_set)}"
        )
    if header_set in UNMAPPED_HEADER_SHAPES:
        return None, None
    return None, f"unrecognized table shape: {sorted(header_set)}"


def _field_indexes(headers):
    """Map logical fields to column indexes by header text, never position."""
    columns = {}
    for index, header in enumerate(headers):
        columns.setdefault(_normalize_name(_join_cell(header)), index)
    indexes = {}
    for header, field in _COLUMN_FIELDS.items():
        if _normalize_name(header) in columns:
            indexes[field] = columns[_normalize_name(header)]
    return indexes


def _cell(row, index):
    if index is not None and index < len(row):
        return row[index]
    return []


def _variant_key(name, base_name, label, base_id):
    if label:
        base_token = base_id if base_id else name
        return f"{base_token}#{_slug(label)}"
    return base_id if base_id else name


def _variant_entry(docs_data, key, model_name, label, matched):
    entry = docs_data.get(key)
    if entry is None:
        entry = {"model_id": key, "model_name": model_name, "matched": matched}
        docs_data[key] = entry
    if matched:
        entry["matched"] = True
    if label:
        entry["variant"] = label
    return entry


def parse_docs_tables(tables):
    """Parse docs page tables into model data plus scrape warnings.

    Returns ``(docs_data, warnings)``. Entries are keyed by Model ID, or
    ``base_id#variant`` (slugified) for tier-variant rows, or the display
    name for orphan rows that never joined a Model ID. Raises
    :class:`DocsScrapeError` only when zero tables match any contract.
    """
    warnings = []
    plans_seen = {"pricing": 0, "rate_limits": 0}
    name_to_id = {}
    docs_data = {}
    matched_any = False

    identified = []
    for index, table in enumerate(tables):
        header_set = {
            _normalize_name(_join_cell(cell))
            for cell in table["headers"]
            if _join_cell(cell)
        }
        family, warning = _identify_table(header_set)
        if warning:
            warnings.append(f"table {index}: {warning}")
        if family:
            matched_any = True
        identified.append((family, table, _field_indexes(table["headers"])))

    if not matched_any:
        raise DocsScrapeError(
            "No tables on the docs page matched any known header contract"
        )

    def row_facts(indexes, row):
        name, promo = _split_promo(_cell(row, indexes.get("model")))
        base_name, label = _split_variant(name)
        base_id = (
            name_to_id.get(_normalize_name(base_name if label else name))
            if name
            else None
        )
        return name, base_name, label, promo, base_id

    # Endpoint (Model ID) tables first: they define the display-name → ID join.
    for family, table, indexes in identified:
        if family != "endpoints":
            continue
        for row in table["rows"]:
            name, base_name, label, promo, _ = row_facts(indexes, row)
            model_id = _join_cell(_cell(row, indexes.get("model_id")))
            if not model_id:
                continue
            effective_name = base_name if label else name
            name_to_id[_normalize_name(name)] = model_id
            if base_name != name:
                name_to_id[_normalize_name(base_name)] = model_id
            entry = docs_data.setdefault(
                model_id,
                {"model_id": model_id, "model_name": effective_name, "matched": True},
            )
            entry["endpoint"] = _join_cell(_cell(row, indexes.get("endpoint")))
            entry["ai_sdk_package"] = _join_cell(
                _cell(row, indexes.get("ai_sdk_package"))
            )
            if promo:
                entry["promo"] = promo

    # Pricing tables: plan-keyed (data-plan-panel attr, fallback to order).
    for family, table, indexes in identified:
        if family != "pricing":
            continue
        plan = _plan_key(table.get("plan_panel"), plans_seen["pricing"])
        plans_seen["pricing"] += 1
        for row in table["rows"]:
            name, base_name, label, promo, base_id = row_facts(indexes, row)
            if not name:
                continue
            plan_pricing = {}
            free_in_row = False
            for field in ("input", "output", "cached_read", "cached_write"):
                value, _note, is_free = _parse_price_cell(
                    _cell(row, indexes.get(field))
                )
                plan_pricing[field] = value
                free_in_row = free_in_row or is_free
            monthly_limit, limit_note, _ = _parse_price_cell(
                _cell(row, indexes.get("monthly_limit"))
            )
            plan_pricing["monthly_limit"] = monthly_limit
            plan_pricing["limit_note"] = limit_note
            plan_pricing["currency"] = CURRENCY
            if free_in_row:
                plan_pricing["pricing_note"] = "free"

            key = _variant_key(name, base_name, label, base_id)
            entry = _variant_entry(
                docs_data, key, base_name if label else name, label, bool(base_id)
            )
            entry.setdefault("pricing", {})[plan] = plan_pricing
            if promo:
                entry["promo"] = promo

    # Rate-limit tables: plan-keyed like pricing.
    for family, table, indexes in identified:
        if family != "rate_limits":
            continue
        plan = _plan_key(table.get("plan_panel"), plans_seen["rate_limits"])
        plans_seen["rate_limits"] += 1
        for row in table["rows"]:
            name, base_name, label, promo, base_id = row_facts(indexes, row)
            if not name:
                continue
            plan_rates = {
                field: _parse_rate_cell(_cell(row, indexes.get(field)))
                for field in ("per_5_hours", "per_week", "per_month")
            }
            key = _variant_key(name, base_name, label, base_id)
            entry = _variant_entry(
                docs_data, key, base_name if label else name, label, bool(base_id)
            )
            entry.setdefault("rate_limits", {})[plan] = plan_rates
            if promo:
                entry["promo"] = promo

    # Policy tables.
    for family, table, indexes in identified:
        if family != "policies":
            continue
        for row in table["rows"]:
            name, base_name, label, promo, base_id = row_facts(indexes, row)
            if not name:
                continue
            key = _variant_key(name, base_name, label, base_id)
            entry = _variant_entry(
                docs_data, key, base_name if label else name, label, bool(base_id)
            )
            entry["model_training"] = _join_cell(
                _cell(row, indexes.get("model_training"))
            )
            entry["data_retention"] = _join_cell(
                _cell(row, indexes.get("data_retention"))
            )

    _inherit_variant_fields(docs_data)
    return docs_data, warnings


def _inherit_variant_fields(docs_data):
    for key, entry in docs_data.items():
        if "#" not in key:
            continue
        base = docs_data.get(key.rsplit("#", 1)[0])
        if not base or base is entry:
            continue
        for field in VARIANT_INHERITED_FIELDS:
            if field not in entry and field in base:
                value = base[field]
                entry[field] = dict(value) if isinstance(value, dict) else value


def fetch_docs_page():
    """Fetch the OpenCode Go docs page HTML."""
    try:
        import httpx2

        response = httpx2.get(DOCS_PAGE_URL, follow_redirects=True)
        response.raise_for_status()
        return response.text
    except httpx2.HTTPError as error:
        raise DocsScrapeError(f"Failed to fetch docs page: {error}") from error

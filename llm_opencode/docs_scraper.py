import re
from html.parser import HTMLParser

DOCS_PAGE_URL = "https://opencode.ai/docs/go/"


class DocsScrapeError(Exception):
    pass


class _DocsTableParser(HTMLParser):
    """Parse HTML into a list of tables, each with ``headers`` and ``rows``."""

    def __init__(self):
        super().__init__()
        self.tables = []
        self._table_stack = []
        self._row = None
        self._cell = None
        self._in_headers = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._table_stack.append({"headers": [], "rows": []})
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
                self._row.append("".join(self._cell).strip())
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_html_tables(html):
    """Parse HTML into a list of tables with ``headers`` and ``rows``."""
    parser = _DocsTableParser()
    parser.feed(html)
    parser.close()
    return parser.tables


def fetch_docs_page():
    """Fetch the OpenCode Go docs page HTML."""
    try:
        import httpx2

        response = httpx2.get(DOCS_PAGE_URL, follow_redirects=True)
        response.raise_for_status()
        return response.text
    except httpx2.HTTPError as error:
        raise DocsScrapeError(f"Failed to fetch docs page: {error}") from error


def _normalize_name(name):
    """Normalize a model display name for matching against Model ID rows."""
    return re.sub(r"[\s_-]+", " ", name.lower().strip())


def _parse_price(value):
    """Parse '$2.00' or '15' into a float, or None for '-' or empty."""
    if not value or value == "-":
        return None
    cleaned = re.sub(r"[^\d.]", "", value)
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def parse_docs_tables(tables):
    """Parse the docs page tables into model data keyed by model ID.

    Tables are identified by their headers, not their position:
    - a table with a "Model ID" column maps display names to model IDs
    - a table with "Input"/"Cached Read" columns holds pricing
    - a table with "requests per" columns holds rate limits
    - a table with "Model training"/"Data retention" columns holds policies
    """
    docs_data = {}
    name_to_id = {}

    for table in tables:
        headers = [h.lower() for h in table["headers"]]
        if "model id" in headers:
            for row in table["rows"]:
                if len(row) >= 2:
                    name_to_id[_normalize_name(row[0])] = row[1]

    for table in tables:
        headers = [h.lower() for h in table["headers"]]
        for row in table["rows"]:
            if not row:
                continue
            model_id = name_to_id.get(_normalize_name(row[0]), row[0])
            entry = docs_data.setdefault(model_id, {})
            entry["model_name"] = row[0]
            if "model id" in headers:
                if len(row) >= 4:
                    entry["endpoint"] = row[2]
                    entry["ai_sdk_package"] = row[3]
            elif "cached read" in headers:
                if len(row) >= 6:
                    entry["pricing"] = {
                        "input": _parse_price(row[1]),
                        "output": _parse_price(row[2]),
                        "cached_read": _parse_price(row[3]),
                        "cached_write": _parse_price(row[4]),
                        "usage": _parse_price(row[5]),
                        "currency": "USD",
                    }
            elif any("requests per" in header for header in headers):
                if len(row) >= 4:
                    entry["rate_limits"] = {
                        "per_5_hours": row[1].replace(",", ""),
                        "per_week": row[2].replace(",", ""),
                        "per_month": row[3].replace(",", ""),
                    }
            elif (
                "model training" in headers or "data retention" in headers
            ) and len(row) >= 3:
                entry["model_training"] = row[1]
                entry["data_retention"] = row[2]

    for model_id, entry in list(docs_data.items()):
        variant_name = entry.get("model_name", "")
        if "(" in variant_name:
            base_name = variant_name.split("(")[0].strip()
            base_id = name_to_id.get(_normalize_name(base_name))
            base_entry = docs_data.get(base_id) if base_id else None
            if base_entry and base_id != model_id:
                for field in (
                    "endpoint",
                    "ai_sdk_package",
                    "model_training",
                    "data_retention",
                    "rate_limits",
                ):
                    if field not in entry and field in base_entry:
                        entry[field] = base_entry[field]

    return docs_data


from llm_opencode import parse_html_tables


def test_parse_html_tables_extracts_headers_and_rows():
    html = """
    <table>
      <thead>
        <tr><th>Model</th><th>Input</th><th>Output</th></tr>
      </thead>
      <tbody>
        <tr><td>Grok 4.5</td><td>$2.00</td><td>$6.00</td></tr>
        <tr><td>GPT 5.6 Luna</td><td>$0.20</td><td>$1.20</td></tr>
      </tbody>
    </table>
    """
    tables = parse_html_tables(html)
    assert tables == [
        {
            "headers": ["Model", "Input", "Output"],
            "rows": [
                ["Grok 4.5", "$2.00", "$6.00"],
                ["GPT 5.6 Luna", "$0.20", "$1.20"],
            ],
        }
    ]


def test_parse_html_tables_multiple_tables():
    html = """
    <table><thead><tr><th>A</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>
    <table><thead><tr><th>B</th></tr></thead><tbody><tr><td>2</td></tr></tbody></table>
    """
    tables = parse_html_tables(html)
    assert len(tables) == 2
    assert tables[0]["headers"] == ["A"]
    assert tables[1]["headers"] == ["B"]


def test_parse_html_tables_strips_nested_tags_in_cells():
    html = """
    <table>
      <thead><tr><th>Model</th></tr></thead>
      <tbody>
        <tr><td>Grok <b>4.5</b></td></tr>
        <tr><td><a href="https://opencode.ai">GPT 5.6</a></td></tr>
      </tbody>
    </table>
    """
    tables = parse_html_tables(html)
    assert tables[0]["rows"] == [["Grok 4.5"], ["GPT 5.6"]]


def test_parse_html_tables_decodes_html_entities():
    html = """
    <table>
      <thead><tr><th>Model &amp; Family</th></tr></thead>
      <tbody><tr><td>Kimi&nbsp;K2.5</td></tr></tbody>
    </table>
    """
    tables = parse_html_tables(html)
    assert tables[0]["headers"] == ["Model & Family"]
    assert tables[0]["rows"] == [["Kimi\u00a0K2.5"]]


def test_parse_html_tables_no_tables():
    assert parse_html_tables("<html><body><p>no tables here</p></body></html>") == []


def test_parse_html_tables_rows_without_thead():
    html = """
    <table>
      <tbody>
        <tr><td>one</td><td>two</td></tr>
      </tbody>
    </table>
    """
    tables = parse_html_tables(html)
    assert tables[0]["headers"] == []
    assert tables[0]["rows"] == [["one", "two"]]


def test_parse_html_tables_skips_empty_cells_and_nested_tables():
    html = """
    <table>
      <thead><tr><th>Model</th></tr></thead>
      <tbody>
        <tr><td>Grok 4.5</td><td>extra</td></tr>
      </tbody>
      <tbody>
        <tr><td>GPT 5.6</td></tr>
      </tbody>
    </table>
    """
    tables = parse_html_tables(html)
    assert tables[0]["rows"] == [["Grok 4.5", "extra"], ["GPT 5.6"]]


def test_parse_html_tables_tolerates_malformed_html():
    html = """
    </table>
    <tr><td>stray</td></tr>
    <td>stray cell</td>
    <table>
      <tr>
        <td>ok</td>
      </tr>
      <tr><td>x</tr></td>
    </table>
    </body>
    """
    tables = parse_html_tables(html)
    assert len(tables) == 1
    assert tables[0]["headers"] == []
    assert tables[0]["rows"] == [["ok"], []]

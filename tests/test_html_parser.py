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
            "headers": [["Model"], ["Input"], ["Output"]],
            "rows": [
                [["Grok 4.5"], ["$2.00"], ["$6.00"]],
                [["GPT 5.6 Luna"], ["$0.20"], ["$1.20"]],
            ],
            "plan_panel": None,
        }
    ]


def test_parse_html_tables_multiple_tables():
    html = """
    <table><thead><tr><th>A</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>
    <table><thead><tr><th>B</th></tr></thead><tbody><tr><td>2</td></tr></tbody></table>
    """
    tables = parse_html_tables(html)
    assert len(tables) == 2
    assert tables[0]["headers"] == [["A"]]
    assert tables[1]["headers"] == [["B"]]


def test_parse_html_tables_preserves_nested_element_boundaries():
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
    # Segments, not flattened strings: nested-element boundaries survive.
    assert tables[0]["rows"] == [
        [["Grok ", "4.5"]],
        [["GPT 5.6"]],
    ]


def test_parse_html_tables_split_cells_across_br():
    html = """
    <table>
      <thead><tr><th>Monthly limit</th></tr></thead>
      <tbody><tr><td><strong>$60</strong><br/><small>limited time</small></td></tr></tbody>
    </table>
    """
    tables = parse_html_tables(html)
    assert tables[0]["rows"] == [[["$60", "limited time"]]]


def test_parse_html_tables_decodes_html_entities():
    html = """
    <table>
      <thead><tr><th>Model &amp; Family</th><th>Range</th></tr></thead>
      <tbody><tr><td>Kimi&nbsp;K2.5</td><td>&gt; 100K tokens</td></tr></tbody>
    </table>
    """
    tables = parse_html_tables(html)
    assert tables[0]["headers"] == [["Model & Family"], ["Range"]]
    assert tables[0]["rows"] == [[["Kimi\u00a0K2.5"], ["> 100K tokens"]]]


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
    assert tables[0]["rows"] == [[["one"], ["two"]]]
    assert tables[0]["plan_panel"] is None


def test_parse_html_tables_skips_empty_headers_rows_without_cells():
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
    assert tables[0]["headers"] == [["Model"]]
    assert tables[0]["rows"] == [[["Grok 4.5"], ["extra"]], [["GPT 5.6"]]]


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
    assert tables[0]["rows"] == [[["ok"]], []]


def test_parse_html_tables_captures_plan_panel():
    html = """
    <div data-plan-tabs data-sync-key="go-plan">
      <button role="tab" data-plan-tab="go">Go</button>
      <div role="tabpanel" data-plan-panel="go"><table><tr><td>go row</td></tr></table></div>
      <div role="tabpanel" hidden data-plan-panel="go-plus"><table><tr><td>plus row</td></tr></table></div>
    </div>
    """
    tables = parse_html_tables(html)
    assert len(tables) == 2
    assert tables[0]["plan_panel"] == "go"
    assert tables[1]["plan_panel"] == "go-plus"
    assert tables[0]["rows"] == [[["go row"]]]
    assert tables[1]["rows"] == [[["plus row"]]]


def test_parse_html_tables_unwrapped_table_has_no_panel():
    html = '<div><table><tr><td>x</td></tr></table></div>'
    tables = parse_html_tables(html)
    assert tables[0]["plan_panel"] is None


def test_parse_html_tables_nested_tables_share_outer_panel():
    html = """
    <div data-plan-panel="go">
      <table><tbody><tr><td>outer
        <table><tbody><tr><td>inner</td></tr></tbody></table>
      </td></tr></tbody></table>
    </div>
    """
    tables = parse_html_tables(html)
    assert len(tables) == 2
    assert tables[0]["plan_panel"] == "go"
    assert tables[1]["plan_panel"] == "go"

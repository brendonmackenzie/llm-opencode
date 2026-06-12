import os
import pytest


def _filter_headers(headers_to_redact):
    def before_record_request(request):
        for key in request.headers:
            if key.lower() in headers_to_redact:
                request.headers[key] = "REDACTED"
        return request
    return before_record_request


@pytest.fixture(scope="session")
def vcr_config():
    return {
        "before_record_request": _filter_headers(["authorization", "x-api-key"]),
    }
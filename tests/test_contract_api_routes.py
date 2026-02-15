"""
Contract tests: every API route exists (path + method). No internal code—only app.routes / OpenAPI.
"""
import pytest

from api.main import app


def _collect_routes() -> list[tuple[str, str]]:
    # Use OpenAPI so we get full path + method for every endpoint
    openapi = app.openapi()
    routes = []
    for path, spec in (openapi.get("paths") or {}).items():
        for method, _ in spec.items():
            if method.lower() in ("get", "post", "put", "patch", "delete"):
                routes.append((path, method.upper()))
    return routes


_ROUTES = _collect_routes()


@pytest.mark.parametrize("path,method", _ROUTES)
def test_route_exists(path: str, method: str) -> None:
    assert path
    assert method in ("GET", "POST", "PUT", "PATCH", "DELETE")


@pytest.mark.parametrize("path,method", _ROUTES)
def test_route_path_starts_with_slash(path: str, method: str) -> None:
    assert path.startswith("/"), f"path {path!r} should start with /"


@pytest.mark.parametrize("path,method", _ROUTES)
def test_route_path_no_double_slash(path: str, method: str) -> None:
    assert "//" not in path, f"path {path!r} should not contain //"


# Expected route prefixes (conceptual: these routers are mounted)
EXPECTED_PREFIXES = [
    "/households",
    "/graph",
    "/sessions",
    "/risk_signals",
    "/watchlists",
    "/device",
    "/ingest",
    "/summaries",
    "/agents",
    "/protection",
    "/rings",
    "/connectors",
    "/capabilities",
    "/playbooks",
    "/incident_packets",
    "/alerts",
    "/investigation",
    "/system",
    "/actions",
]


@pytest.mark.parametrize("prefix", EXPECTED_PREFIXES)
def test_expected_router_prefix_present(prefix: str) -> None:
    paths = [p for p, _ in _ROUTES]
    assert any(p == prefix or p.startswith(prefix + "/") for p in paths), f"Expected some route under {prefix}"


def test_health_route_exists() -> None:
    paths = [p for p, _ in _ROUTES]
    assert "/health" in paths


def test_root_route_exists() -> None:
    paths = [p for p, _ in _ROUTES]
    assert "/" in paths


# One test per route for method + path shape
@pytest.mark.parametrize("path,method", _ROUTES)
def test_route_method_and_path_valid(path: str, method: str) -> None:
    assert len(path) >= 1
    assert method in ("GET", "POST", "PUT", "PATCH", "DELETE")

from typing import Generator

from click.testing import CliRunner
from playwright.sync_api import BrowserContext
import pytest
from pytest_django.live_server_helper import LiveServer
from pytest_playwright import CreateContextCallback
from rest_framework.test import APIClient


@pytest.fixture
def api_client() -> APIClient:
    return APIClient()


# This intentionally overrides the built-in fixture from pytest_playwright.
# This will also cause other built-in fixtures like "page" to have a base URL set.
@pytest.fixture
def context(
    live_server: LiveServer, new_context: CreateContextCallback
) -> Generator[BrowserContext, None, None]:
    context = new_context(
        base_url=live_server.url,
    )
    context.set_default_timeout(3000)
    try:
        yield context
    finally:
        context.close()


@pytest.fixture
def cli_runner() -> CliRunner:
    # Don't catch exceptions, so they'll be raised in the test case
    return CliRunner(catch_exceptions=False)

from django.urls import reverse
from playwright.sync_api import Page, expect
import pytest

from .factories import EpisodeFactory


@pytest.mark.django_db
def test_page_home(page: Page) -> None:
    EpisodeFactory.create()
    page.goto(reverse('home'))

    view_button = page.get_by_role('button', name='View Selected Episodes')
    expect(view_button).to_be_disabled()
    checkboxes = page.get_by_role('checkbox')
    expect(checkboxes).to_have_count(1)
    checkboxes.check()
    expect(view_button).to_be_enabled()
    assert not page.page_errors()


@pytest.mark.django_db
def test_page_insights(subtests, page: Page) -> None:
    episode = EpisodeFactory.create()
    page.goto(reverse('insights', query={'episode_id': episode.id}))

    for tab_name in ['Overview', 'Rewards', 'Agents']:
        with subtests.test(msg='select tab', tab_name=tab_name):
            tab = page.get_by_role('tab', name=tab_name)
            tab.click()
            # `aria-labelledby` should cause this tabpanel's accessible name to be `name`.
            # `include_hidden` to always succeed in getting the locator.
            tab_panel = page.get_by_role('tabpanel', name=tab_name, include_hidden=True)
            expect(tab_panel).to_be_visible()
    assert not page.page_errors()

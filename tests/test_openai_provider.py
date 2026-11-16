from types import SimpleNamespace as NS

from askrank.config import Location
from askrank.providers.openai_provider import OpenAIProvider


def _fake_client(capture):
    def create(**kw):
        capture.update(kw)
        ann = [NS(type="url_citation", url="https://a.co.za", title="A"),
               NS(type="url_citation", url="https://a.co.za", title="A"),
               NS(type="url_citation", url="https://b.co.za", title=None)]
        return NS(output_text="answer",
                  output=[NS(type="web_search_call"),
                          NS(type="message", content=[NS(annotations=ann)])])
    return NS(responses=NS(create=create))


def test_web_search_tool_and_location_sent():
    captured = {}
    p = OpenAIProvider("m", client=_fake_client(captured))
    r = p.ask("best GP?", Location(city="Brackenfell", country="ZA"))
    assert captured["tools"] == [{"type": "web_search", "user_location":
                                  {"type": "approximate", "city": "Brackenfell", "country": "ZA"}}]
    assert r.text == "answer"
    assert [c.url for c in r.citations] == ["https://a.co.za", "https://b.co.za"]


def test_web_search_can_be_disabled():
    captured = {}
    OpenAIProvider("m", web_search=False, client=_fake_client(captured)).ask("q")
    assert captured["tools"] == []

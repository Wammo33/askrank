"""Every adapter is tested with a fake client: no network, no keys, no cost."""

from types import SimpleNamespace as NS

import pytest

from askrank.config import Location, ProviderConfig
from askrank.providers import get_provider
from askrank.providers.anthropic_provider import AnthropicProvider
from askrank.providers.gemini_provider import GeminiProvider
from askrank.providers.perplexity_provider import PerplexityProvider

LOC = Location(city="Brackenfell", country="ZA")


# ---------- Anthropic ----------

def _claude_client(responses, calls):
    def create(**kw):
        calls.append(kw)
        return responses.pop(0)
    return NS(beta=NS(messages=NS(create=create)))


def _text(text, urls=()):
    cites = [NS(type="web_search_result_location", url=u, title=u) for u in urls]
    return NS(type="text", text=text, citations=cites or None)


def test_anthropic_text_citations_and_tool_config():
    calls = []
    resp = NS(stop_reason="end_turn", stop_details=None, content=[
        NS(type="server_tool_use"),
        NS(type="web_search_tool_result"),
        _text("Dr Wayne Kretschmer ", ["https://a.co.za", "https://a.co.za"]),
        _text("is well reviewed.", ["https://b.co.za"]),
    ])
    p = AnthropicProvider("claude-test", client=_claude_client([resp], calls))
    r = p.ask("best GP?", LOC)
    assert r.text == "Dr Wayne Kretschmer is well reviewed."
    assert [c.url for c in r.citations] == ["https://a.co.za", "https://b.co.za"]
    tool = calls[0]["tools"][0]
    assert tool["type"] == "web_search_20260209"
    assert tool["user_location"] == {"type": "approximate", "city": "Brackenfell",
                                     "country": "ZA"}
    assert calls[0]["fallbacks"] == "default"


def test_anthropic_continues_after_pause_turn():
    calls = []
    paused = NS(stop_reason="pause_turn", stop_details=None, content=[NS(type="server_tool_use")])
    done = NS(stop_reason="end_turn", stop_details=None, content=[_text("done")])
    r = AnthropicProvider("m", client=_claude_client([paused, done], calls)).ask("q")
    assert r.text == "done"
    assert len(calls) == 2
    assert calls[1]["messages"][-1]["role"] == "assistant"


def test_anthropic_refusal_raises():
    resp = NS(stop_reason="refusal", stop_details={"category": "x"}, content=[])
    with pytest.raises(RuntimeError, match="declined"):
        AnthropicProvider("m", client=_claude_client([resp], [])).ask("q")


# ---------- Gemini ----------

def test_gemini_rewrites_redirect_citations():
    captured = {}

    def generate_content(**kw):
        captured.update(kw)
        chunks = [
            NS(web=NS(uri="https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc",
                      title="medpages.info")),
            NS(web=NS(uri="https://example.co.za/about", title="About us")),
            NS(web=None),
        ]
        return NS(text="answer", candidates=[NS(grounding_metadata=NS(grounding_chunks=chunks))])

    p = GeminiProvider("gemini-test", client=NS(models=NS(generate_content=generate_content)))
    r = p.ask("q", LOC)
    assert r.text == "answer"
    assert [c.url for c in r.citations] == ["https://medpages.info/",
                                            "https://example.co.za/about"]
    assert captured["config"].tools[0].google_search is not None


# ---------- Perplexity ----------

def test_perplexity_search_results_and_location():
    captured = {}

    def create(**kw):
        captured.update(kw)
        return NS(choices=[NS(message=NS(content="Try Dr Kretschmer."))],
                  search_results=[{"url": "https://a.co.za", "title": "A"}],
                  citations=["https://a.co.za", "https://b.co.za"])

    p = PerplexityProvider("sonar", client=NS(chat=NS(completions=NS(create=create))))
    r = p.ask("q", LOC)
    assert r.text == "Try Dr Kretschmer."
    assert [c.url for c in r.citations] == ["https://a.co.za", "https://b.co.za"]
    assert captured["extra_body"] == {"web_search_options": {"user_location": {"country": "ZA"}}}


# ---------- factory ----------

@pytest.mark.parametrize("name,env", [("anthropic", "ANTHROPIC_API_KEY"),
                                      ("gemini", "GEMINI_API_KEY"),
                                      ("perplexity", "PERPLEXITY_API_KEY"),
                                      ("openai", "OPENAI_API_KEY")])
def test_missing_key_gives_clear_error(name, env, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
              "PERPLEXITY_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match=env):
        get_provider(ProviderConfig(name=name, model="m"))

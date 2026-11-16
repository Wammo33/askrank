from types import SimpleNamespace

from askrank.extract import LLMExtractor, RuleExtractor, match_business, normalise
from askrank.schemas import Answer, Citation, Extraction, Mention, Sentiment


def _answer(text, citations=()):
    return Answer(provider="fake", model="m", query_id="q", query_text="?", run_index=0,
                  text=text, citations=list(citations))


def test_normalise():
    assert normalise("Dr. Smith & Partners!") == "dr smith and partners"


def test_match_business(config):
    b = config.businesses
    assert match_business("Example FP", b) == "Example Family Practice"
    assert match_business("example family practice, brackenfell", b) == "Example Family Practice"
    assert match_business("Some Other Clinic", b) is None
    assert match_business("", b) is None


def test_rule_extractor_positions_and_urls(config):
    text = ("Try Competitor A first. Example Family Practice is also good "
            "(see https://example-practice.co.za/hours).")
    ex = RuleExtractor(config).extract(_answer(text, [Citation(url="https://medpages.info/x")]))
    assert [m.matched for m in ex.mentions] == ["Competitor Medical Centre A",
                                                "Example Family Practice"]
    assert [m.position for m in ex.mentions] == [1, 2]
    assert ex.cited_urls == ["https://medpages.info/x", "https://example-practice.co.za/hours"]


def test_rule_extractor_word_boundaries(config):
    # "Example FPX" must not count as "Example FP"
    ex = RuleExtractor(config).extract(_answer("Example FPX clinic is nearby."))
    assert ex.mentions == []


def test_llm_extractor_postprocesses_model_output(config):
    """The fake client returns messy output; postprocess must clean it up."""
    messy = Extraction(
        mentions=[
            Mention(name="Example FP", position=3, sentiment=Sentiment.positive),
            Mention(name="Dr Unknown Clinic", position=1, sentiment=Sentiment.neutral),
            Mention(name="Example Family Practice", position=5, sentiment=Sentiment.positive),
        ],
        cited_urls=["https://medpages.info/x"],
    )
    client = SimpleNamespace(responses=SimpleNamespace(
        parse=lambda **kw: SimpleNamespace(output_parsed=messy)))
    ex = LLMExtractor(config, model="test", client=client).extract(_answer("..."))
    assert [(m.name, m.position, m.matched) for m in ex.mentions] == [
        ("Dr Unknown Clinic", 1, None),
        ("Example FP", 2, "Example Family Practice"),  # duplicate at pos 5 dropped
    ]
    assert ex.cited_urls == ["https://medpages.info/x"]

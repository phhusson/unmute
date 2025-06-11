import pytest

from unmute.llm.llm_utils import rechunk_to_words


async def make_iterator(s: str):
    parts = s.split("|")
    for part in parts:
        yield part


@pytest.mark.asyncio
async def test_rechunk_to_words():
    test_strings = [
        "hel|lo| |w|orld",
        "hello world",
        "hello \nworld",
        "hello| |world",
        "hello| |world|.",
        "h|e|l|l|o| |\tw|o|r|l|d|.",
        "h|e|l|l|o\n| |w|o|r|l|d|.",
    ]

    for s in test_strings:
        parts = [x async for x in rechunk_to_words(make_iterator(s))]
        assert parts[0] == "hello"
        assert parts[1] == " world" or parts[1] == " world."

    async def f(s: str):
        x = [x async for x in rechunk_to_words(make_iterator(s))]
        print(x)
        return x

    assert await f("i am ok") == ["i", " am", " ok"]
    assert await f(" i am ok") == [" i", " am", " ok"]
    assert await f(" they are ok") == [" they", " are", " ok"]
    assert await f("  foo bar") == [" foo", " bar"]
    assert await f(" \t foo  bar") == [" foo", " bar"]

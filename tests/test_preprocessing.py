from src.inference import head_tail_ids


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(300))}


def test_head_tail_keeps_expected_length_and_boundaries():
    ids = head_tail_ids(FakeTokenizer(), "a review")
    assert len(ids) == 256
    assert ids[:3] == [0, 1, 2]
    assert ids[-3:] == [297, 298, 299]


def test_short_text_is_not_truncated():
    class ShortTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [101, 200, 102]}
    assert head_tail_ids(ShortTokenizer(), "short") == [101, 200, 102]

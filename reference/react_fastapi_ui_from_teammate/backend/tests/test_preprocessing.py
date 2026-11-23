from app.models.preprocessing import EncodedText, head_tail_truncate, pad_batch


def test_short_sequence_unchanged():
    ids = list(range(10))
    assert head_tail_truncate(ids) == ids


def test_exactly_max_len_unchanged():
    ids = list(range(256))
    assert head_tail_truncate(ids) == ids


def test_long_sequence_keeps_head_and_tail():
    ids = [101] + list(range(1000, 1400)) + [102]  # [CLS] ... [SEP], 402 tokens
    out = head_tail_truncate(ids, max_len=256, head_tokens=192, tail_tokens=64)
    assert len(out) == 256
    assert out[:192] == ids[:192]
    assert out[192:] == ids[-64:]
    assert out[0] == 101 and out[-1] == 102


def test_one_over_max_len():
    ids = list(range(257))
    out = head_tail_truncate(ids)
    assert len(out) == 256
    assert out[-1] == 256 and out[191] == 191 and out[192] == 257 - 64


def test_zero_tail_tokens_guard():
    ids = list(range(300))
    out = head_tail_truncate(ids, max_len=256, head_tokens=256, tail_tokens=0)
    assert out == ids[:256]


def test_returns_copy():
    ids = [1, 2, 3]
    out = head_tail_truncate(ids)
    out.append(4)
    assert ids == [1, 2, 3]


def test_encoded_text_flags():
    e = EncodedText(input_ids=[1] * 256, original_tokens=400)
    assert e.truncated and e.num_tokens == 256
    assert not EncodedText(input_ids=[1, 2], original_tokens=2).truncated


def test_pad_batch():
    assert pad_batch([[1, 2, 3], [4]]) == [[1, 2, 3], [4, 0, 0]]

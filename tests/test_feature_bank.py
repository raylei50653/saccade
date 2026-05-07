import torch

from saccade.perception.feature_bank import FeatureBank


def make_bank(**kwargs):
    defaults = dict(max_ids=10, feat_dim=4, similarity_threshold=0.85, device="cpu")
    defaults.update(kwargs)
    return FeatureBank(**defaults)


def rand_emb(dim=4):
    v = torch.randn(dim)
    return v / v.norm()


# --- reset ---


def test_reset_clears_state():
    bank = make_bank()
    bank.update(1, rand_emb(), frame_id=0)
    bank.reset()
    assert bank._slot_map == {}
    assert bank.ptr == 0
    assert not bank.is_active.any()


# --- _alloc_slot ---


def test_alloc_slot_same_key_reuses_slot():
    bank = make_bank()
    s1 = bank._alloc_slot(1, 0)
    s2 = bank._alloc_slot(1, 0)
    assert s1 == s2
    assert bank.ptr == 1


def test_alloc_slot_evicts_old_key_on_wrap():
    bank = make_bank(max_ids=2)
    bank._alloc_slot(0, 0)
    bank._alloc_slot(1, 0)
    bank._alloc_slot(2, 0)  # wraps, evicts slot 0's old key
    assert (0, 0) not in bank._slot_map


# --- update ---


def test_update_normalizes_embedding():
    bank = make_bank()
    raw = torch.tensor([3.0, 4.0, 0.0, 0.0])
    bank.update(1, raw, frame_id=0)
    slot = bank._slot_map[(1, 0)]
    norm = bank.features[slot].norm().item()
    assert abs(norm - 1.0) < 1e-5


def test_update_marks_slot_active():
    bank = make_bank()
    bank.update(1, rand_emb(), frame_id=5)
    slot = bank._slot_map[(1, 0)]
    assert bank.is_active[slot].item() is True
    assert bank.last_seen[slot].item() == 5


# --- update_batch ---


def test_update_batch_matches_sequential():
    bank_seq = make_bank()
    bank_bat = make_bank()
    ids = [1, 2, 3]
    embs = [rand_emb() for _ in ids]
    for tid, emb in zip(ids, embs):
        bank_seq.update(tid, emb, frame_id=0)
    track_ids = torch.tensor(ids)
    embeddings = torch.stack(embs)
    bank_bat.update_batch(track_ids, embeddings, frame_id=0)
    for tid in ids:
        s_seq = bank_seq._slot_map[(tid, 0)]
        s_bat = bank_bat._slot_map[(tid, 0)]
        assert torch.allclose(
            bank_seq.features[s_seq], bank_bat.features[s_bat], atol=1e-5
        )


def test_update_batch_empty_no_error():
    bank = make_bank()
    bank.update_batch(
        torch.zeros((0,), dtype=torch.long), torch.zeros((0, 4)), frame_id=0
    )
    assert bank._slot_map == {}


# --- find_matches_batch ---


def test_find_matches_batch_finds_correct_id():
    bank = make_bank(similarity_threshold=0.8)
    emb = rand_emb()
    bank.update(99, emb, frame_id=0)
    noisy = emb + torch.randn(4) * 0.01
    result = bank.find_matches_batch(noisy.unsqueeze(0), lost_ids=[99], current_frame=1)
    assert result == {0: 99}


def test_find_matches_batch_empty_query():
    bank = make_bank()
    bank.update(1, rand_emb(), frame_id=0)
    assert (
        bank.find_matches_batch(torch.zeros((0, 4)), lost_ids=[1], current_frame=1)
        == {}
    )


def test_find_matches_batch_empty_lost_ids():
    bank = make_bank()
    assert (
        bank.find_matches_batch(rand_emb().unsqueeze(0), lost_ids=[], current_frame=1)
        == {}
    )


def test_find_matches_batch_below_threshold_no_match():
    bank = make_bank(similarity_threshold=0.99)
    bank.update(1, torch.tensor([1.0, 0.0, 0.0, 0.0]), frame_id=0)
    query = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # orthogonal
    result = bank.find_matches_batch(query, lost_ids=[1], current_frame=1)
    assert result == {}


# --- find_cross_camera_matches ---


def test_cross_camera_ignores_same_stream():
    bank = make_bank(similarity_threshold=0.8)
    emb = rand_emb()
    bank.update(1, emb, frame_id=0, stream_id=0)
    result = bank.find_cross_camera_matches(emb.unsqueeze(0), current_stream_id=0)
    assert result == {}


def test_cross_camera_finds_other_stream():
    bank = make_bank(similarity_threshold=0.8)
    emb = rand_emb()
    bank.update(1, emb, frame_id=0, stream_id=0)
    result = bank.find_cross_camera_matches(emb.unsqueeze(0), current_stream_id=1)
    assert 0 in result
    assert result[0] == (0, 1)


def test_cross_camera_empty_query():
    bank = make_bank()
    bank.update(1, rand_emb(), frame_id=0, stream_id=0)
    result = bank.find_cross_camera_matches(torch.zeros((0, 4)), current_stream_id=1)
    assert result == {}

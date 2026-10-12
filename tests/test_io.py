import pytest
import torch
from grpo.io import load_policy, state_hash, file_hash


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    class Backbone(torch.nn.Module):
        n_layer = 2
        n_head = 1
        head_size = 64

        def __init__(self, path, device):
            super().__init__()
            self.register_buffer(
                "embedding", torch.zeros(1, 64, device=device, dtype=torch.bfloat16)
            )

        def weight(self, name):
            return self.embedding

    monkeypatch.setattr("grpo.model.Backbone", Backbone)
    path = tmp_path / "backbone.pth"
    path.write_bytes(b"fixture backbone")
    return str(path)


@pytest.mark.parametrize("setting", [{}, {"state": ""}, {"state": None}])
def test_default_state_matches_explicit_zero_checkpoint(model_path, tmp_path, setting):
    zero = torch.zeros(2, 1, 64, 64, dtype=torch.float32)
    checkpoint = tmp_path / "state.pt"
    torch.save(
        {
            "initial_state": zero,
            "state_version": state_hash(zero),
            "backbone_sha256": file_hash(model_path),
            "layout": "layer,head,value,key",
        },
        checkpoint,
    )
    default, snapshot = load_policy({"model": model_path, **setting}, "cpu")
    explicit, expected = load_policy(
        {"model": model_path, "state": str(checkpoint)}, "cpu"
    )
    assert snapshot == expected
    assert torch.equal(default.initial_state, explicit.initial_state)
    assert default.initial_state.dtype == torch.float32
    assert (
        isinstance(default.initial_state, torch.nn.Parameter)
        and default.initial_state.requires_grad
    )


def test_missing_explicit_state_does_not_silently_reset(model_path, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policy({"model": model_path, "state": str(tmp_path / "missing.pt")}, "cpu")


@pytest.mark.parametrize('layout', ['value-key', 'key-value'])
def test_raw_sft_initialization_and_export_axes(model_path, tmp_path, layout):
    from grpo.init_state import export_time_state
    state = torch.arange(2 * 64 * 64).reshape(2, 1, 64, 64).bfloat16()
    raw = {f'blocks.{i}.att.time_state': state[i] if layout == 'value-key' else state[i].transpose(-1, -2) for i in range(2)}
    path = tmp_path / 'sft.pth'
    torch.save(raw, path)
    policy, snapshot = load_policy({'model': model_path, 'state': str(path), 'state_layout': layout}, 'cpu')
    assert torch.equal(policy.initial_state, state.float())
    assert snapshot['state_version'] == state_hash(state)
    assert policy.initial_state.dtype == torch.float32 and policy.initial_state.requires_grad
    exported = export_time_state(policy.initial_state)
    assert set(exported) == set(raw)
    for i, tensor in enumerate(exported.values()):
        assert tensor.dtype == torch.bfloat16 and tensor.is_contiguous()
        assert torch.equal(tensor, state[i])
    # The export itself is directly reusable as the next run's SFT initialization.
    torch.save(exported, path)
    reloaded, expected = load_policy({'model': model_path, 'state': str(path), 'state_layout': 'value-key'}, 'cpu')
    assert snapshot == expected
    assert torch.equal(reloaded.initial_state, policy.initial_state)


@pytest.mark.parametrize('failure', ['layout', 'missing', 'extra', 'shape', 'nonfinite'])
def test_raw_sft_rejects_invalid_input(model_path, tmp_path, failure):
    raw = {f'blocks.{i}.att.time_state': torch.zeros(1, 64, 64) for i in range(2)}
    layout = 'value-key'
    if failure == 'layout':
        layout = None
    elif failure == 'missing':
        del raw['blocks.1.att.time_state']
    elif failure == 'extra':
        raw['blocks.2.att.time_state'] = torch.zeros(1, 64, 64)
    elif failure == 'shape':
        raw['blocks.1.att.time_state'] = torch.zeros(64, 64)
    else:
        raw['blocks.1.att.time_state'][0, 0, 0] = float('nan')
    path = tmp_path / 'invalid.pth'
    torch.save(raw, path)
    with pytest.raises(ValueError):
        load_policy({'model': model_path, 'state': str(path), 'state_layout': layout}, 'cpu')


def test_export_rejects_nonfinite_and_bf16_overflow():
    from grpo.init_state import export_time_state
    for value in (float('inf'), torch.finfo(torch.float32).max):
        state = torch.full((1, 1, 64, 64), value)
        with pytest.raises(ValueError):
            export_time_state(state)

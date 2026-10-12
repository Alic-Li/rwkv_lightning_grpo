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

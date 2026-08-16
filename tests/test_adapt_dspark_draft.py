import json
from pathlib import Path

from scripts.adapt_dspark_draft import adapt_config


def test_adapt_config_rewrites_specforge_contract() -> None:
    src = {
        "architectures": ["DSparkDraftModel"],
        "model_type": "dspark",
        "hidden_size": 5120,
        "dflash_config": {"projector_type": "other"},
    }
    out = adapt_config(src)
    assert out["architectures"] == ["Qwen3DSparkModel"]
    assert out["model_type"] == "qwen3"
    assert out["dspark_target_layer_ids"] == [4, 16, 28, 40, 52]
    assert out["n_predict"] == 7
    assert out["dspark_block_size"] == 7
    assert out["dflash_config"]["projector_type"] == "dspark"
    assert out["dflash_config"]["target_layer_ids"] == [4, 16, 28, 40, 52]
    assert out["auto_map"]["AutoModel"] == "dspark.DSparkDraftModel"


def test_adapted_fixture_matches_campaign_draft() -> None:
    fixture = Path(__file__).resolve().parents[1] / "reference" / "dspark-adapted-config.json"
    expected = json.loads(fixture.read_text())
    rebuilt = adapt_config({
        "architectures": ["DSparkDraftModel"],
        "model_type": "dspark",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "block_size": 7,
        "bos_token_id": 248044,
        "confidence_head_with_markov": True,
        "dtype": "bfloat16",
        "enable_confidence_head": True,
        "eos_token_id": 248046,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 10240,
        "layer_types": ["full_attention"] * 5,
        "markov_head_type": "vanilla",
        "markov_rank": 256,
        "max_position_embeddings": 262144,
        "max_window_layers": 5,
        "num_attention_heads": 40,
        "num_hidden_layers": 5,
        "num_key_value_heads": 8,
        "num_target_layers": 64,
        "pad_token_id": 248044,
        "rms_norm_eps": 1e-06,
        "rope_parameters": expected["rope_parameters"],
        "sliding_window": None,
        "tie_word_embeddings": False,
        "transformers_version": "5.12.1",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 248320,
        "dflash_config": {"mask_token_id": 248077},
    })
    assert rebuilt["architectures"] == expected["architectures"]
    assert rebuilt["model_type"] == expected["model_type"]
    assert rebuilt["dspark_target_layer_ids"] == expected["dspark_target_layer_ids"]
    assert rebuilt["n_predict"] == expected["n_predict"]
    assert rebuilt["dflash_config"]["target_layer_ids"] == expected["dflash_config"]["target_layer_ids"]
    assert rebuilt["dflash_config"]["projector_type"] == "dspark"

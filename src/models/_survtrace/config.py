"""SurvTRACE default configuration (replaces EasyDict with plain dict)."""


def default_config() -> dict:
    """Return a mutable copy of the default SurvTRACE configuration."""
    return {
        "data": "custom",
        "num_durations": 5,
        "horizons": [0.25, 0.5, 0.75],
        "seed": 1234,
        "checkpoint": "./checkpoints/survtrace.pt",
        "vocab_size": 8,
        "hidden_size": 16,
        "intermediate_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "hidden_dropout_prob": 0.0,
        "num_feature": 9,
        "num_numerical_feature": 5,
        "num_categorical_feature": 4,
        "out_feature": 3,
        "num_event": 1,
        "hidden_act": "gelu",
        "attention_probs_dropout_prob": 0.1,
        "early_stop_patience": 5,
        "initializer_range": 0.001,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "chunk_size_feed_forward": 0,
        "output_attentions": False,
        "output_hidden_states": False,
        "tie_word_embeddings": True,
        "pruned_heads": {},
    }

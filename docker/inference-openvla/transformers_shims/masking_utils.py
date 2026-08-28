"""Compatibility shim: lerobot 0.5.1's pi_gemma.py imports create_causal_mask
from this module (added in a later transformers than the 4.40.1 pinned by
openvla-oft's transformers fork, which this environment needs for OpenVLA's
bidirectional-attention parallel decoding). pi_gemma.py backs PI0/PI0Fast,
which this environment never constructs — only the import must succeed, not
the function's actual behavior. Real implementation, if ever exercised, would
need to match a much newer transformers' semantics; do not rely on this for
anything beyond satisfying the import.
"""


def create_causal_mask(*args, **kwargs):
    raise NotImplementedError(
        "create_causal_mask is a compatibility import-shim stub (transformers 4.40.1, "
        "openvla-oft's bidirectional-attention fork) — PI0/PI0Fast are not usable in "
        "this environment."
    )

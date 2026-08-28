"""Compatibility shim, see masking_utils.py in transformers/ for why."""


class _PaliGemmaStub:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "PaliGemma classes are a compatibility import-shim stub — "
            "PI0/PI0Fast are not usable in this environment."
        )


class PaliGemmaForConditionalGeneration(_PaliGemmaStub):
    pass


class PaliGemmaModel(_PaliGemmaStub):
    pass

"""CloudTrack FastAPI package bootstrap."""

from __future__ import annotations

import inspect
import sys
import typing


# This workspace currently uses Python 3.14.0rc2. Its private typing._eval_type
# predates the `prefer_fwd_module` keyword used by the installed Pydantic build.
# Keep the workaround narrowly gated; final/supported Python signatures bypass it.
if sys.version_info[:2] == (3, 14) and "prefer_fwd_module" not in inspect.signature(typing._eval_type).parameters:
    _original_eval_type = typing._eval_type

    def _cloudtrack_eval_type_compat(*args, **kwargs):
        kwargs.pop("prefer_fwd_module", None)
        return _original_eval_type(*args, **kwargs)

    typing._eval_type = _cloudtrack_eval_type_compat

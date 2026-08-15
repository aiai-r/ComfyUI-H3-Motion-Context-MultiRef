"""MiniMax H3 Motion Context / MultiRef / latent-masking custom nodes.

Importing the node pack registers nodes only. Compatibility behavior remains
lazy and capability-aware, so native ComfyUI functionality is preferred when
the installed version already provides the required H3 behavior.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

"""H3 Motion Context — MultiRef + Existing Video Extension.

Update 2 keeps all runtime compatibility lazy. Importing the custom node pack
registers nodes only; ComfyUI core is patched in memory only when a workflow
actually executes a feature that needs compatibility.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

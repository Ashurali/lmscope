"""lmscope: step through LSTM / Transformer language-model inference in the browser."""
from .inference import Inspector, load_checkpoint
from .server import create_app, serve

__all__ = ["serve", "create_app", "Inspector", "load_checkpoint"]
__version__ = "0.2.0"

"""Flask app: serves the single-page UI and a small JSON API around one Inspector."""
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .inference import Inspector, load_checkpoint

STATIC = Path(__file__).resolve().parent / "static"


def create_app(inspector, reference_text=None):
    app = Flask(__name__, static_folder=None)
    lock = threading.Lock()  # one model, one sequence: serialize requests

    def body():
        return request.get_json(silent=True) or {}

    def api(fn):
        def wrapper():
            try:
                with lock:
                    return jsonify(fn(body()))
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        wrapper.__name__ = fn.__name__
        return wrapper

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/api/model-info")
    @api
    def model_info(_):
        return {**inspector.model_info(), "reference_text": reference_text}

    @app.get("/api/state")
    @api
    def state(_):
        return inspector.state()

    @app.post("/api/set-prompt")
    @api
    def set_prompt(b):
        return inspector.set_prompt(b.get("prompt", ""))

    @app.post("/api/reset")
    @api
    def reset(_):
        return inspector.reset()

    @app.post("/api/step")
    @api
    def step(b):
        """One forward pass: distribution + internals for predicting the token at `position`."""
        return inspector.analyze(position=b.get("position"), temperature=float(b.get("temperature", 1.0)),
                                 layer=int(b.get("layer", 0)), head=b.get("head", "mean"),
                                 top_k=int(b.get("top_k", 20)))

    @app.post("/api/sample")
    @api
    def sample(b):
        """Samples and appends the next token; returns the choice and the updated sequence."""
        chosen = inspector.sample(float(b.get("temperature", 1.0)))
        return {"sampled": chosen, "state": inspector.state()}

    @app.post("/api/score")
    @api
    def score(b):
        return inspector.score(b.get("text", ""))

    return app


def serve(model_path, model_type=None, tokenizer_path=None, tokenizer_type=None, port=8765,
          host="127.0.0.1", prompt="ROMEO:\n", seed=0, reference_file=None, open_browser=True, device=None):
    """Loads a checkpoint and starts the UI at http://host:port."""
    model, tokenizer, config = load_checkpoint(model_path, model_type, tokenizer_path, tokenizer_type, device)
    inspector = Inspector(model, tokenizer, config, seed=seed)
    inspector.set_prompt(prompt)
    reference_text = Path(reference_file).read_text(encoding="utf-8") if reference_file else None
    app = create_app(inspector, reference_text)
    url = f"http://{host}:{port}"
    info = inspector.model_info()
    print(f"lmscope: {info['model_type']} ({info['params'] / 1e6:.2f}M params, {info['tokenizer_type']} "
          f"tokenizer, vocab {info['vocab_size']}) on {info['device']} -> {url}")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False, threaded=True)

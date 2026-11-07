"""CLI: lmscope serve --model ckpt.pt [--model-type ...] [--tokenizer ...] [--tokenizer-type ...]"""
import argparse

from .server import serve


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lmscope", description="Interactive LM inference visualizer")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="load a checkpoint and open the browser UI")
    s.add_argument("--model", required=True, help="checkpoint .pt in the lmscope format")
    s.add_argument("--model-type", choices=["transformer", "lstm"], help="checked against the checkpoint")
    s.add_argument("--tokenizer", help="tokenizer JSON (BPE: HuggingFace tokenizer file; char: char_to_idx)")
    s.add_argument("--tokenizer-type", choices=["char", "bpe"], help="default: from the checkpoint")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--prompt", default="ROMEO:\n", help="initial prompt (\\n is a newline)")
    s.add_argument("--seed", type=int, default=0, help="sampling seed, restored on reset")
    s.add_argument("--reference", help="text file preloaded as the reference continuation for loss/perplexity")
    s.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    s.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args(argv)
    serve(model_path=args.model, model_type=args.model_type, tokenizer_path=args.tokenizer,
          tokenizer_type=args.tokenizer_type, port=args.port, host=args.host,
          prompt=args.prompt.replace("\\n", "\n"), seed=args.seed, reference_file=args.reference,
          open_browser=not args.no_browser, device=args.device)


if __name__ == "__main__":
    main()

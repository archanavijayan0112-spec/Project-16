from __future__ import annotations

import argparse
import sys


def cmd_ingest(args):
    from docmind import Pipeline
    pipeline = Pipeline.from_config(args.config)
    result = pipeline.ingest(args.path)
    print(f"✓ Loaded {result.files_processed} files · {result.chunks_stored} chunks · {result.elapsed_s}s")
    if result.skipped:
        print(f"  (skipped {result.skipped} unchanged files)")


def cmd_query(args):
    from docmind import Pipeline
    pipeline = Pipeline.from_config(args.config)
    if args.stream:
        for token in pipeline.stream(args.question):
            print(token, end="", flush=True)
        print()
    else:
        answer = pipeline.query(args.question)
        print(answer.text)
        print()
        for src in answer.sources:
            print(f"  [{src.score}] {src.doc} (page {src.page})")


def cmd_serve(args):
    import uvicorn
    uvicorn.run("docmind.server:app", host=args.host, port=args.port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(prog="docmind", description="DocMind RAG CLI")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Index documents")
    p_ingest.add_argument("path", help="File or directory to index")

    # query
    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question", help="Question string")
    p_query.add_argument("--stream", action="store_true", help="Stream output")

    # serve
    p_serve = sub.add_parser("serve", help="Start the API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

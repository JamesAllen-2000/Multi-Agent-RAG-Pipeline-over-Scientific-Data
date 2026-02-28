"""
CLI to ingest documents (PDF/txt), structured (CSV), arXiv, or Kaggle datasets.
Usage:
  python -m scripts.ingest document path/to/file.pdf [--title "My Paper"]
  python -m scripts.ingest structured path/to/data.csv [--title "Dataset"]
  python -m scripts.ingest arxiv [path_ignored] --query "ti:electron" [--max-results 20]
  python -m scripts.ingest arxiv --id-list "2301.00001,2302.00002"
  python -m scripts.ingest kaggle --dataset "Cornell-University/arxiv"
"""
import argparse
import sys
from pathlib import Path

# Ensure app is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import get_ingester, list_source_types


def main():
    parser = argparse.ArgumentParser(description="Ingest documents, structured data, arXiv, or Kaggle datasets")
    parser.add_argument("source_type", choices=list_source_types(), help="document, structured, arxiv, or kaggle")
    parser.add_argument("path", type=Path, nargs="?", default=None, help="Path to file (required for document/structured; ignored for arxiv/kaggle)")
    parser.add_argument("--title", type=str, default=None, help="Optional title (document/structured/kaggle)")
    parser.add_argument("--query", type=str, default=None, help="arXiv search_query (e.g. 'all:electron', 'ti:transformer')")
    parser.add_argument("--id-list", type=str, default=None, help="arXiv id_list (comma-separated IDs)")
    parser.add_argument("--max-results", type=int, default=20, help="arXiv max results to fetch (default 20)")
    parser.add_argument("--dataset", type=str, default=None, help="Kaggle dataset slug (e.g. Cornell-University/arxiv)")
    args = parser.parse_args()

    if args.source_type in ("document", "structured") and args.path is None:
        parser.error("path is required for document and structured")
    if args.source_type == "arxiv" and not (args.query or args.id_list):
        parser.error("arxiv requires --query and/or --id-list")
    if args.source_type == "kaggle" and not args.dataset:
        parser.error("kaggle requires --dataset (e.g. Cornell-University/arxiv)")

    ingester = get_ingester(args.source_type)
    try:
        if args.source_type == "arxiv":
            meta = ingester.ingest(
                Path("."),
                search_query=args.query or "all:all",
                id_list=args.id_list,
                max_results=args.max_results,
            )
            count = meta.metadata.get("ingested_count", 1)
            print(f"Ingested {count} arXiv paper(s); first: {meta.source_id} - {meta.title[:60]}...")
        elif args.source_type == "kaggle":
            meta = ingester.ingest(Path("."), dataset_slug=args.dataset, title=args.title)
            count = meta.metadata.get("ingested_count", 0)
            print(f"Kaggle dataset {meta.source_id}: ingested {count} file(s) from {args.dataset}")
        else:
            meta = ingester.ingest(args.path, title=args.title)
            print(f"Ingested: {meta.source_id} ({meta.source_type}) - {meta.title}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

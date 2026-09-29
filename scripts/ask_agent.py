"""Index local files, ask one agent question, and print the answer."""

import argparse
import logging

from app.core.config import settings
from app.services import build_services


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "documents",
        nargs="+",
        help="PDF/text files to index before asking the question",
    )
    parser.add_argument(
        "question",
        help="Question to send to the agent after indexing the files",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    services = build_services(settings)
    for path in args.documents:
        chunks = services.pipeline.ingest(path)
        print(f"Indexed {chunks[0].filename}: {len(chunks)} chunks")

    result = services.agent.run(args.question)
    print(f"\nQUESTION: {args.question}")
    print(f"\nANSWER:\n{result.answer}")

    if result.tool_calls:
        print("\nTOOLS:")
        for trace in result.tool_calls:
            print(
                f"  step={trace.step} tool={trace.tool} "
                f"success={trace.success} args={trace.arguments}"
            )

    if result.sources:
        print(
            "\nSOURCES:",
            [source.model_dump(exclude_none=True) for source in result.sources],
        )


if __name__ == "__main__":
    main()

"""Upload local files in memory, then demonstrate six agent behaviors."""

import argparse
import logging

from app.core.config import settings
from app.services import build_services


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", help="PDF/text files to index")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    services = build_services(settings)
    filenames: list[str] = []
    for path in args.documents:
        chunks = services.pipeline.ingest(path)
        filenames.append(chunks[0].filename)
        print(f"Indexed {chunks[0].filename}: {len(chunks)} chunks")

    first = filenames[0]
    second = filenames[1] if len(filenames) > 1 else first
    examples = [
        "Hello, what can you do?",
        f"What is the main method discussed in {first}?",
        "If PSNR improves from 25.4 to 27.1, what is the absolute improvement?",
        f"Compare the main approaches in {first} and {second}.",
        f"Summarize {first}.",
        "What does the collection say about an unsupported imaginary topic?",
    ]

    for number, message in enumerate(examples, start=1):
        print(f"\n=== Example {number} ===\nUSER: {message}")
        result = services.agent.run(message)
        print(f"ANSWER: {result.answer}")
        print("TOOLS:")
        for trace in result.tool_calls:
            print(
                f"  step={trace.step} tool={trace.tool} "
                f"success={trace.success} args={trace.arguments}"
            )
        print("SOURCES:", [source.model_dump() for source in result.sources])


if __name__ == "__main__":
    main()

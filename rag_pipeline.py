import hashlib
import logging
from pathlib import Path

from llm_service import LLMConfigurationError, answer_from_matches
from rag_service import index_pdf, retrieve_chunks

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data" / "pdfs"

TOP_K = 5
MIN_SCORE = 0.2

logger = logging.getLogger(__name__)


def generate_document_id(pdf_path: Path) -> str:
    """Keep the same ID for the same relative PDF path."""
    relative_path = pdf_path.relative_to(PDF_DIR).as_posix()

    return hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()


def index_documents() -> list[str]:
    """Index PDFs and return IDs of successfully indexed documents."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(
        path
        for path in PDF_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        print(f"No PDFs found. Add your PDFs to:\n{PDF_DIR}")
        return []

    print(f"\nFound {len(pdf_files)} PDF file(s).")
    print("The embedding model may download on the first run.\n")

    document_ids = []

    for pdf_path in pdf_files:
        document_id = generate_document_id(pdf_path)
        source_name = pdf_path.relative_to(PDF_DIR).as_posix()

        print(f"Indexing: {source_name}")

        try:
            chunk_count = index_pdf(
                document_id=document_id,
                filename=source_name,
                pdf_path=pdf_path,
            )

        except Exception:
            logger.exception("Could not index %s", source_name)
            continue

        document_ids.append(document_id)
        print(f"Indexed successfully: {chunk_count} chunks\n")

    return document_ids


def display_sources(sources: list[dict]) -> None:
    """Display the reference numbers used in the generated answer."""
    if not sources:
        return

    print("\nRetrieved sources:")

    for source in sources:
        print(
            f"[{source['reference']}] "
            f"{source['source']} — page {source['page']} "
            f"(similarity: {source['similarity_score']:.3f})"
        )


def ask_questions(document_ids: list[str]) -> None:
    print("Your PDFs are ready.")
    print("Enter a question, or type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return

        if not question:
            print("Please enter a question.")
            continue

        try:
            print("\nSearching PDFs...")

            matches = retrieve_chunks(
                question=question,
                document_ids=document_ids,
                top_k=TOP_K,
                min_score=MIN_SCORE,
            )

            if not matches:
                print(
                    "No relevant passages found. "
                    "Try a different question or add explanatory notes."
                )
                continue

            print(
                f"Retrieved {len(matches)} passage(s). "
                "Generating answer..."
            )

            result = answer_from_matches(
                question=question,
                matches=matches,
            )

            print("\nAnswer:\n")
            print(result["answer"])

            display_sources(result["sources"])

        except LLMConfigurationError as exc:
            print(f"\nConfiguration error: {exc}")
            print(
                "Set GROQ_API_KEY in your project's .env file, "
                "then restart the pipeline."
            )
            return

        except ValueError as exc:
            print(f"\nInvalid input or configuration: {exc}")

        except Exception:
            logger.exception("Question processing failed")
            print(
                "\nCould not complete this question. "
                "Check the error above and try again."
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    print("PDF RAG Pipeline")
    print(f"PDF directory: {PDF_DIR}")

    try:
        document_ids = index_documents()

        if not document_ids:
            print("\nNo documents were successfully indexed.")
            return

        ask_questions(document_ids)

    except (KeyboardInterrupt, EOFError):
        print("\nPipeline stopped.")


if __name__ == "__main__":
    main()
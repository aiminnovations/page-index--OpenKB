"""PageIndex indexer for long documents (pageindex 0.4.0, local store under ``.openkb``)."""

from __future__ import annotations

import json as json_mod
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pageindex import PageIndexClient

from openkb.config import resolve_concurrency, resolve_effective_config
from openkb.tree_renderer import render_summary_md

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Result of indexing a long document via PageIndex."""

    doc_id: str
    description: str
    tree: dict


def _normalize_page_content(raw_pages: Any) -> list[dict[str, Any]]:
    """Normalize PageIndex/local PDF page content into OpenKB's JSON shape."""
    if not isinstance(raw_pages, list):
        return []

    pages: list[dict[str, Any]] = []
    for index, item in enumerate(raw_pages, start=1):
        if isinstance(item, str):
            content = item.strip()
            if content:
                pages.append({"page": index, "content": content, "images": []})
            continue

        if not isinstance(item, dict):
            continue

        raw_page = item.get("page", item.get("page_number", item.get("page_num", index)))
        try:
            page_number = int(raw_page)
        except (TypeError, ValueError):
            page_number = index
        if page_number < 1:
            page_number = index

        content = item.get("content", item.get("markdown", item.get("text", "")))
        if content is None:
            content = ""
        content = str(content).strip()

        images = item.get("images", [])
        if not isinstance(images, list):
            images = []
        normalized_images = [
            image
            for image in images
            if isinstance(image, dict) and isinstance(image.get("path"), str)
        ]

        if content or normalized_images:
            pages.append(
                {
                    "page": page_number,
                    "content": content,
                    "images": normalized_images,
                }
            )

    return pages


def _convert_pdf_to_pages(pdf_path: Path, doc_name: str, images_dir: Path) -> list[dict[str, Any]]:
    from openkb.images import convert_pdf_to_pages

    return convert_pdf_to_pages(pdf_path, doc_name, images_dir)


def _write_long_doc_artifacts(
    tree: dict,
    pages: list[dict[str, Any]],
    doc_name: str,
    doc_id: str,
    kb_dir: Path,
    description: str = "",
) -> Path:
    """Write ``wiki/sources/<doc_name>.json`` + ``wiki/summaries/<doc_name>.md``.

    Returns the summary path. Shared by :func:`index_long_document` and
    ``cli.import_prepared`` (a tree + pages indexed elsewhere) so both produce
    identical artifacts. Page images, when present, are written separately by the
    caller's page extractor — this helper only persists page text + summary.
    """
    sources_dir = kb_dir / "wiki" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    (sources_dir / f"{doc_name}.json").write_text(
        json_mod.dumps(pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summaries_dir = kb_dir / "wiki" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summaries_dir / f"{doc_name}.md"
    summary_path.write_text(
        render_summary_md(tree, doc_name, doc_id, description=description), encoding="utf-8"
    )
    return summary_path


def _local_index(config: dict[str, Any], model: str, openkb_dir: Path) -> dict[str, Any]:
    """The pageindex 0.4.0 ``index=`` slot (a ``LocalIndexConfig``) for this KB.

    0.4.0 indexes locally with Flash (node summaries and the document
    description included); node text comes back from ``get_tree``. The KB's
    ``concurrency`` caps simultaneous indexing LLM calls (``summary_concurrency``).
    """
    index: dict[str, Any] = {"mode": "local", "model": model, "storage_path": str(openkb_dir)}
    concurrency = resolve_concurrency(config)
    if concurrency is not None:
        index["summary_concurrency"] = concurrency
    return index


def index_long_document(pdf_path: Path, kb_dir: Path, doc_name: str | None = None) -> IndexResult:
    """Index a long PDF document using PageIndex and write wiki pages.

    ``doc_name`` is the collision-resistant wiki name used for all written
    artifacts; defaults to the PDF's stem for backward compatibility.
    """
    source_name = doc_name or pdf_path.stem
    openkb_dir = kb_dir / ".openkb"
    config = resolve_effective_config(kb_dir)[0]

    model: str = config.get("model", "gpt-5.4")
    client = PageIndexClient(index=_local_index(config, model, openkb_dir))

    # Add PDF (retry up to 3 times — PageIndex TOC accuracy is stochastic)
    max_retries = 3
    doc_id = None
    for attempt in range(1, max_retries + 1):
        try:
            doc_id = client.submit_document(str(pdf_path))["doc_id"]
            logger.info(
                "PageIndex added %s → doc_id=%s (attempt %d)", pdf_path.name, doc_id, attempt
            )
            break
        except Exception as exc:
            logger.warning(
                "PageIndex attempt %d/%d failed for %s: %s",
                attempt,
                max_retries,
                pdf_path.name,
                exc,
            )
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to index {pdf_path.name} after {max_retries} attempts: {exc}"
                ) from exc

    # The PageIndex document (.openkb/docs/<doc_id>) is now durably on disk.
    # The add mutation registers it via snapshot.track_new() only on a
    # successful return, so if any step below fails, delete the document we
    # just added; otherwise it leaks as an orphan that the manifest (rolled
    # back by the snapshot) no longer lists and no reaper reclaims.
    try:
        doc = client.get_document(doc_id)
        description: str = doc.get("description") or ""
        structure: list = client.get_tree(doc_id, node_summary=True, include_text=True).get(
            "result"
        ) or []
        tree = {
            "doc_name": doc.get("name") or pdf_path.stem,
            "doc_description": description,
            "structure": structure,
        }

        # Write wiki/sources/ — per-page content (text + images via pymupdf)
        images_dir = kb_dir / "wiki" / "sources" / "images" / source_name
        all_pages = _normalize_page_content(
            _convert_pdf_to_pages(pdf_path, source_name, images_dir)
        )
        if not all_pages:
            raise RuntimeError(f"No page content extracted for {pdf_path.name}")

        _write_long_doc_artifacts(
            tree, all_pages, source_name, doc_id, kb_dir, description=description
        )
        return IndexResult(doc_id=doc_id, description=description, tree=tree)
    except BaseException:
        # Best-effort: remove the document this add created. A failure here
        # (e.g. a second interrupt) only means it may stay orphaned — the
        # original error still propagates so the caller (mutation coordinator)
        # rolls back everything else it snapshotted.
        try:
            client.delete_document(doc_id)
        except Exception:
            logger.warning(
                "PageIndex cleanup of %s failed after error; document may be orphaned", doc_id
            )
        raise

"""Tests for openkb.indexer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from openkb.indexer import (
    IndexResult,
    _local_index,
    _normalize_page_content,
    index_long_document,
)


class TestLocalIndex:
    """The pageindex 0.4.0 ``index=`` slot: a local store under .openkb."""

    def test_local_mode_model_and_storage(self, tmp_path):
        assert _local_index({}, "gpt-4o-mini", tmp_path) == {
            "mode": "local",
            "model": "gpt-4o-mini",
            "storage_path": str(tmp_path),
        }

    def test_forwards_concurrency(self, tmp_path):
        assert _local_index({"concurrency": 8}, "m", tmp_path)["summary_concurrency"] == 8

    @pytest.mark.parametrize("value", [None, 0])
    def test_unset_or_invalid_concurrency_is_left_to_pageindex_default(self, tmp_path, value):
        assert "summary_concurrency" not in _local_index({"concurrency": value}, "m", tmp_path)


class TestNormalizePageContent:
    def test_normalizes_pageindex_dicts(self):
        pages = _normalize_page_content(
            [
                {
                    "page_number": "2",
                    "markdown": "  Page two  ",
                    "images": [{"path": "sources/images/doc/a.png"}],
                },
                {"page_num": 3, "text": "Page three", "images": "bad"},
            ]
        )

        assert pages == [
            {
                "page": 2,
                "content": "Page two",
                "images": [{"path": "sources/images/doc/a.png"}],
            },
            {"page": 3, "content": "Page three", "images": []},
        ]

    def test_normalizes_string_pages(self):
        pages = _normalize_page_content([" page one ", "", "page three"])

        assert pages == [
            {"page": 1, "content": "page one", "images": []},
            {"page": 3, "content": "page three", "images": []},
        ]

    def test_rejects_unusable_shapes(self):
        assert _normalize_page_content({"page": 1}) == []
        assert _normalize_page_content([None, {}, {"content": ""}]) == []


class TestIndexLongDocument:
    def _make_fake_client(self, doc_id: str, sample_tree: dict):
        """A pageindex 0.4.0 client returning the sample_tree fixture data."""
        client = MagicMock()
        client.submit_document.return_value = {"doc_id": doc_id, "name": "sample.pdf"}
        client.get_document.return_value = {
            "id": doc_id,
            "name": sample_tree["doc_name"],
            "description": sample_tree["doc_description"],
            "status": "completed",
        }
        client.get_tree.return_value = {"doc_id": doc_id, "result": sample_tree["structure"]}
        return client

    def _fake_pages(self):
        return [
            {"page": 1, "content": "Page one text.", "images": []},
            {"page": 2, "content": "Page two text.", "images": []},
        ]

    def _run(self, client, pdf_path, kb_dir, pages=None, **kw):
        with (
            patch("openkb.indexer.PageIndexClient", return_value=client) as mock_cls,
            patch(
                "openkb.indexer._convert_pdf_to_pages",
                return_value=self._fake_pages() if pages is None else pages,
            ),
        ):
            return index_long_document(pdf_path, kb_dir, **kw), mock_cls

    def test_returns_index_result(self, kb_dir, sample_tree, tmp_path):
        client = self._make_fake_client("abc-123", sample_tree)
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        result, _ = self._run(client, pdf_path, kb_dir)

        assert isinstance(result, IndexResult)
        assert result.doc_id == "abc-123"
        assert result.description == sample_tree["doc_description"]
        assert result.tree["structure"] == sample_tree["structure"]
        client.submit_document.assert_called_once_with(str(pdf_path))
        client.get_tree.assert_called_once_with("abc-123", node_summary=True, include_text=True)

    def test_deletes_pageindex_doc_when_a_post_add_step_fails(self, kb_dir, sample_tree, tmp_path):
        """The PageIndex document is durably written by submit_document(), but
        .openkb/docs is not in the add mutation's eager snapshot — track_new only
        registers it on a successful return. So if any step after the submit
        raises (here: get_document), index_long_document must delete the doc it
        just added; otherwise it leaks as an orphan the manifest no longer lists."""
        client = self._make_fake_client("abc-123", sample_tree)
        client.get_document.side_effect = RuntimeError("get_document blew up")
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with pytest.raises(RuntimeError, match="get_document blew up"):
            self._run(client, pdf_path, kb_dir)

        client.delete_document.assert_called_once_with("abc-123")

    def test_source_page_written_as_json(self, kb_dir, sample_tree, tmp_path):
        """Long doc source should be written as JSON, not markdown."""
        import json as json_mod

        client = self._make_fake_client("abc-123", sample_tree)
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        self._run(client, pdf_path, kb_dir)

        json_file = kb_dir / "wiki" / "sources" / "sample.json"
        assert json_file.exists()
        assert not (kb_dir / "wiki" / "sources" / "sample.md").exists()
        data = json_mod.loads(json_file.read_text())
        assert len(data) == 2
        assert data[0]["page"] == 1
        assert data[0]["content"] == "Page one text."

    def test_summary_page_written(self, kb_dir, sample_tree, tmp_path):
        client = self._make_fake_client("abc-123", sample_tree)
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        self._run(client, pdf_path, kb_dir)

        summary_file = kb_dir / "wiki" / "summaries" / "sample.md"
        assert summary_file.exists()
        content = summary_file.read_text(encoding="utf-8")
        assert "doc_type: pageindex" in content
        assert "Summary:" in content

    def test_client_is_local_under_openkb(self, kb_dir, sample_tree, tmp_path):
        """The client is a 0.4.0 local client storing under .openkb."""
        client = self._make_fake_client("xyz-456", sample_tree)
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        _, mock_cls = self._run(client, pdf_path, kb_dir)

        mock_cls.assert_called_once()
        index = mock_cls.call_args.kwargs["index"]
        assert index["mode"] == "local"
        assert index["storage_path"] == str(kb_dir / ".openkb")

    def test_concurrency_flows_from_kb_config(self, kb_dir, sample_tree, tmp_path):
        """The KB's real config.yaml, loaded by index_long_document itself, must
        reach the index slot passed to PageIndexClient."""
        (kb_dir / ".openkb" / "config.yaml").write_text(
            "model: gpt-4o-mini\nconcurrency: 7\n", encoding="utf-8"
        )
        client = self._make_fake_client("conc-789", sample_tree)
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        _, mock_cls = self._run(client, pdf_path, kb_dir)

        index = mock_cls.call_args.kwargs["index"]
        assert index["summary_concurrency"] == 7
        assert index["model"] == "gpt-4o-mini"

    def test_empty_pages_fail_and_clean_up(self, kb_dir, sample_tree, tmp_path):
        client = self._make_fake_client("empty-123", sample_tree)
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with pytest.raises(RuntimeError, match="No page content extracted"):
            self._run(client, pdf_path, kb_dir, pages=[])
        client.delete_document.assert_called_once_with("empty-123")


def test_index_long_document_uses_explicit_doc_name(kb_dir):
    client = MagicMock()
    client.submit_document.return_value = {"doc_id": "doc-123"}
    client.get_document.return_value = {"name": "original.pdf", "description": "desc"}
    client.get_tree.return_value = {"result": []}

    pdf = kb_dir / "raw" / "original.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with (
        patch("openkb.indexer.PageIndexClient", return_value=client),
        patch(
            "openkb.indexer._convert_pdf_to_pages", return_value=[{"page": 1, "text": "p1"}]
        ) as mock_convert,
    ):
        result = index_long_document(pdf, kb_dir, doc_name="original-abc12345")

    assert result.doc_id == "doc-123"
    assert (kb_dir / "wiki" / "sources" / "original-abc12345.json").exists()
    assert (kb_dir / "wiki" / "summaries" / "original-abc12345.md").exists()
    # nothing written under the raw stem
    assert not (kb_dir / "wiki" / "sources" / "original.json").exists()
    assert not (kb_dir / "wiki" / "summaries" / "original.md").exists()
    # the page extractor receives the explicit doc_name and its images dir
    expected_images = kb_dir / "wiki" / "sources" / "images" / "original-abc12345"
    mock_convert.assert_called_once_with(pdf, "original-abc12345", expected_images)
    # summary frontmatter points full_text at the doc_name artifact
    summary_text = (kb_dir / "wiki" / "summaries" / "original-abc12345.md").read_text(
        encoding="utf-8"
    )
    assert "original-abc12345" in summary_text


def test_write_long_doc_artifacts_writes_json_and_summary(kb_dir, sample_tree):
    from openkb.indexer import _write_long_doc_artifacts

    pages = [{"page": 1, "content": "Hello.", "images": []}]
    summary_path = _write_long_doc_artifacts(sample_tree, pages, "my-doc", "doc-1", kb_dir)

    assert summary_path == kb_dir / "wiki" / "summaries" / "my-doc.md"
    assert summary_path.exists()
    json_file = kb_dir / "wiki" / "sources" / "my-doc.json"
    assert json_file.exists()
    assert '"content": "Hello."' in json_file.read_text(encoding="utf-8")
    assert "doc_type: pageindex" in summary_path.read_text(encoding="utf-8")

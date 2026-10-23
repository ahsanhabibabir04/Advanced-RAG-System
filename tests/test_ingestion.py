
import tempfile
import unittest
from pathlib import Path

import pymupdf

from ingestion import extract_pdf_pages, chunk_pdf_pages


class TestPDFIngestion(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.temp_dir.name) / "sample.pdf"

        with pymupdf.open() as document:
            page1 = document.new_page()
            page1.insert_text(
                (72, 72),
                "Python is a programming language."
            )

            page2 = document.new_page()
            page2.insert_text(
                (72, 72),
                "Machine learning uses data to learn patterns."
            )

            document.save(self.pdf_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_pdf_pages(self):
        pages = extract_pdf_pages(self.pdf_path)

        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["page"], 1)
        self.assertEqual(pages[1]["page"], 2)
        self.assertEqual(pages[0]["source"], "sample.pdf")
        self.assertIn("Python", pages[0]["text"])

    def test_chunk_pdf_pages_preserves_metadata(self):
        chunks = chunk_pdf_pages(self.pdf_path)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["page"], 1)
        self.assertEqual(chunks[1]["page"], 2)
        self.assertTrue(all(c["source"] == "sample.pdf" for c in chunks))

    def test_missing_pdf_raises_error(self):
        missing_path = Path(self.temp_dir.name) / "missing.pdf"

        with self.assertRaises(FileNotFoundError):
            extract_pdf_pages(missing_path)

    def test_non_pdf_file_raises_error(self):
        text_path = Path(self.temp_dir.name) / "sample.txt"
        text_path.write_text("Not a PDF", encoding="utf-8")

        with self.assertRaises(ValueError):
            extract_pdf_pages(text_path)


if __name__ == "__main__":
    unittest.main()

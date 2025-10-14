import arxiv
import pandas as pd
from typing import List, Dict


class ArXivCollector:
	def __init__(self, max_results: int = 50):
		self.max_results = max_results

	def fetch_papers(self, query: str) -> pd.DataFrame:
		search = arxiv.Search(
			query=query,
			max_results=self.max_results,
			sort_by=arxiv.SortCriterion.SubmittedDate,
		)
		rows: List[Dict] = []
		for r in search.results():
			rows.append({
				"id": r.get_short_id(),
				"title": r.title,
				"abstract": r.summary,
				"authors": ", ".join(a.name for a in r.authors),
				"published": r.published.isoformat() if r.published else None,
				"updated": r.updated.isoformat() if r.updated else None,
				"categories": ",".join(r.categories) if r.categories else None,
				"entry_id": r.entry_id,
				"pdf_url": r.pdf_url,
			})
		return pd.DataFrame(rows)

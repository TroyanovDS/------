import os
import sys
import argparse
import pandas as pd

# Ensure project root is on sys.path to import local arxiv_collector.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from arxiv_collector import ArXivCollector
import os

CATEGORIES = {
    "text_mining": "all:machine learning",
	"information_retrieval": "all:information retrieval"
}


def collect_arxiv():
    """Функция для сбора, сохранения исходных данных и перевода"""
    collector = ArXivCollector(max_results=20)


    # Создаем папки заранее
    os.makedirs("data/arxiv_docs", exist_ok=True)

    for filename, query in CATEGORIES.items():
        print(f"\nОбработка: {query}")

        # Получаем и сохраняем исходные данные
        df_original = collector.fetch_papers(query)
        print(f"Получено {len(df_original)} документов")
        if len(df_original) > 0:
            print(f"Пример заголовка: {df_original.iloc[0]['title']}")
        df_original.to_csv(f"data/arxiv_docs/{filename}.csv", index=False, encoding='utf-8-sig')



if __name__ == "__main__":
    collect_arxiv()


import os
import sys
import argparse
import pandas as pd

# Ensure project root is on sys.path to import local arxiv_collector.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from arxiv_collector import ArXivCollector
import os

CATEGORIES = {
    "text_mining": "ti:\"text mining\" OR abs:\"text mining\" OR cat:cs.CL",
    "information_retrieval": "ti:\"information retrieval\" OR abs:\"information retrieval\" OR cat:cs.IR",
}


def collect_arxiv(count_per_topic: int = 50):
    """Функция для сбора, сохранения исходных данных и перевода"""
    collector = ArXivCollector(max_results=count_per_topic)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    collect_arxiv(args.count)


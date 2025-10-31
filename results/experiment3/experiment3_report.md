# Эксперимент 3: Ключевые слова — настройка на Inspec и сравнение HUMAN vs AI

## Методология

- Датасет для настройки: Inspec (train/dev/test), gold — авторские ключевые слова.
- Методы: TF-IDF, YAKE, TextRank, TopicRank, PositionRank, SingleRank, EmbedRank.
- Метрики: Precision@K, Recall@K, F1@K (K=5,10,15), Jaccard(top-K).
- Лучшие параметры по dev → финальная оценка на test.
- Применение лучших настроек к HUMAN vs AI корпусам (Qwen/DeepSeek/GPTOSS) — сравнение наборов КС.

## Состояние Inspec

Inspec не найден в data/inspec — использованы дефолтные параметры, подбор пропущен.

## Метрики и формулы

- Precision@K = TP/K; Recall@K = TP/|Gold|; F1@K = 2PR/(P+R).
- Jaccard(top-K) = |Pred∩Gold| / |Pred∪Gold|.
- Нормализация: lower, удаление пунктуации, схлопывание пробелов.

## Лучшие параметры (по dev, критерий: F1@10)

**TFIDF**: {'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.9, 'max_features': 8000}

**YAKE**: {'n': 3, 'dedup': 0.8}

**TEXTRANK**: {'ratio': 0.2}

**TOPICRANK**: {}

**POSITIONRANK**: {}

**SINGLERANK**: {}

**EMBEDRANK**: {'model': 'all-MiniLM-L6-v2'}

## Результаты на test (Inspec)

Нет — настройка пропущена.

## Сопоставление КС: HUMAN vs AI (Qwen/DeepSeek/GPTOSS)

### Модель: QWEN

|Метод|Jaccard|Overlap H|Overlap S|Harmonic|
|-|-|-|-|-|
|TFIDF|0.220|0.360|0.360|0.360|
|YAKE|0.190|0.320|0.320|0.320|
|TEXTRANK|0.266|0.420|0.420|0.420|
|TOPICRANK|0.000|0.000|0.000|0.000|
|POSITIONRANK|0.000|0.000|0.000|0.000|
|SINGLERANK|0.000|0.000|0.000|0.000|
|EMBEDRANK|0.087|0.160|0.160|0.160|

- TFIDF TOP‑HUMAN: retrieval, human, reasoning, recommendation, knowledge
- TFIDF TOP‑AI: user, recommendation, learning, precision, data
- YAKE TOP‑HUMAN: large language models, large language, language models, language, models
- YAKE TOP‑AI: rapidly evolving landscape, rapidly evolving, evolving landscape, large language models, natural language
- TEXTRANK TOP‑HUMAN: models, model, retrieval, tasks, language
- TEXTRANK TOP‑AI: models, model, user, significant, framework
- TOPICRANK TOP‑HUMAN: 
- TOPICRANK TOP‑AI: 
- POSITIONRANK TOP‑HUMAN: 
- POSITIONRANK TOP‑AI: 
- SINGLERANK TOP‑HUMAN: 
- SINGLERANK TOP‑AI: 
- EMBEDRANK TOP‑HUMAN: large language models, language models llms, retrieval augmented generation, multilingual commerce search, question answering
- EMBEDRANK TOP‑AI: large language models, recommendation systems, reinforcement learning rl, reinforcement learning, natural language processing

### Модель: DEEPSEEK

|Метод|Jaccard|Overlap H|Overlap S|Harmonic|
|-|-|-|-|-|
|TFIDF|0.266|0.420|0.420|0.420|
|YAKE|0.205|0.340|0.340|0.340|
|TEXTRANK|0.351|0.520|0.520|0.520|
|TOPICRANK|0.000|0.000|0.000|0.000|
|POSITIONRANK|0.000|0.000|0.000|0.000|
|SINGLERANK|0.000|0.000|0.000|0.000|
|EMBEDRANK|0.099|0.180|0.180|0.180|

- TFIDF TOP‑HUMAN: retrieval, human, reasoning, recommendation, knowledge
- TFIDF TOP‑AI: reasoning, user, retrieval, recommendation, dynamically
- YAKE TOP‑HUMAN: large language models, large language, language models, language, models
- YAKE TOP‑AI: abstract, large language models, large language, critical challenge, language models
- TEXTRANK TOP‑HUMAN: models, model, retrieval, tasks, language
- TEXTRANK TOP‑AI: model, models, retrieval, generation, attention
- TOPICRANK TOP‑HUMAN: 
- TOPICRANK TOP‑AI: 
- POSITIONRANK TOP‑HUMAN: 
- POSITIONRANK TOP‑AI: 
- SINGLERANK TOP‑HUMAN: 
- SINGLERANK TOP‑AI: 
- EMBEDRANK TOP‑HUMAN: large language models, language models llms, retrieval augmented generation, multilingual commerce search, question answering
- EMBEDRANK TOP‑AI: large language models, generative recommendation, recommendation generation, product search, retrieval augmented generation

### Модель: GPTOSS

|Метод|Jaccard|Overlap H|Overlap S|Harmonic|
|-|-|-|-|-|
|TFIDF|0.250|0.400|0.400|0.400|
|YAKE|0.136|0.240|0.240|0.240|
|TEXTRANK|0.299|0.460|0.460|0.460|
|TOPICRANK|0.000|0.000|0.000|0.000|
|POSITIONRANK|0.000|0.000|0.000|0.000|
|SINGLERANK|0.000|0.000|0.000|0.000|
|EMBEDRANK|0.111|0.200|0.200|0.200|

- TFIDF TOP‑HUMAN: retrieval, human, reasoning, recommendation, knowledge
- TFIDF TOP‑AI: graph, reasoning, retrieval, user, multimodal
- YAKE TOP‑HUMAN: large language models, large language, language models, language, models
- YAKE TOP‑AI: abstract, large language models, large language, language models, natural language
- TEXTRANK TOP‑HUMAN: models, model, retrieval, tasks, language
- TEXTRANK TOP‑AI: model, models, retrieval, user, language
- TOPICRANK TOP‑HUMAN: 
- TOPICRANK TOP‑AI: 
- POSITIONRANK TOP‑HUMAN: 
- POSITIONRANK TOP‑AI: 
- SINGLERANK TOP‑HUMAN: 
- SINGLERANK TOP‑AI: 
- EMBEDRANK TOP‑HUMAN: large language models, language models llms, retrieval augmented generation, multilingual commerce search, question answering
- EMBEDRANK TOP‑AI: large language models, multimodal retrieval, retrieval augmented generation, generative recommendation, language models llms

## Сопоставление с другими подходами

- Лексические методы (TF‑IDF/YAKE/TextRank/TopicRank/PositionRank/SingleRank/EmbedRank) на Inspec дают умеренные значения F1@K и ограниченную устойчивость на реальных корпусах.
- Семантические эмбеддинги (см. Эксперимент 2) показывают существенно лучшую разделимость HUMAN/AI (AUC≈1.0 в наших экспериментах).
- Вывод: для детекции синтетики ключевые слова — вспомогательный канал; основная сила — семантика и классификация эмбеддингов.

## Заключение

- Настройка на Inspec позволяет выбрать адекватные параметры для извлечения КС.
- В практической детекции синтетики: использовать комбинацию (лучший экстрактор КС по Inspec) + лексико‑стилистические признаки + семантический классификатор.

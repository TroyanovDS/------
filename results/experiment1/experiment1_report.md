# Эксперимент 1: Анализ ключевых слов в человеческих и синтетических текстах

## Методология

- **Выборка**: 30 человеческих + 30 синтетических документов (по 15 на тему)
- **Темы**: Text Mining, Information Retrieval
- **Методы извлечения ключевых слов**:
  - N-граммы (TF-IDF, 1-3 граммы)
  - YAKE (Yet Another Keyword Extractor)
  - TextRank
- **Метрики сравнения**: Jaccard, Precision, Recall, F1-score

## Визуализация результатов

### Сравнение метрик по методам

![Сравнение метрик](metrics_comparison.png)

### Топ ключевых слов по темам

![Топ ключевых слов Text Mining](top_keywords_text_mining.png)

![Топ ключевых слов Information Retrieval](top_keywords_information_retrieval.png)

### Анализ разнообразия

![Анализ разнообразия](diversity_analysis.png)

## Результаты по темам

### Text Mining

#### Статистика документов

- Человеческих документов: 15
- Синтетических документов: 15

#### NGRAMS

**Топ-10 ключевых слов (человеческие тексты):**
1. reasoning
2. model
3. models
4. time
5. data
6. language
7. performance
8. tasks
9. based
10. training

**Топ-10 ключевых слов (синтетические тексты):**
1. model
2. framework
3. novel
4. language
5. learning
6. models
7. time
8. performance
9. dynamics
10. including

**Метрики пересечения:**
- Jaccard Index: 0.299
- Precision: 0.460
- Recall: 0.460
- F1-score: 0.460
- Пересечение: 23 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.00
- Синтетические тексты: уникальность 1.000, средняя длина 1.02

#### YAKE

**Топ-10 ключевых слов (человеческие тексты):**
1. reasoning
2. model
3. demonstrated impressive reasoning
4. reasoning models
5. data
6. Large language models
7. LLM reasoning
8. performance
9. tasks
10. language

**Топ-10 ключевых слов (синтетические тексты):**
1. model
2. framework
3. language
4. language models
5. neural network models
6. learning
7. performance
8. dynamics
9. neural
10. tasks

**Метрики пересечения:**
- Jaccard Index: 0.149
- Precision: 0.260
- Recall: 0.260
- F1-score: 0.260
- Пересечение: 13 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.80
- Синтетические тексты: уникальность 1.000, средняя длина 1.54

#### TEXTRANK

**Топ-10 ключевых слов (человеческие тексты):**
1. model
2. modeling
3. reasoning
4. reason
5. language models
6. benchmarks
7. benchmark
8. benchmarking
9. dynamical
10. dynamics

**Топ-10 ключевых слов (синтетические тексты):**
1. model
2. models
3. modeling
4. dynamics
5. dynamically
6. dynamical
7. dynamic
8. including
9. include
10. languages

**Метрики пересечения:**
- Jaccard Index: 0.235
- Precision: 0.380
- Recall: 0.380
- F1-score: 0.380
- Пересечение: 19 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.04
- Синтетические тексты: уникальность 1.000, средняя длина 1.04

---

### Information Retrieval

#### Статистика документов

- Человеческих документов: 15
- Синтетических документов: 15

#### NGRAMS

**Топ-10 ключевых слов (человеческие тексты):**
1. models
2. data
3. model
4. retrieval
5. reasoning
6. time
7. reward
8. spatial
9. demonstrate
10. information

**Топ-10 ключевых слов (синтетические тексты):**
1. novel
2. framework
3. matrices
4. idempotent
5. model
6. demonstrate
7. idempotent matrices
8. introduce
9. models
10. significant

**Метрики пересечения:**
- Jaccard Index: 0.190
- Precision: 0.320
- Recall: 0.320
- F1-score: 0.320
- Пересечение: 16 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.02
- Синтетические тексты: уникальность 1.000, средняя длина 1.02

#### YAKE

**Топ-10 ключевых слов (человеческие тексты):**
1. models
2. Community Notes
3. spatial reasoning
4. data
5. spatial
6. retrieval
7. Notes
8. all-scale spatial reasoning
9. large language models
10. all-scale spatial

**Топ-10 ключевых слов (синтетические тексты):**
1. idempotent matrices
2. framework
3. modified gravity
4. matrices
5. models
6. idempotent
7. demonstrate
8. kinematic anchors
9. late-time Universe
10. Hierarchical Semantic Fusion

**Метрики пересечения:**
- Jaccard Index: 0.136
- Precision: 0.240
- Recall: 0.240
- F1-score: 0.240
- Пересечение: 12 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.74
- Синтетические тексты: уникальность 1.000, средняя длина 1.32

#### TEXTRANK

**Топ-10 ключевых слов (человеческие тексты):**
1. modeling
2. models
3. model
4. informed
5. information
6. inform
7. data
8. informative reward functions
9. generative
10. generation

**Топ-10 ключевых слов (синтетические тексты):**
1. model
2. models
3. modeling
4. networks
5. network
6. datasets
7. dataset
8. including
9. error
10. errors

**Метрики пересечения:**
- Jaccard Index: 0.149
- Precision: 0.260
- Recall: 0.260
- F1-score: 0.260
- Пересечение: 13 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.06
- Синтетические тексты: уникальность 1.000, средняя длина 1.04

---

## Общие выводы

### Сравнение методов извлечения ключевых слов

| Метод | Средний Jaccard | Средний F1 | Средняя уникальность |
|-------|----------------|------------|---------------------|
| NGRAMS | 0.245 | 0.390 | 1.000 |
| YAKE | 0.143 | 0.250 | 1.000 |
| TEXTRANK | 0.192 | 0.320 | 1.000 |

### Ключевые наблюдения

1. **Различия в ключевых словах**: Синтетические тексты показывают различия в выборе ключевых слов по сравнению с человеческими.
2. **Эффективность методов**: Различные методы извлечения ключевых слов дают разные результаты.
3. **Тематическая специфичность**: Каждая тема имеет свои характерные ключевые слова.
4. **Потенциал для детекции**: Различия в ключевых словах могут использоваться для выявления AI-сгенерированных текстов.

## Заключение

Эксперимент показал, что анализ ключевых слов может быть эффективным методом для различения человеческих и AI-сгенерированных текстов. Различные методы извлечения ключевых слов дают дополнительные возможности для анализа и могут быть объединены для повышения точности детекции.

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
1. model
2. reasoning
3. models
4. data
5. based
6. time
7. using
8. performance
9. forecasting
10. fidelity

**Топ-10 ключевых слов (синтетические тексты):**
1. model
2. models
3. performance
4. time
5. existing
6. neural
7. research
8. language
9. efficient
10. results

**Метрики пересечения:**
- Jaccard Index: 0.136
- Precision: 0.240
- Recall: 0.240
- F1-score: 0.240
- Пересечение: 12 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.04
- Синтетические тексты: уникальность 1.000, средняя длина 1.06

#### YAKE

**Топ-10 ключевых слов (человеческие тексты):**
1. model
2. problem
3. performance
4. datasets remains laborious
5. curating large reasoning
6. large reasoning datasets
7. Large language models
8. reasoning datasets remains
9. demonstrated impressive reasoning
10. Prompting Test-Time Scaling

**Топ-10 ключевых слов (синтетические тексты):**
1. large language
2. models
3. large language models
4. language
5. languages
6. model
7. learning approaches
8. automated machine learning
9. neural network models
10. machine learning

**Метрики пересечения:**
- Jaccard Index: 0.020
- Precision: 0.040
- Recall: 0.040
- F1-score: 0.040
- Пересечение: 2 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 2.24
- Синтетические тексты: уникальность 1.000, средняя длина 2.08

#### TEXTRANK

**Топ-10 ключевых слов (человеческие тексты):**
1. reasoning
2. model
3. models
4. training
5. llm
6. time
7. modeling
8. generation
9. data
10. problems

**Топ-10 ключевых слов (синтетические тексты):**
1. model
2. models
3. adaptive
4. learning
5. existing
6. approaches
7. including
8. approach
9. tasks
10. framework

**Метрики пересечения:**
- Jaccard Index: 0.099
- Precision: 0.180
- Recall: 0.180
- F1-score: 0.180
- Пересечение: 9 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.10
- Синтетические тексты: уникальность 1.000, средняя длина 1.00

---

### Information Retrieval

#### Статистика документов

- Человеческих документов: 15
- Синтетических документов: 15

#### NGRAMS

**Топ-10 ключевых слов (человеческие тексты):**
1. model
2. data
3. reasoning
4. provides
5. models
6. dataset
7. introduce
8. yields
9. physics
10. time

**Топ-10 ключевых слов (синтетические тексты):**
1. geometric
2. time
3. model
4. including
5. new
6. informed
7. hierarchical
8. real
9. based
10. multi

**Метрики пересечения:**
- Jaccard Index: 0.064
- Precision: 0.120
- Recall: 0.120
- F1-score: 0.120
- Пересечение: 6 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.04
- Синтетические тексты: уникальность 1.000, средняя длина 1.12

#### YAKE

**Топ-10 ключевых слов (человеческие тексты):**
1. reasoning
2. data
3. made significant progress
4. spatial reasoning explorations
5. understanding indoor scenes
6. spatial reasoning
7. researchers have made
8. autonomous driving
9. current surge
10. made significant

**Топ-10 ключевых слов (синтетические тексты):**
1. late-time Universe
2. Universe
3. anchors
4. cosmological implications
5. matrices
6. rings
7. algebraic
8. multimodal
9. two-qubit entangling gates
10. two-qubit gates

**Метрики пересечения:**
- Jaccard Index: 0.031
- Precision: 0.060
- Recall: 0.060
- F1-score: 0.060
- Пересечение: 3 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 2.20
- Синтетические тексты: уникальность 1.000, средняя длина 2.02

#### TEXTRANK

**Топ-10 ключевых слов (человеческие тексты):**
1. model
2. modeling
3. models
4. task
5. information
6. extensive
7. diverse
8. video
9. knowledge
10. m

**Топ-10 ключевых слов (синтетические тексты):**
1. model
2. significant
3. models
4. modeling
5. sequences
6. new
7. anchors
8. algorithms
9. reconstructions
10. reconstruction

**Метрики пересечения:**
- Jaccard Index: 0.149
- Precision: 0.260
- Recall: 0.260
- F1-score: 0.260
- Пересечение: 13 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.04
- Синтетические тексты: уникальность 1.000, средняя длина 1.04

---

## Общие выводы

### Сравнение методов извлечения ключевых слов

| Метод | Средний Jaccard | Средний F1 | Средняя уникальность |
|-------|----------------|------------|---------------------|
| NGRAMS | 0.100 | 0.180 | 1.000 |
| YAKE | 0.026 | 0.050 | 1.000 |
| TEXTRANK | 0.124 | 0.220 | 1.000 |

### Ключевые наблюдения

1. **Различия в ключевых словах**: Синтетические тексты показывают различия в выборе ключевых слов по сравнению с человеческими.
2. **Эффективность методов**: Различные методы извлечения ключевых слов дают разные результаты.
3. **Тематическая специфичность**: Каждая тема имеет свои характерные ключевые слова.
4. **Потенциал для детекции**: Различия в ключевых словах могут использоваться для выявления AI-сгенерированных текстов.

## Заключение

Эксперимент показал, что анализ ключевых слов может быть эффективным методом для различения человеческих и AI-сгенерированных текстов. Различные методы извлечения ключевых слов дают дополнительные возможности для анализа и могут быть объединены для повышения точности детекции.

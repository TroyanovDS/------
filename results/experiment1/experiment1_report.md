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
3. time
4. llm
5. based
6. using
7. existing
8. environments
9. enables
10. training

**Топ-10 ключевых слов (синтетические тексты):**
1. framework
2. models
3. model
4. learning
5. focus
6. language
7. efficient
8. time
9. task
10. neural

**Метрики пересечения:**
- Jaccard Index: 0.190
- Precision: 0.320
- Recall: 0.320
- F1-score: 0.320
- Пересечение: 16 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.06
- Синтетические тексты: уникальность 1.000, средняя длина 1.02

#### YAKE

**Топ-10 ключевых слов (человеческие тексты):**
1. datasets remains laborious
2. curating large reasoning
3. large reasoning datasets
4. Large language models
5. reasoning datasets remains
6. demonstrated impressive reasoning
7. Prompting Test-Time Scaling
8. Large language
9. curating large
10. laborious and resource-intensive

**Топ-10 ключевых слов (синтетические тексты):**
1. large language
2. large language models
3. learning approaches
4. automated machine learning
5. machine learning
6. distributed computing
7. study introduces
8. language processing conferences
9. top-tier computer vision
10. multimodal fusion techniques

**Метрики пересечения:**
- Jaccard Index: 0.031
- Precision: 0.060
- Recall: 0.060
- F1-score: 0.060
- Пересечение: 3 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 2.52
- Синтетические тексты: уникальность 1.000, средняя длина 2.26

#### TEXTRANK

**Топ-10 ключевых слов (человеческие тексты):**
1. reasoning
2. models
3. model
4. training
5. modeling
6. generation
7. learning
8. data
9. problems
10. problem

**Топ-10 ключевых слов (синтетические тексты):**
1. model
2. models
3. approaches
4. learning
5. language
6. adaptive
7. increase
8. bound
9. existing
10. including

**Метрики пересечения:**
- Jaccard Index: 0.075
- Precision: 0.140
- Recall: 0.140
- F1-score: 0.140
- Пересечение: 7 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.06
- Синтетические тексты: уникальность 1.000, средняя длина 1.06

---

### Information Retrieval

#### Статистика документов

- Человеческих документов: 15
- Синтетических документов: 15

#### NGRAMS

**Топ-10 ключевых слов (человеческие тексты):**
1. data
2. models
3. reasoning
4. dataset
5. model
6. language
7. retrieval
8. scale
9. spatial
10. knowledge

**Топ-10 ключевых слов (синтетические тексты):**
1. framework
2. geometric
3. real
4. proposed
5. multi
6. anchors
7. universe
8. cosmological
9. reconstruction
10. distance

**Метрики пересечения:**
- Jaccard Index: 0.136
- Precision: 0.240
- Recall: 0.240
- F1-score: 0.240
- Пересечение: 12 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.00
- Синтетические тексты: уникальность 1.000, средняя длина 1.10

#### YAKE

**Топ-10 ключевых слов (человеческие тексты):**
1. made significant progress
2. spatial reasoning explorations
3. understanding indoor scenes
4. spatial reasoning
5. researchers have made
6. autonomous driving
7. current surge
8. made significant
9. significant progress
10. progress in understanding

**Топ-10 ключевых слов (синтетические тексты):**
1. late-time Universe
2. anchors
3. matrices
4. rings
5. two-qubit entangling gates
6. gate
7. two-qubit
8. kinematic anchors
9. kinematic
10. kinematic reconstructions

**Метрики пересечения:**
- Jaccard Index: 0.000
- Precision: 0.000
- Recall: 0.000
- F1-score: 0.000
- Пересечение: 0 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 2.42
- Синтетические тексты: уникальность 1.000, средняя длина 2.10

#### TEXTRANK

**Топ-10 ключевых слов (человеческие тексты):**
1. models
2. model
3. modeling
4. information
5. learning
6. extensive
7. diverse
8. video
9. knowledge
10. data

**Топ-10 ключевых слов (синтетические тексты):**
1. model
2. significant
3. models
4. modeling
5. new
6. anchors
7. algorithms
8. reconstructions
9. reconstruction
10. expansion

**Метрики пересечения:**
- Jaccard Index: 0.163
- Precision: 0.280
- Recall: 0.280
- F1-score: 0.280
- Пересечение: 14 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.06
- Синтетические тексты: уникальность 1.000, средняя длина 1.20

---

## Общие выводы

### Сравнение методов извлечения ключевых слов

| Метод | Средний Jaccard | Средний F1 | Средняя уникальность |
|-------|----------------|------------|---------------------|
| NGRAMS | 0.163 | 0.280 | 1.000 |
| YAKE | 0.015 | 0.030 | 1.000 |
| TEXTRANK | 0.119 | 0.210 | 1.000 |

### Ключевые наблюдения

1. **Различия в ключевых словах**: Синтетические тексты показывают различия в выборе ключевых слов по сравнению с человеческими.
2. **Эффективность методов**: Различные методы извлечения ключевых слов дают разные результаты.
3. **Тематическая специфичность**: Каждая тема имеет свои характерные ключевые слова.
4. **Потенциал для детекции**: Различия в ключевых словах могут использоваться для выявления AI-сгенерированных текстов.

## Заключение

Эксперимент показал, что анализ ключевых слов может быть эффективным методом для различения человеческих и AI-сгенерированных текстов. Различные методы извлечения ключевых слов дают дополнительные возможности для анализа и могут быть объединены для повышения точности детекции.

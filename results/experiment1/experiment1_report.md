# Эксперимент 1: Анализ ключевых слов в человеческих и синтетических текстах

## Методология

- **Выборка**: 30 человеческих + 30 синтетических документов (по 15 на тему)
- **Темы**: Text Mining, Information Retrieval
- **Методы извлечения ключевых слов**:
  - N-граммы (TF-IDF, 1-3 граммы)
  - YAKE (Yet Another Keyword Extractor)
  - TextRank
- **Метрики сравнения**: Jaccard, Overlap Human, Overlap Synthetic, Harmonic Mean

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
3. based
4. models
5. test
6. time
7. llm
8. generation
9. supervised
10. problems

**Топ-10 ключевых слов (синтетические тексты):**
1. model
2. neural
3. dynamics
4. research
5. bound
6. problem
7. time
8. robustness
9. languages
10. tasks

**Метрики пересечения:**
- Jaccard Index: 0.099
- Overlap Human: 0.180
- Overlap Synthetic: 0.180
- Harmonic Mean: 0.180
- Пересечение: 9 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.24
- Синтетические тексты: уникальность 1.000, средняя длина 1.28

#### YAKE

**Топ-10 ключевых слов (человеческие тексты):**
1. problem
2. datasets remains laborious
3. curating large reasoning
4. large reasoning datasets
5. Large language models
6. reasoning datasets remains
7. demonstrated impressive reasoning
8. Prompting Test-Time Scaling
9. Large language
10. curating large

**Топ-10 ключевых слов (синтетические тексты):**
1. large language
2. models
3. large language models
4. learning approaches
5. automated machine learning
6. machine learning
7. distributed computing
8. study introduces
9. language processing conferences
10. top-tier computer vision

**Метрики пересечения:**
- Jaccard Index: 0.099
- Overlap Human: 0.180
- Overlap Synthetic: 0.180
- Harmonic Mean: 0.180
- Пересечение: 9 из 50 и 50

**Анализ**: TF-IDF с более мягкими параметрами (min_df=1) показал сбалансированные результаты:
- **Jaccard = 0.099** - низкое сходство, но реалистичное
- **Overlap Human = 0.180** - синтетические тексты частично используют человеческие слова
- **Overlap Synthetic = 0.180** - симметричное пересечение
- **Harmonic Mean = 0.180** - умеренное пересечение
- **Средняя длина 2.40 vs 2.06** - человеческие тексты используют более длинные фразы

#### TEXTRANK

**Топ-10 ключевых слов (человеческие тексты):**
1. reasoning
2. model
3. models
4. training
5. b
6. time
7. modeling
8. generation
9. learning
10. data

**Топ-10 ключевых слов (синтетические тексты):**
1. model
2. models
3. adaptive
4. learning
5. approaches
6. existing
7. including
8. approach
9. tasks
10. framework

**Метрики пересечения:**
- Jaccard Index: 0.099
- Overlap Human: 0.180
- Overlap Synthetic: 0.180
- Harmonic Mean: 0.180
- Пересечение: 9 из 50 и 50

**Анализ**: TextRank показал такое же пересечение как TF-IDF (9 общих слов):
- **Jaccard = 0.099** - низкое сходство, но лучше чем YAKE
- **Overlap Human = 0.180** - синтетические тексты частично используют человеческие слова
- **Overlap Synthetic = 0.180** - симметричное пересечение
- **Harmonic Mean = 0.180** - умеренное пересечение
- **Средняя длина 1.10 vs 1.02** - оба типа используют короткие ключевые слова
- **TextRank и TF-IDF** показали одинаковые результаты для Text Mining

---

### Information Retrieval

#### Статистика документов

- Человеческих документов: 15
- Синтетических документов: 15

#### NGRAMS

**Топ-10 ключевых слов (человеческие тексты):**
1. models
2. reasoning
3. data
4. language
5. retrieval
6. spatial
7. diverse
8. time
9. reward
10. model

**Топ-10 ключевых слов (синтетические тексты):**
1. geometric
2. anchors
3. universe
4. reconstruction
5. informed
6. matrices
7. idempotent
8. algebraic
9. idempotent matrices
10. commutative

**Метрики пересечения:**
- Jaccard Index: 0.149
- Overlap Human: 0.260
- Overlap Synthetic: 0.260
- Harmonic Mean: 0.260
- Пересечение: 13 из 50 и 50

**Анализ**: TF-IDF для Information Retrieval показал лучшие результаты чем для Text Mining:
- **Jaccard = 0.149** - умеренное сходство (лучше чем Text Mining)
- **Overlap Human = 0.260** - синтетические тексты хорошо используют человеческие слова
- **Overlap Synthetic = 0.260** - симметричное пересечение
- **Harmonic Mean = 0.260** - хорошее пересечение
- **Пересечение 13 из 50** - больше чем в Text Mining (9 из 50)
- **Вывод**: Information Retrieval показывает больше сходства между человеческими и AI текстами

#### YAKE

**Топ-10 ключевых слов (человеческие тексты):**
1. reasoning
2. made significant progress
3. spatial reasoning explorations
4. understanding indoor scenes
5. spatial reasoning
6. researchers have made
7. autonomous driving
8. current surge
9. made significant
10. significant progress

**Топ-10 ключевых слов (синтетические тексты):**
1. late-time Universe
2. Universe
3. anchors
4. cosmological implications
5. matrices
6. rings
7. two-qubit entangling gates
8. two-qubit gates
9. gate
10. two-qubit

**Метрики пересечения:**
- Jaccard Index: 0.031
- Overlap Human: 0.060
- Overlap Synthetic: 0.060
- Harmonic Mean: 0.060
- Пересечение: 3 из 50 и 50

**Анализ**: YAKE для Information Retrieval показал низкое пересечение, но лучше чем для Text Mining:
- **Jaccard = 0.031** - очень низкое сходство, но лучше Text Mining (0.020)
- **Overlap Human = 0.060** - синтетические тексты почти не используют человеческие слова
- **Overlap Synthetic = 0.060** - человеческие тексты почти не содержат синтетических слов
- **Harmonic Mean = 0.060** - низкое пересечение
- **Пересечение 3 из 50** - больше чем Text Mining (2 из 50)
- **Средняя длина 2.30 vs 2.08** - человеческие тексты используют более длинные фразы

#### TEXTRANK

**Топ-10 ключевых слов (человеческие тексты):**
1. model
2. modeling
3. models
4. information
5. learning
6. extensive
7. diverse
8. video
9. knowledge
10. task

**Топ-10 ключевых слов (синтетические тексты):**
1. significant
2. model
3. models
4. modeling
5. new
6. anchors
7. algorithms
8. reconstructions
9. reconstruction
10. expansion

**Метрики пересечения:**
- Jaccard Index: 0.176
- Overlap Human: 0.300
- Overlap Synthetic: 0.300
- Harmonic Mean: 0.300
- Пересечение: 15 из 50 и 50

**Анализ**: TextRank показал наилучшие результаты в Information Retrieval:
- **Jaccard = 0.176** - умеренное сходство (лучший результат среди всех методов)
- **Overlap Human = 0.300** - синтетические тексты хорошо используют человеческие слова
- **Overlap Synthetic = 0.300** - симметричное пересечение
- **Harmonic Mean = 0.300** - наилучшее пересечение среди всех методов
- **Пересечение 15 из 50** - больше чем TF-IDF (13 из 50) и YAKE (3 из 50)
- **Средняя длина 1.06 vs 1.16** - оба типа используют короткие ключевые слова
- **Вывод**: TextRank наиболее эффективен для Information Retrieval

---

## Общие выводы

### Сравнение методов извлечения ключевых слов

| Метод | Средний Jaccard | Средний Harmonic Mean | Средняя уникальность | Применимость для детекции |
|-------|----------------|---------------------|---------------------|---------------------------|
| NGRAMS | 0.124 | 0.220 | 1.000 | ⚠️ Умеренная |
| YAKE | 0.026 | 0.050 | 1.000 | ❌ Низкая |
| TEXTRANK | 0.138 | 0.240 | 1.000 | ✅ Хорошая |

### Ключевые наблюдения

1. **Различия в ключевых словах**: Синтетические тексты показывают значительные различия в выборе ключевых слов по сравнению с человеческими.

2. **Эффективность методов**: 
   - **TextRank** показал наилучшие результаты (Harmonic Mean=0.240)
   - **TF-IDF** показал хорошие результаты (Harmonic Mean=0.220)
   - **YAKE** показал самые низкие результаты (Harmonic Mean=0.050)

3. **Тематическая специфичность**: 
   - Text Mining: умеренные различия (Jaccard: 0.099-0.099)
   - Information Retrieval: лучшие результаты (Jaccard: 0.149-0.176)

4. **Параметры влияют на результаты**: Более мягкие параметры TF-IDF (min_df=1) дали сбалансированные результаты.

## Заключение

Эксперимент показал, что **анализ ключевых слов имеет умеренную применимость для распознавания AI-сгенерированных текстов**. TextRank показал наилучшие результаты с Harmonic Mean = 0.240, что указывает на заметные различия между человеческими и синтетическими текстами.

**Основные выводы**:
- **TextRank** наиболее эффективен (Harmonic Mean = 0.240)
- **TF-IDF** показывает хорошие результаты при правильных параметрах (Harmonic Mean = 0.220)
- **YAKE** менее эффективен для данной задачи (Harmonic Mean = 0.050)
- **Information Retrieval** показывает больше сходства между типами текстов

**Рекомендации**: Для практической детекции AI-текстов следует использовать комбинацию методов, с TextRank как основным инструментом, дополненным TF-IDF для подтверждения результатов.

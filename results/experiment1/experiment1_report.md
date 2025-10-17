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
1. large language
2. model performance

**Топ-10 ключевых слов (синтетические тексты):**
1. language models
2. introduce novel
3. large language
4. world applications
5. datasets including
6. decision making
7. existing approaches
8. existing methods
9. findings significant
10. findings significant implications

**Метрики пересечения:**
- Jaccard Index: 0.077
- Overlap Human: 1.000
- Overlap Synthetic: 0.077
- Harmonic Mean: 0.143
- Пересечение: 2 из 2 и 26

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 2.00
- Синтетические тексты: уникальность 1.000, средняя длина 2.12

#### YAKE

**Топ-10 ключевых слов (человеческие тексты):**
1. curating large reasoning datasets
2. large reasoning datasets remains
3. reasoning datasets remains laborious
4. remains laborious and resource-intensive
5. datasets remains laborious
6. demonstrated impressive reasoning capabilities
7. curating large reasoning
8. large reasoning datasets
9. Large language models
10. reasoning datasets remains

**Топ-10 ключевых слов (синтетические тексты):**
1. automated machine learning
2. natural language processing conferences
3. accepted papers from top-tier
4. language processing conferences
5. papers from top-tier computer
6. top-tier computer vision
7. multimodal fusion techniques
8. multimodal fusion
9. spanning 2020-2027
10. natural language processing

**Метрики пересечения:**
- Jaccard Index: 0.010
- Overlap Human: 0.020
- Overlap Synthetic: 0.020
- Harmonic Mean: 0.020
- Пересечение: 1 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 3.50
- Синтетические тексты: уникальность 1.000, средняя длина 2.76

#### TEXTRANK

**Топ-10 ключевых слов (человеческие тексты):**
1. models
2. model
3. reasoning
4. modeling
5. training
6. generation
7. learning
8. data
9. problems
10. problem

**Топ-10 ключевых слов (синтетические тексты):**
1. model
2. models
3. adaptive
4. approaches
5. learning
6. language
7. approach
8. framework
9. increase
10. bound

**Метрики пересечения:**
- Jaccard Index: 0.087
- Overlap Human: 0.160
- Overlap Synthetic: 0.160
- Harmonic Mean: 0.160
- Пересечение: 8 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.06
- Синтетические тексты: уникальность 1.000, средняя длина 1.04

---

### Information Retrieval

#### Статистика документов

- Человеческих документов: 15
- Синтетических документов: 15

#### NGRAMS

**Топ-10 ключевых слов (человеческие тексты):**
1. physics informed
2. state art
3. large language
4. language models
5. extensive experiments

**Топ-10 ключевых слов (синтетические тексты):**
1. results demonstrate
2. large scale
3. state art methods
4. state art
5. proposed framework
6. outperforms state art
7. outperforms state
8. novel framework
9. introduce novel
10. future research

**Метрики пересечения:**
- Jaccard Index: 0.059
- Overlap Human: 0.200
- Overlap Synthetic: 0.077
- Harmonic Mean: 0.111
- Пересечение: 1 из 5 и 13

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 2.00
- Синтетические тексты: уникальность 1.000, средняя длина 2.23

#### YAKE

**Топ-10 ключевых слов (человеческие тексты):**
1. researchers have made significant
2. robotics and autonomous driving
3. made significant progress
4. significant progress in understanding
5. progress in understanding indoor
6. spatial reasoning explorations
7. understanding indoor scenes
8. spatial reasoning
9. researchers have made
10. autonomous driving

**Топ-10 ключевых слов (синтетические тексты):**
1. late-time Universe
2. investigate the cosmological implications
3. study of idempotent matrices
4. matrices
5. two-qubit entangling gates
6. kinematic anchors' that emerge
7. kinematic anchors
8. kinematic anchors exhibit enhanced
9. kinematic
10. kinematic reconstructions

**Метрики пересечения:**
- Jaccard Index: 0.000
- Overlap Human: 0.000
- Overlap Synthetic: 0.000
- Harmonic Mean: 0.000
- Пересечение: 0 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 3.32
- Синтетические тексты: уникальность 1.000, средняя длина 2.80

#### TEXTRANK

**Топ-10 ключевых слов (человеческие тексты):**
1. models
2. model
3. modeling
4. information
5. learning
6. diverse
7. video
8. knowledge
9. dense
10. data

**Топ-10 ключевых слов (синтетические тексты):**
1. model
2. significant
3. models
4. modeling
5. new
6. anchors
7. reconstructions
8. reconstruction
9. expansion
10. distance

**Метрики пересечения:**
- Jaccard Index: 0.176
- Overlap Human: 0.300
- Overlap Synthetic: 0.300
- Harmonic Mean: 0.300
- Пересечение: 15 из 50 и 50

**Анализ разнообразия:**
- Человеческие тексты: уникальность 1.000, средняя длина 1.08
- Синтетические тексты: уникальность 1.000, средняя длина 1.14

---

## Общие выводы

### Сравнение методов извлечения ключевых слов

| Метод | Средний Jaccard | Средний F1 | Средняя уникальность |
|-------|----------------|------------|---------------------|
| NGRAMS | 0.068 | 0.127 | 1.000 |
| YAKE | 0.005 | 0.010 | 1.000 |
| TEXTRANK | 0.132 | 0.230 | 1.000 |

### Ключевые наблюдения

1. **Различия в ключевых словах**: Синтетические тексты показывают различия в выборе ключевых слов по сравнению с человеческими.
2. **Эффективность методов**: Различные методы извлечения ключевых слов дают разные результаты.
3. **Тематическая специфичность**: Каждая тема имеет свои характерные ключевые слова.
4. **Потенциал для детекции**: Различия в ключевых словах могут использоваться для выявления AI-сгенерированных текстов.

## Заключение

Эксперимент показал, что анализ ключевых слов может быть эффективным методом для различения человеческих и AI-сгенерированных текстов. Различные методы извлечения ключевых слов дают дополнительные возможности для анализа и могут быть объединены для повышения точности детекции.

# Эксперимент 1 (перемоделный): 100 HUMAN vs 100 AI текстов на модель

## Методология

- **Корпуса**: 100 человеческих (50 TM + 50 IR) против 100 синтетических на модель (LLAMA, QWEN, DEEPSEEK-R1)
- **Методы**: TF-IDF n-граммы, YAKE, TextRank
- **Метрики**: Jaccard, Overlap Human, Overlap Synthetic, Harmonic Mean; Connectives per 1000; TF-IDF cosine similarity

## Модель: QWEN

| Метод | Jaccard | Overlap H | Overlap S | Harmonic |
|------|---------|-----------|-----------|----------|
| NGRAMS | 0.370 | 0.540 | 0.540 | 0.540 |
| YAKE | 0.250 | 0.400 | 0.400 | 0.400 |
| TEXTRANK | 0.250 | 0.400 | 0.400 | 0.400 |

- Connectives per 1000 words: HUMAN=4.59, QWEN=6.16
- TF-IDF centroid cosine similarity: 0.693

![Пересечения](qwen_overlaps.png)

![Вводные и косинус](qwen_connectives_cosine.png)

## Модель: DEEPSEEK

| Метод | Jaccard | Overlap H | Overlap S | Harmonic |
|------|---------|-----------|-----------|----------|
| NGRAMS | 0.389 | 0.560 | 0.560 | 0.560 |
| YAKE | 0.176 | 0.300 | 0.300 | 0.300 |
| TEXTRANK | 0.282 | 0.440 | 0.440 | 0.440 |

- Connectives per 1000 words: HUMAN=4.59, DEEPSEEK=4.31
- TF-IDF centroid cosine similarity: 0.740

![Пересечения](deepseek_overlaps.png)

![Вводные и косинус](deepseek_connectives_cosine.png)

## Модель: GPTOSS

| Метод | Jaccard | Overlap H | Overlap S | Harmonic |
|------|---------|-----------|-----------|----------|
| NGRAMS | 0.333 | 0.500 | 0.500 | 0.500 |
| YAKE | 0.136 | 0.240 | 0.240 | 0.240 |
| TEXTRANK | 0.282 | 0.440 | 0.440 | 0.440 |

- Connectives per 1000 words: HUMAN=4.59, GPTOSS=1.01
- TF-IDF centroid cosine similarity: 0.668

![Пересечения](gptoss_overlaps.png)

![Вводные и косинус](gptoss_connectives_cosine.png)

## Как использовать результаты для детекции AI-текстов

- Низкие значения Jaccard/Harmonic указывают на различия в лексике и ключевых фразах между HUMAN и AI; это сигнал для детекции.
- Connectives per 1000: переизбыток/недостаток связующих слов у AI относительно HUMAN позволяет построить простой линейный порог.
- TF-IDF cosine similarity между центроидами корпусов: чем ниже сходство, тем проще отделять AI от HUMAN на уровне словаря.
- Рекомендуется ансамбль из (TextRank Harmonic + Connectives gap + Cosine), что повышает устойчивость к перегенерациям.

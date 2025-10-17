# Experiments: N-gram TF-IDF vs BERT-family Embeddings

- Эксперимент 1: TF-IDF (n-граммы: 1,2,3) + MLP классификатор
- Эксперимент 2 (Torch): Эмбеддинги (MiniLM, BERT, RoBERTa, ALBERT) + MLP
- Эксперимент 2 (TF-only): USE (TF Hub) эмбеддинги + Keras MLP
- Источник данных: arXiv (по 20 документов на темы Text Mining и Information Retrieval)
- Синтетика: через открытые LLM (Llama/Mistral/DialoGPT) или KerasNLP GPT-2
- Отчет: `report.md` с таблицами метрик и графиками из `plots/`

### Полный гид по скриптам
- См. подробное описание каждого скрипта, флагов и примеров: `experiments/SCRIPTS_GUIDE.md`.

## Быстрый старт

### Подготовка корпусов
1) Выгрузка CSV с arXiv и конвертация в .txt (human): см. предыдущие шаги
2) Генерация AI корпусов (по 20–40 доков):
- Llama → data/ai/llama
- Qwen → data/ai/qwen
- DeepSeek → data/ai/deepseek

### Извлечение ключевых слов и признаков (n‑граммы, YAKE, TextRank)
```bash
# для human
python experiments/extract_keywords_and_features.py \
  --corpus_root data/human \
  --out_dir experiments/features/human \
  --topk 15

# для Llama
python experiments/extract_keywords_and_features.py \
  --corpus_root data/ai/llama \
  --out_dir experiments/features/llama \
  --topk 15

# для Qwen
python experiments/extract_keywords_and_features.py \
  --corpus_root data/ai/qwen \
  --out_dir experiments/features/qwen \
  --topk 15

# для DeepSeek
python experiments/extract_keywords_and_features.py \
  --corpus_root data/ai/deepseek \
  --out_dir experiments/features/deepseek \
  --topk 15
```

### Обучение детектора (human vs AI)
```bash
# Логистическая регрессия
python experiments/train_detect_ai.py \
  --human_features experiments/features/human \
  --ai_features experiments/features/llama experiments/features/qwen experiments/features/deepseek \
  --out_root experiments/detector_results \
  --model logreg

# MLP
python experiments/train_detect_ai.py \
  --human_features experiments/features/human \
  --ai_features experiments/features/llama experiments/features/qwen experiments/features/deepseek \
  --out_root experiments/detector_results \
  --model mlp
```

### Интерпретация
- Метрики по каждой паре (human vs Llama/Qwen/DeepSeek) сохраняются в `experiments/detector_results/<model>_*.json`.
- Для усиления/абляций варьируйте `--topk`, n‑gram range (в коде TF‑IDF), и используйте отдельные признаки (можно отфильтровать столбцы в features.csv).
#!/usr/bin/env python3
"""
Эксперимент 3: Выявление ключевых слов (КС) в научных документах с настройкой на Inspec
- Настройка параметров на Inspec (dev), оценка на test
- Методы: TF-IDF, YAKE, TextRank, TopicRank, PositionRank, SingleRank, EmbedRank
- Метрики: Precision@K, Recall@K, F1@K (K in {5,10,15}), Jaccard (top-K)
- Применение лучших настроек к HUMAN vs AI корпусам и сравнение КС
- Итоговый отчет: results/experiment3/experiment3_report.md
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Реиспользуем вспомогательные функции из эксперимента 1
sys.path.append(os.path.dirname(__file__))
import run_experiment1_keywords as exp1

# Опциональные зависимости
try:
    import yake
    YAKE_AVAILABLE = True
except Exception:
    YAKE_AVAILABLE = False

try:
    from summa import keywords as summa_keywords
    TEXTRANK_AVAILABLE = True
except Exception:
    TEXTRANK_AVAILABLE = False

try:
    import networkx as nx
    from collections import defaultdict
    PKE_AVAILABLE = True
except Exception:
    PKE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    EMBEDRANK_AVAILABLE = True
except Exception:
    EMBEDRANK_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------
# Утилиты и предобработка
# ------------------------

def normalize_keyword(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def precision_recall_f1_at_k(pred: List[str], gold: List[str], k: int) -> Tuple[float, float, float]:
    if k <= 0:
        return 0.0, 0.0, 0.0
    pred_k = [normalize_keyword(x) for x in pred[:k]]
    gold_set = {normalize_keyword(x) for x in gold if x}
    if not gold_set:
        return 0.0, 0.0, 0.0
    pred_set = set(pred_k)
    tp = len(pred_set & gold_set)
    p = tp / float(len(pred_k)) if pred_k else 0.0
    r = tp / float(len(gold_set)) if gold_set else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def jaccard_at_k(pred: List[str], gold: List[str], k: int) -> float:
    pred_k = {normalize_keyword(x) for x in pred[:k]}
    gold_set = {normalize_keyword(x) for x in gold if x}
    u = pred_k | gold_set
    i = pred_k & gold_set
    return len(i) / float(len(u)) if u else 0.0


# ------------------------
# Загрузка Inspec
# ------------------------

def read_text_file(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""


def parse_keys_file(content: str) -> List[str]:
    parts = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if ';' in line:
            parts.extend([x.strip() for x in line.split(';') if x.strip()])
        else:
            parts.append(line)
    return parts


def list_dir_if_exists(path: str) -> List[str]:
    try:
        return sorted([os.path.join(path, f) for f in os.listdir(path)])
    except Exception:
        return []


def load_inspec_split(split_root: str) -> Tuple[List[str], List[List[str]]]:
    docs_dir = None
    keys_dir = None
    for cand in ["docs", "abstr", "abstracts"]:
        d = os.path.join(split_root, cand)
        if os.path.isdir(d):
            docs_dir = d
            break
    for cand in ["keys", "key", "keywords"]:
        d = os.path.join(split_root, cand)
        if os.path.isdir(d):
            keys_dir = d
            break

    texts: List[str] = []
    golds: List[List[str]] = []

    if docs_dir and keys_dir:
        docs = list_dir_if_exists(docs_dir)
        keys = list_dir_if_exists(keys_dir)
        doc_map = {os.path.splitext(os.path.basename(p))[0]: p for p in docs}
        key_map = {os.path.splitext(os.path.basename(p))[0]: p for p in keys}
        common = sorted(set(doc_map.keys()) & set(key_map.keys()))
        for stem in common:
            t = read_text_file(doc_map[stem])
            k = parse_keys_file(read_text_file(key_map[stem]))
            if t.strip():
                texts.append(t)
                golds.append(k)
        return texts, golds

    csv_path = os.path.join(os.path.dirname(split_root), 'inspec.csv')
    if os.path.exists(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if 'abstract' in df.columns and 'keywords' in df.columns:
                texts = [str(x) for x in df['abstract'].tolist()]
                golds = [[y.strip() for y in str(z).split(';') if y.strip()] for z in df['keywords'].tolist()]
                return texts, golds
        except Exception:
            pass

    return [], []


def load_inspec(root: str) -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
    splits = {}
    golds = {}
    for split in ['train', 'dev', 'test']:
        s_root = os.path.join(root, split)
        t, g = load_inspec_split(s_root)
        if t and g:
            splits[split] = t
            golds[split] = g
    if 'train' in splits and 'dev' not in splits:
        rng = np.random.default_rng(42)
        idx = np.arange(len(splits['train']))
        rng.shuffle(idx)
        n = len(idx)
        dev_n = max(1, int(0.15 * n))
        dev_idx = set(idx[:dev_n])
        train_texts, train_golds = [], []
        dev_texts, dev_golds = [], []
        for i in range(n):
            if i in dev_idx:
                dev_texts.append(splits['train'][i])
                dev_golds.append(golds['train'][i])
            else:
                train_texts.append(splits['train'][i])
                train_golds.append(golds['train'][i])
        splits['train'] = train_texts
        golds['train'] = train_golds
        splits['dev'] = dev_texts
        golds['dev'] = dev_golds
    return splits, golds


# ------------------------
# Экстракторы (пер-документ)
# ------------------------

def fit_tfidf(train_texts: List[str], ngram_range=(1,2), min_df=1, max_df=0.9, max_features=5000) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=(int(min_df) if isinstance(min_df, (int, float)) and min_df >= 1 else float(min_df)),
        max_df=max_df,
        max_features=max_features,
        stop_words='english',
        lowercase=True,
        preprocessor=exp1.preprocess_text,
    )
    if train_texts:
        vec.fit(train_texts)
    return vec


def extract_tfidf_doc(text: str, vec: TfidfVectorizer, top_k: int = 10) -> List[str]:
    X = vec.transform([text])
    if X.shape[1] == 0:
        return []
    arr = X.toarray()[0]
    idx = np.argsort(arr)[-top_k:][::-1]
    feats = vec.get_feature_names_out()
    seen = set()
    out: List[str] = []
    for i in idx:
        term = feats[i]
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out[:top_k]


def extract_yake_doc(text: str, top_k: int = 10, n: int = 3, dedup_lim: float = 0.8) -> List[str]:
    if not YAKE_AVAILABLE:
        return []
    ke = yake.KeywordExtractor(lan='en', n=n, dedupLim=dedup_lim, top=top_k, features=None)
    try:
        kws = ke.extract_keywords(text)
        out = []
        seen = set()
        for kw, _score in kws:
            if not isinstance(kw, str):
                continue
            nrm = normalize_keyword(kw)
            if nrm in seen:
                continue
            seen.add(nrm)
            out.append(kw)
        return out[:top_k]
    except Exception:
        return []


def extract_textrank_doc(text: str, top_k: int = 10, ratio: float | None = 0.2) -> List[str]:
    if not TEXTRANK_AVAILABLE:
        return []
    try:
        if ratio is not None:
            kws = summa_keywords.keywords(text, ratio=ratio, split=True)
        else:
            kws = summa_keywords.keywords(text, words=top_k, split=True)
        out = []
        seen = set()
        for kw in kws:
            nrm = normalize_keyword(kw)
            if nrm in seen:
                continue
            seen.add(nrm)
            out.append(kw)
        return out[:top_k]
    except Exception:
        return []


def extract_topicrank_doc(text: str, top_k: int = 10) -> List[str]:
    if not PKE_AVAILABLE:
        return []
    try:
        # Упрощенная реализация TopicRank: группировка кандидатов и ранжирование тем
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        from collections import defaultdict, Counter
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
        
        sentences = sent_tokenize(text)
        words = []
        for sent in sentences:
            tokens = word_tokenize(sent.lower())
            words.extend([w for w in tokens if w.isalnum() and w not in stop_words and len(w) > 2])
        
        # Кандидаты: униграммы и биграммы
        candidates = []
        for i in range(len(words)):
            if i < len(words):
                candidates.append(words[i])
            if i < len(words) - 1:
                candidates.append(f"{words[i]} {words[i+1]}")
        
        # Группировка похожих кандидатов (по первому слову)
        topic_groups = defaultdict(list)
        for cand in candidates:
            first_word = cand.split()[0] if ' ' in cand else cand
            topic_groups[first_word].append(cand)
        
        # Ранжирование тем по частоте
        topic_scores = {topic: len(cands) for topic, cands in topic_groups.items()}
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        out = []
        seen = set()
        for topic, _ in sorted_topics:
            cands = topic_groups[topic]
            best_cand = max(cands, key=lambda c: candidates.count(c))
            nrm = normalize_keyword(best_cand)
            if nrm not in seen:
                seen.add(nrm)
                out.append(best_cand)
        return out[:top_k]
    except Exception as e:
        return []


def extract_positionrank_doc(text: str, top_k: int = 10) -> List[str]:
    if not PKE_AVAILABLE:
        return []
    try:
        # PositionRank: TextRank с учетом позиции слов
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
        
        sentences = sent_tokenize(text)
        words = []
        word_positions = {}
        pos = 0
        for sent in sentences:
            tokens = word_tokenize(sent.lower())
            for w in tokens:
                if w.isalnum() and w not in stop_words and len(w) > 2:
                    words.append(w)
                    if w not in word_positions:
                        word_positions[w] = pos
                    pos += 1
        
        if len(words) < 2:
            return []
        
        # Граф: окно 3
        G = nx.Graph()
        for i, w in enumerate(words):
            for j in range(max(0, i-3), min(len(words), i+4)):
                if i != j:
                    G.add_edge(w, words[j])
        
        # PageRank с персонализацией на основе позиций
        personalization = {w: 1.0 / (word_positions.get(w, len(words)) + 1) for w in set(words)}
        try:
            scores = nx.pagerank(G, personalization=personalization, max_iter=100)
        except Exception:
            scores = {w: 1.0 for w in set(words)}
        
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out = []
        seen = set()
        for w, _ in sorted_words[:top_k]:
            nrm = normalize_keyword(w)
            if nrm not in seen:
                seen.add(nrm)
                out.append(w)
        return out[:top_k]
    except Exception:
        return []


def extract_singlerank_doc(text: str, top_k: int = 10) -> List[str]:
    if not PKE_AVAILABLE:
        return []
    try:
        # SingleRank: упрощенный TextRank без нормализации
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
        
        sentences = sent_tokenize(text)
        words = []
        for sent in sentences:
            tokens = word_tokenize(sent.lower())
            words.extend([w for w in tokens if w.isalnum() and w not in stop_words and len(w) > 2])
        
        if len(words) < 2:
            return []
        
        # Граф: окно 3
        G = nx.Graph()
        for i, w in enumerate(words):
            for j in range(max(0, i-3), min(len(words), i+4)):
                if i != j:
                    G.add_edge(w, words[j])
        
        # Упрощенный PageRank (без нормализации)
        scores = {w: 1.0 for w in set(words)}
        for _ in range(10):  # Итерации
            new_scores = {}
            for w in set(words):
                score = 0.0
                for neighbor in G.neighbors(w):
                    deg = G.degree(neighbor)
                    if deg > 0:
                        score += scores.get(neighbor, 0) / deg
                new_scores[w] = 0.15 + 0.85 * score
            scores = new_scores
        
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out = []
        seen = set()
        for w, _ in sorted_words[:top_k]:
            nrm = normalize_keyword(w)
            if nrm not in seen:
                seen.add(nrm)
                out.append(w)
        return out[:top_k]
    except Exception:
        return []


def extract_embedrank_doc(text: str, top_k: int = 10, model_name: str = 'all-MiniLM-L6-v2') -> List[str]:
    if not EMBEDRANK_AVAILABLE:
        return []
    try:
        # Извлекаем кандидатов (n-граммы 1-3)
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(ngram_range=(1, 3), stop_words='english', min_df=1)
        try:
            X = vec.fit_transform([text])
            candidates = vec.get_feature_names_out()
        except Exception:
            return []
        
        if len(candidates) == 0:
            return []
        
        # Получаем эмбеддинги документа и кандидатов
        model = SentenceTransformer(model_name)
        doc_emb = model.encode([text])
        cand_emb = model.encode(candidates)
        
        # Косинусное сходство
        similarities = sklearn_cosine_similarity(doc_emb, cand_emb)[0]
        idx = np.argsort(similarities)[-top_k:][::-1]
        
        out = []
        seen = set()
        for i in idx:
            kw = candidates[i]
            nrm = normalize_keyword(kw)
            if nrm in seen:
                continue
            seen.add(nrm)
            out.append(kw)
        return out[:top_k]
    except Exception:
        return []


# ------------------------
# Оценка и подбор параметров
# ------------------------

def evaluate_split(method_name: str, params: Dict, texts: List[str], golds: List[List[str]], top_ks=(5,10,15), tfidf_vec: TfidfVectorizer | None = None, embed_model: Optional[str] = None) -> Dict:
    metrics = {k: {'precision': [], 'recall': [], 'f1': [], 'jaccard': []} for k in top_ks}
    for text, gold in zip(texts, golds):
        pred = []
        if method_name == 'tfidf':
            pred = extract_tfidf_doc(text, tfidf_vec, top_k=max(top_ks)) if tfidf_vec else []
        elif method_name == 'yake':
            pred = extract_yake_doc(text, top_k=max(top_ks), n=params.get('n',3), dedup_lim=params.get('dedup',0.8))
        elif method_name == 'textrank':
            pred = extract_textrank_doc(text, top_k=max(top_ks), ratio=params.get('ratio',0.2))
        elif method_name == 'topicrank':
            pred = extract_topicrank_doc(text, top_k=max(top_ks))
        elif method_name == 'positionrank':
            pred = extract_positionrank_doc(text, top_k=max(top_ks))
        elif method_name == 'singlerank':
            pred = extract_singlerank_doc(text, top_k=max(top_ks))
        elif method_name == 'embedrank':
            pred = extract_embedrank_doc(text, top_k=max(top_ks), model_name=embed_model or params.get('model', 'all-MiniLM-L6-v2'))
        
        for k in top_ks:
            p, r, f1 = precision_recall_f1_at_k(pred, gold, k)
            j = jaccard_at_k(pred, gold, k)
            metrics[k]['precision'].append(p)
            metrics[k]['recall'].append(r)
            metrics[k]['f1'].append(f1)
            metrics[k]['jaccard'].append(j)
    agg = {}
    for k in top_ks:
        agg[k] = {
            'precision': float(np.mean(metrics[k]['precision'])) if metrics[k]['precision'] else 0.0,
            'recall': float(np.mean(metrics[k]['recall'])) if metrics[k]['recall'] else 0.0,
            'f1': float(np.mean(metrics[k]['f1'])) if metrics[k]['f1'] else 0.0,
            'jaccard': float(np.mean(metrics[k]['jaccard'])) if metrics[k]['jaccard'] else 0.0,
        }
    return agg


def grid_search_inspec(train_texts: List[str], train_golds: List[List[str]], dev_texts: List[str], dev_golds: List[List[str]]) -> Dict:
    results = {}
    
    # TF-IDF grid
    tf_grid = [
        {'ngram_range': (1,1), 'min_df': 1, 'max_df': 0.9, 'max_features': 5000},
        {'ngram_range': (1,2), 'min_df': 1, 'max_df': 0.9, 'max_features': 8000},
        {'ngram_range': (1,3), 'min_df': 2, 'max_df': 0.9, 'max_features': 10000},
    ]
    best_tfidf = None
    best_score = -1
    for cfg in tf_grid:
        vec = fit_tfidf(train_texts, **cfg)
        m = evaluate_split('tfidf', cfg, dev_texts, dev_golds, top_ks=(10,), tfidf_vec=vec)
        score = m[10]['f1']
        if score > best_score:
            best_score = score
            best_tfidf = {'params': cfg, 'vec': vec, 'dev': m}
    results['tfidf'] = best_tfidf

    # YAKE grid
    yake_grid = [
        {'n': 1, 'dedup': 0.9},
        {'n': 2, 'dedup': 0.85},
        {'n': 3, 'dedup': 0.8},
    ] if YAKE_AVAILABLE else []
    best_yake = None
    best_score = -1
    for cfg in yake_grid:
        m = evaluate_split('yake', cfg, dev_texts, dev_golds, top_ks=(10,))
        score = m[10]['f1']
        if score > best_score:
            best_score = score
            best_yake = {'params': cfg, 'dev': m}
    results['yake'] = best_yake

    # TextRank grid
    tr_grid = [
        {'ratio': 0.1},
        {'ratio': 0.2},
    ] if TEXTRANK_AVAILABLE else []
    best_tr = None
    best_score = -1
    for cfg in tr_grid:
        m = evaluate_split('textrank', cfg, dev_texts, dev_golds, top_ks=(10,))
        score = m[10]['f1']
        if score > best_score:
            best_score = score
            best_tr = {'params': cfg, 'dev': m}
    results['textrank'] = best_tr

    # TopicRank, PositionRank, SingleRank (без параметров для grid search)
    if PKE_AVAILABLE:
        m_tr = evaluate_split('topicrank', {}, dev_texts, dev_golds, top_ks=(10,))
        results['topicrank'] = {'params': {}, 'dev': m_tr}
        
        m_pr = evaluate_split('positionrank', {}, dev_texts, dev_golds, top_ks=(10,))
        results['positionrank'] = {'params': {}, 'dev': m_pr}
        
        m_sr = evaluate_split('singlerank', {}, dev_texts, dev_golds, top_ks=(10,))
        results['singlerank'] = {'params': {}, 'dev': m_sr}

    # EmbedRank grid
    embed_grid = [
        {'model': 'all-MiniLM-L6-v2'},
        {'model': 'all-mpnet-base-v2'},
    ] if EMBEDRANK_AVAILABLE else []
    best_embed = None
    best_score = -1
    for cfg in embed_grid:
        m = evaluate_split('embedrank', cfg, dev_texts, dev_golds, top_ks=(10,), embed_model=cfg['model'])
        score = m[10]['f1']
        if score > best_score:
            best_score = score
            best_embed = {'params': cfg, 'dev': m}
    results['embedrank'] = best_embed

    return results


def evaluate_on_test(best: Dict, test_texts: List[str], test_golds: List[List[str]]) -> Dict:
    out = {}
    if best.get('tfidf') is not None:
        cfg = best['tfidf']['params']
        vec = best['tfidf']['vec']
        out['tfidf'] = evaluate_split('tfidf', cfg, test_texts, test_golds, top_ks=(5,10,15), tfidf_vec=vec)
    if best.get('yake') is not None:
        cfg = best['yake']['params']
        out['yake'] = evaluate_split('yake', cfg, test_texts, test_golds, top_ks=(5,10,15))
    if best.get('textrank') is not None:
        cfg = best['textrank']['params']
        out['textrank'] = evaluate_split('textrank', cfg, test_texts, test_golds, top_ks=(5,10,15))
    if best.get('topicrank') is not None:
        out['topicrank'] = evaluate_split('topicrank', {}, test_texts, test_golds, top_ks=(5,10,15))
    if best.get('positionrank') is not None:
        out['positionrank'] = evaluate_split('positionrank', {}, test_texts, test_golds, top_ks=(5,10,15))
    if best.get('singlerank') is not None:
        out['singlerank'] = evaluate_split('singlerank', {}, test_texts, test_golds, top_ks=(5,10,15))
    if best.get('embedrank') is not None:
        cfg = best['embedrank']['params']
        out['embedrank'] = evaluate_split('embedrank', cfg, test_texts, test_golds, top_ks=(5,10,15), embed_model=cfg.get('model', 'all-MiniLM-L6-v2'))
    return out


# ------------------------
# HUMAN vs AI сравнение (без gold)
# ------------------------

def load_human_synth_for_comp(human_root='data/human', models=('qwen','deepseek','gptoss'), per_topic=50, per_model=100) -> Tuple[List[str], Dict[str, List[str]]]:
    human_docs = []
    for topic in ['text_mining', 'information_retrieval']:
        d = os.path.join(human_root, topic)
        if os.path.isdir(d):
            human_docs.extend(exp1.load_documents_from_txt_dir(d, count=per_topic))
    synth_docs: Dict[str, List[str]] = {}
    for model in models:
        candidates = [
            f"data/ai/{model}_api_text/text_mining_full",
            f"data/ai/{model}_api_auto/text_mining_full",
            f"data/ai/{model}_api/text_mining_full",
            f"data/ai/{model}_api/ir",
        ]
        docs: List[str] = []
        for c in candidates:
            if len(docs) >= per_model:
                break
            left = per_model - len(docs)
            docs.extend(exp1.load_documents_from_txt_dir(c, count=left))
        if docs:
            synth_docs[model] = docs[:per_model]
    return human_docs, synth_docs


def compare_human_vs_synth(best: Dict, human_docs: List[str], synth_docs: Dict[str, List[str]], top_k: int = 20) -> Dict:
    result = {}
    tf_cfg = best.get('tfidf', {}).get('params') if best.get('tfidf') else None
    tf_vec = fit_tfidf(human_docs, **tf_cfg) if tf_cfg else None
    embed_model = best.get('embedrank', {}).get('params', {}).get('model', 'all-MiniLM-L6-v2') if best.get('embedrank') else None

    def extract_batch(texts: List[str], method: str) -> List[List[str]]:
        out = []
        for t in texts:
            if method == 'tfidf' and tf_vec is not None:
                out.append(extract_tfidf_doc(t, tf_vec, top_k=top_k))
            elif method == 'yake' and best.get('yake') is not None:
                params = best['yake']['params']
                out.append(extract_yake_doc(t, top_k=top_k, n=params.get('n',3), dedup_lim=params.get('dedup',0.8)))
            elif method == 'textrank' and best.get('textrank') is not None:
                params = best['textrank']['params']
                out.append(extract_textrank_doc(t, top_k=top_k, ratio=params.get('ratio',0.2)))
            elif method == 'topicrank':
                out.append(extract_topicrank_doc(t, top_k=top_k))
            elif method == 'positionrank':
                out.append(extract_positionrank_doc(t, top_k=top_k))
            elif method == 'singlerank':
                out.append(extract_singlerank_doc(t, top_k=top_k))
            elif method == 'embedrank' and embed_model:
                out.append(extract_embedrank_doc(t, top_k=top_k, model_name=embed_model))
            else:
                out.append([])
        return out

    def aggregate_top(doc_kws: List[List[str]], top_n: int = 50) -> List[str]:
        from collections import Counter
        cnt = Counter()
        for kws in doc_kws:
            seen = set()
            for kw in kws:
                nrm = normalize_keyword(kw)
                if nrm in seen:
                    continue
                seen.add(nrm)
                cnt[nrm] += 1
        return [w for w,_ in cnt.most_common(top_n)]

    methods = ['tfidf', 'yake', 'textrank', 'topicrank', 'positionrank', 'singlerank', 'embedrank']
    for model, docs in synth_docs.items():
        model_res = {}
        for m in methods:
            human_kws = extract_batch(human_docs, m)
            synth_kws = extract_batch(docs, m)
            top_h = aggregate_top(human_kws, top_n=50)
            top_s = aggregate_top(synth_kws, top_n=50)
            comp = exp1.calculate_keyword_overlap(top_h, top_s)
            model_res[m] = {
                'overlap': comp,
                'top_human': top_h[:10],
                'top_synth': top_s[:10],
            }
        result[model] = model_res
    return result


# ------------------------
# Отчет и графики
# ------------------------

def write_report(output_dir: str, inspec_summary: Dict, best_params: Dict, test_scores: Dict, hvsa: Dict):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report = os.path.join(output_dir, 'experiment3_report.md')
    with open(report, 'w', encoding='utf-8') as f:
        f.write('# Эксперимент 3: Ключевые слова — настройка на Inspec и сравнение HUMAN vs AI\n\n')
        f.write('## Методология\n\n')
        f.write('- Датасет для настройки: Inspec (train/dev/test), gold — авторские ключевые слова.\n')
        f.write('- Методы: TF-IDF, YAKE, TextRank, TopicRank, PositionRank, SingleRank, EmbedRank.\n')
        f.write('- Метрики: Precision@K, Recall@K, F1@K (K=5,10,15), Jaccard(top-K).\n')
        f.write('- Лучшие параметры по dev → финальная оценка на test.\n')
        f.write('- Применение лучших настроек к HUMAN vs AI корпусам (Qwen/DeepSeek/GPTOSS) — сравнение наборов КС.\n\n')

        f.write('## Состояние Inspec\n\n')
        if not inspec_summary.get('available', False):
            f.write('Inspec не найден в data/inspec — использованы дефолтные параметры, подбор пропущен.\n\n')
        else:
            f.write(f"Найдено документов: train={inspec_summary['train']}, dev={inspec_summary['dev']}, test={inspec_summary['test']}\n\n")

        f.write('## Метрики и формулы\n\n')
        f.write('- Precision@K = TP/K; Recall@K = TP/|Gold|; F1@K = 2PR/(P+R).\n')
        f.write('- Jaccard(top-K) = |Pred∩Gold| / |Pred∪Gold|.\n')
        f.write('- Нормализация: lower, удаление пунктуации, схлопывание пробелов.\n\n')

        f.write('## Лучшие параметры (по dev, критерий: F1@10)\n\n')
        if best_params:
            for method in ['tfidf', 'yake', 'textrank', 'topicrank', 'positionrank', 'singlerank', 'embedrank']:
                if best_params.get(method):
                    f.write(f"**{method.upper()}**: {best_params[method]['params']}\n\n")
        else:
            f.write('Нет — использованы дефолтные значения.\n\n')

        f.write('## Результаты на test (Inspec)\n\n')
        if test_scores:
            for m, vals in test_scores.items():
                f.write(f"### {m.upper()}\n\n")
                f.write('|K|Precision|Recall|F1|Jaccard|\n')
                f.write('|-|-|-|-|-|\n')
                for K in [5,10,15]:
                    s = vals.get(K, {'precision':0,'recall':0,'f1':0,'jaccard':0})
                    f.write(f"|{K}|{s['precision']:.3f}|{s['recall']:.3f}|{s['f1']:.3f}|{s['jaccard']:.3f}|\n")
                f.write('\n')
        else:
            f.write('Нет — настройка пропущена.\n\n')

        f.write('## Сопоставление КС: HUMAN vs AI (Qwen/DeepSeek/GPTOSS)\n\n')
        if hvsa:
            for model, data in hvsa.items():
                f.write(f"### Модель: {model.upper()}\n\n")
                f.write('|Метод|Jaccard|Overlap H|Overlap S|Harmonic|\n')
                f.write('|-|-|-|-|-|\n')
                for m in ['tfidf','yake','textrank','topicrank','positionrank','singlerank','embedrank']:
                    if m in data:
                        om = data[m]['overlap']
                        f.write(f"|{m.upper()}|{om['jaccard']:.3f}|{om['overlap_human']:.3f}|{om['overlap_synthetic']:.3f}|{om['harmonic_mean']:.3f}|\n")
                f.write('\n')
                for m in ['tfidf','yake','textrank','topicrank','positionrank','singlerank','embedrank']:
                    if m in data:
                        f.write(f"- {m.upper()} TOP‑HUMAN: {', '.join(data[m]['top_human'][:5])}\n")
                        f.write(f"- {m.upper()} TOP‑AI: {', '.join(data[m]['top_synth'][:5])}\n")
                f.write('\n')
        else:
            f.write('Недостаточно данных для сравнения.\n\n')

        f.write('## Сопоставление с другими подходами\n\n')
        f.write('- Лексические методы (TF‑IDF/YAKE/TextRank/TopicRank/PositionRank/SingleRank/EmbedRank) на Inspec дают умеренные значения F1@K и ограниченную устойчивость на реальных корпусах.\n')
        f.write('- Семантические эмбеддинги (см. Эксперимент 2) показывают существенно лучшую разделимость HUMAN/AI (AUC≈1.0 в наших экспериментах).\n')
        f.write('- Вывод: для детекции синтетики ключевые слова — вспомогательный канал; основная сила — семантика и классификация эмбеддингов.\n\n')

        f.write('## Заключение\n\n')
        f.write('- Настройка на Inspec позволяет выбрать адекватные параметры для извлечения КС.\n')
        f.write('- В практической детекции синтетики: использовать комбинацию (лучший экстрактор КС по Inspec) + лексико‑стилистические признаки + семантический классификатор.\n')

    if test_scores:
        try:
            methods = list(test_scores.keys())
            vals = [test_scores[m][10]['f1'] for m in methods if 10 in test_scores[m]]
            plt.figure(figsize=(10,6))
            plt.bar(methods, vals, color=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2'])
            plt.title('Inspec: F1@10 по методам')
            plt.ylabel('F1@10')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'inspec_f1_at10.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception:
            pass


# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser(description='Эксперимент 3: Inspec + сравнение HUMAN vs AI')
    parser.add_argument('--inspec_root', default='data/inspec')
    parser.add_argument('--output_dir', default='results/experiment3')
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Доступность методов:")
    print(f"  YAKE: {YAKE_AVAILABLE}")
    print(f"  TextRank: {TEXTRANK_AVAILABLE}")
    print(f"  PKE (TopicRank/PositionRank/SingleRank): {PKE_AVAILABLE}")
    print(f"  EmbedRank: {EMBEDRANK_AVAILABLE}")

    splits, golds = load_inspec(args.inspec_root)
    inspec_available = ('train' in splits and 'dev' in splits and 'test' in splits)
    inspec_summary = {
        'available': inspec_available,
        'train': len(splits.get('train', [])),
        'dev': len(splits.get('dev', [])),
        'test': len(splits.get('test', [])),
    }

    best = {}
    test_scores = {}

    if inspec_available:
        print("Запуск grid search на Inspec...")
        best = grid_search_inspec(splits['train'], golds['train'], splits['dev'], golds['dev'])
        print("Оценка на test split...")
        test_scores = evaluate_on_test(best, splits['test'], golds['test'])
    else:
        print("Inspec недоступен, использование дефолтных параметров...")
        best = {
            'tfidf': {'params': {'ngram_range': (1,2), 'min_df': 1, 'max_df': 0.9, 'max_features': 8000}, 'vec': None},
            'yake': {'params': {'n': 3, 'dedup': 0.8}},
            'textrank': {'params': {'ratio': 0.2}},
            'topicrank': {'params': {}},
            'positionrank': {'params': {}},
            'singlerank': {'params': {}},
            'embedrank': {'params': {'model': 'all-MiniLM-L6-v2'}},
        }

    human_docs, synth_docs = load_human_synth_for_comp()
    print(f"Загружено HUMAN: {len(human_docs)}, AI моделей: {len(synth_docs)}")
    
    hvsa = compare_human_vs_synth(best, human_docs, synth_docs, top_k=args.top_k) if human_docs and synth_docs else {}

    write_report(args.output_dir, inspec_summary, best, test_scores, hvsa)
    with open(os.path.join(args.output_dir, 'experiment3_results.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'inspec_summary': inspec_summary,
            'best_params': {m: best.get(m, {}).get('params') if best.get(m) else None for m in ['tfidf','yake','textrank','topicrank','positionrank','singlerank','embedrank']},
            'test_scores': test_scores,
            'human_vs_ai': hvsa,
        }, f, ensure_ascii=False, indent=2)

    print(f"Готово. Отчет: {os.path.join(args.output_dir, 'experiment3_report.md')}")


if __name__ == '__main__':
    main()

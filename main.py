#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Único Autocontenido: Pipeline Completo CDMX (Limpieza, XidooTopic, Sentimientos, Tópicos, Outliers y Tablero HTML).
No importa ni depende de ningún otro módulo local de src/.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ============================================================================
# SECCIÓN INTEGRADA: src/embedding_utils.py
# ============================================================================
"""
Utilidades de Embeddings y Modelos Offline usando XidooTopic.
Integra HFencoder, MeanPooling y TextEmbedder de xidootopic con soporte
para ejecución 100% offline (local_files_only), aceleración CUDA y gestión
eficiente de memoria RAM/VRAM.
"""
import gc
import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from xidootopic import TextEmbedder, HFencoder, MeanPooling
from keybert.backend import BaseEmbedder


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_device():
    """Retorna 'cuda' si hay GPU disponible, de lo contrario 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def free_memory(*objs):
    """Libera referencias pesadas, recolecta basura y limpia caché CUDA."""
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_rss_mb():
    """Retorna el uso actual de RAM (RSS) del proceso en MB."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return round(kb / 1024.0, 1)
    except Exception:
        return 0.0
    return 0.0


def load_hf_model_offline_first(model_name=DEFAULT_EMBEDDING_MODEL):
    """
    Carga tokenizer y modelo HuggingFace priorizando caché local (offline)
    para garantizar ejecución autónoma sin conexión tras instalar dependencias.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model_hf = AutoModel.from_pretrained(model_name, local_files_only=True)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_hf = AutoModel.from_pretrained(model_name)
    return tokenizer, model_hf


def build_xidoo_embedder(model_name=DEFAULT_EMBEDDING_MODEL, batch_size=64, max_length=256):
    """
    Construye una instancia de xidootopic.TextEmbedder configurada con
    HFencoder y MeanPooling sobre el dispositivo disponible (GPU/CPU).
    """
    device = get_device()
    print(f"[XidooTopic-Embedder] Cargando '{model_name}' en dispositivo '{device}'...")
    tokenizer, model_hf = load_hf_model_offline_first(model_name)
    model_hf.to(device)
    model_hf.eval()

    encoder = HFencoder(model_hf, tokenizer, device, max_length=max_length)
    pooling = MeanPooling()

    embedder = TextEmbedder(
        encoder=encoder,
        pooling=pooling,
        batch_size=batch_size,
        normalize=True,
        use_prefix=False,
    )
    return embedder


class XidooKeyBERTBackend(BaseEmbedder):
    """
    Adaptador ligero para KeyBERT que hereda de keybert.backend.BaseEmbedder y
    reutiliza el xidootopic.TextEmbedder ya cargado en memoria, evitando duplicar
    pesos del modelo Transformer en RAM/VRAM y sin peticiones de red.
    """

    def __init__(self, embedder: TextEmbedder):
        super().__init__(embedding_model=embedder)
        self.embedder = embedder

    def embed(self, documents, verbose=False):
        docs = [str(d) for d in documents]
        if not docs:
            return np.empty((0, 384), dtype=np.float32)
        return self.embedder.embed_texts(docs).astype(np.float32)


def build_offline_keybert(embedder=None, model_name=DEFAULT_EMBEDDING_MODEL):
    """
    Instancia KeyBERT reutilizando el embedder de XidooTopic si está disponible
    (cero consumo adicional de RAM), o cargando SentenceTransformer offline.
    """
    from keybert import KeyBERT

    if embedder is not None:
        return KeyBERT(model=XidooKeyBERTBackend(embedder))

    from sentence_transformers import SentenceTransformer

    device = get_device()
    try:
        st_model = SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        st_model = SentenceTransformer(model_name, device=device)
    return KeyBERT(model=st_model)


def load_spacy_nlp(lang="es"):
    """
    Carga el pipeline de spaCy correspondiente al idioma ('es', 'en', 'fr').
    """
    import spacy

    model_map = {
        "es": "es_core_news_sm",
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
    }
    spacy_name = model_map.get(lang.lower(), "es_core_news_sm")
    try:
        return spacy.load(spacy_name, disable=["parser", "ner"])
    except Exception:
        installed = spacy.util.get_installed_models()
        if installed:
            return spacy.load(installed[0], disable=["parser", "ner"])
        return spacy.blank(lang.lower() if lang.lower() in model_map else "es")


# ============================================================================
# SECCIÓN INTEGRADA: src/data_loader.py
# ============================================================================
"""
Módulo de carga y consolidación de datos multifuente.
Armoniza las fuentes raw (DMO y UGC) y genera el corpus unificado.
"""
import os
import pandas as pd
import numpy as np


def build_unified_dataset(raw_dir="data/raw", output_path="data/processed/cdmx_unified.csv"):
    """
    Carga las cuatro fuentes raw de CDMX y las consolida en un solo DataFrame estandarizado.
    Columnas generadas:
      - id: Identificador único entero
      - texto: Texto del comentario / publicación
      - fuente: Etiqueta de la fuente ('sectur', 'tripadvisor', 'instagram', 'reddit')
      - tipo_imagen: 'proyectada' (si fuente == 'sectur') o 'percibida' (demás)
    """
    dfs = []
    
    # 1. SECTUR (DMO oficial)
    sectur_file = os.path.join(raw_dir, "sectur.csv")
    if os.path.exists(sectur_file):
        df_sec = pd.read_csv(sectur_file)
        t_col = "texto" if "texto" in df_sec.columns else df_sec.columns[1]
        sec_df = pd.DataFrame({
            "texto": df_sec[t_col].astype(str).str.strip(),
            "fuente": "sectur",
            "tipo_imagen": "proyectada"
        })
        dfs.append(sec_df)
        print(f"[DataLoader] SECTUR (DMO) cargado: {len(sec_df):,} registros.")
    
    # 2. TripAdvisor (UGC reseñas)
    trip_file = os.path.join(raw_dir, "TripAdvisor.csv")
    if os.path.exists(trip_file):
        df_trip = pd.read_csv(trip_file)
        t_col = "texto" if "texto" in df_trip.columns else df_trip.columns[1]
        trip_df = pd.DataFrame({
            "texto": df_trip[t_col].astype(str).str.strip(),
            "fuente": "tripadvisor",
            "tipo_imagen": "percibida"
        })
        dfs.append(trip_df)
        print(f"[DataLoader] TripAdvisor cargado: {len(trip_df):,} registros.")
        
    # 3. Instagram UGC
    insta_file = os.path.join(raw_dir, "instaUGC.csv")
    if os.path.exists(insta_file):
        df_insta = pd.read_csv(insta_file)
        t_col = "description" if "description" in df_insta.columns else df_insta.columns[0]
        insta_df = pd.DataFrame({
            "texto": df_insta[t_col].astype(str).str.strip(),
            "fuente": "instagram",
            "tipo_imagen": "percibida"
        })
        dfs.append(insta_df)
        print(f"[DataLoader] Instagram UGC cargado: {len(insta_df):,} registros.")
        
    # 4. Reddit CDMX
    reddit_file = os.path.join(raw_dir, "redditcdmx.csv")
    if os.path.exists(reddit_file):
        df_red = pd.read_csv(reddit_file)
        reddit_texts = []
        for _, row in df_red.iterrows():
            parts = []
            if pd.notnull(row.get("title")) and str(row.get("title")).strip():
                parts.append(str(row["title"]).strip())
            if pd.notnull(row.get("text")) and str(row.get("text")).strip():
                parts.append(str(row["text"]).strip())
            if pd.notnull(row.get("comments")) and str(row.get("comments")).strip():
                parts.append(str(row["comments"]).strip())
            reddit_texts.append(" ".join(parts).strip())
            
        reddit_df = pd.DataFrame({
            "texto": reddit_texts,
            "fuente": "reddit",
            "tipo_imagen": "percibida"
        })
        dfs.append(reddit_df)
        print(f"[DataLoader] Reddit CDMX cargado: {len(reddit_df):,} registros.")

    if not dfs:
        raise FileNotFoundError(f"No se encontraron archivos CSV válidos en {raw_dir}")

    unified = pd.concat(dfs, ignore_index=True)
    
    # Filtrar textos vacíos o nulos
    unified = unified[unified["texto"].str.len() > 3].reset_index(drop=True)
    unified["id"] = unified.index
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    unified.to_csv(output_path, index=False)
    print(f"[DataLoader] Dataset unificado guardado en: {output_path} ({len(unified):,} registros totales).")
    
    # Generar CSV de ejemplo para demostración rápida (100 filas estratificadas)
    sample_path = os.path.join(os.path.dirname(output_path), "cdmx_sample_example.csv")
    sample_parts = []
    for src in unified["fuente"].unique():
        sub_src = unified[unified["fuente"] == src]
        sample_parts.append(sub_src.sample(min(len(sub_src), 25), random_state=42))
    sample_df = pd.concat(sample_parts, ignore_index=True)
    sample_df.to_csv(sample_path, index=False)
    print(f"[DataLoader] CSV de ejemplo generado en: {sample_path} ({len(sample_df)} registros).")
    
    return unified


def load_dataset(filepath, text_col, source_col):
    """
    Carga un archivo CSV validando la existencia de las columnas obligatorias.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo especificado no existe: {filepath}")
    
    df = pd.read_csv(filepath)
    if text_col not in df.columns:
        raise ValueError(f"Columna de texto '{text_col}' no encontrada en {df.columns.tolist()}")
    if source_col not in df.columns:
        raise ValueError(f"Columna de fuente '{source_col}' no encontrada en {df.columns.tolist()}")
        
    # Limpiar nulos iniciales
    df = df.dropna(subset=[text_col, source_col]).copy()
    df[text_col] = df[text_col].astype(str)
    df[source_col] = df[source_col].astype(str)
    return df


def _resolve_col(df, col_name, filepath):
    """Resuelve el nombre de columna exacto o insensible a mayúsculas."""
    if col_name in df.columns:
        return col_name
    low_map = {c.lower(): c for c in df.columns}
    if col_name.lower() in low_map:
        return low_map[col_name.lower()]
    raise ValueError(f"Columna '{col_name}' no encontrada en {filepath}: {df.columns.tolist()}")


def load_dual_datasets(
    ugc_path,
    ugc_text_col,
    dmo_path,
    dmo_text_col,
    ugc_label=None,
    dmo_label="sectur",
    out_text_col="texto",
    out_source_col="fuente",
):
    """
    Carga uno o varios CSV de UGC y uno o varios CSV de DMO.
    - Todas las fuentes DMO se fusionan bajo una sola etiqueta (`dmo_label`).
    - Cada fuente UGC conserva su propia etiqueta (`ugc_label` o derivada del archivo)
      para permitir el análisis individual por fuente.
    """
    ugc_paths = [ugc_path] if isinstance(ugc_path, str) else list(ugc_path)
    dmo_paths = [dmo_path] if isinstance(dmo_path, str) else list(dmo_path)

    for p in ugc_paths + dmo_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"El archivo especificado no existe: {p}")

    def _expand_cols(cols, n, param_name):
        c_list = [cols] if isinstance(cols, str) else list(cols)
        if len(c_list) == 1:
            return c_list * n
        if len(c_list) != n:
            raise ValueError(f"{param_name} debe tener 1 valor o {n} valores (uno por CSV).")
        return c_list

    ugc_cols = _expand_cols(ugc_text_col, len(ugc_paths), "--ugc_text_col")
    dmo_cols = _expand_cols(dmo_text_col, len(dmo_paths), "--dmo_text_col")

    if ugc_label is None:
        ugc_labels = []
        for p in ugc_paths:
            stem = os.path.splitext(os.path.basename(p))[0].strip().lower()
            if stem == dmo_label.lower():
                stem = f"{stem}_ugc"
            ugc_labels.append(stem)
    else:
        u_list = [ugc_label] if isinstance(ugc_label, str) else list(ugc_label)
        if len(u_list) == 1 and len(ugc_paths) == 1:
            ugc_labels = u_list
        elif len(u_list) == len(ugc_paths):
            ugc_labels = u_list
        else:
            raise ValueError(f"--ugc_label debe tener {len(ugc_paths)} etiquetas (una por cada CSV en --ugc_dataset).")

    frames = []
    for p, col, lbl in zip(ugc_paths, ugc_cols, ugc_labels):
        if lbl.lower() == dmo_label.lower():
            lbl = f"{lbl}_ugc"
        df_raw = pd.read_csv(p, encoding="utf-8-sig")
        res_col = _resolve_col(df_raw, col, p)
        frames.append(pd.DataFrame({
            out_text_col: df_raw[res_col],
            out_source_col: lbl,
        }))

    for p, col in zip(dmo_paths, dmo_cols):
        df_raw = pd.read_csv(p, encoding="utf-8-sig")
        res_col = _resolve_col(df_raw, col, p)
        frames.append(pd.DataFrame({
            out_text_col: df_raw[res_col],
            out_source_col: dmo_label,
        }))

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[out_text_col, out_source_col]).copy()
    df[out_text_col] = df[out_text_col].astype(str)
    df[out_source_col] = df[out_source_col].astype(str)
    return df



# ============================================================================
# SECCIÓN INTEGRADA: src/preprocessor.py
# ============================================================================
"""
Módulo Unificado de Limpieza y Preprocesamiento de Lenguaje Natural (PLN).
Ejecuta en un solo flujo:
  1. Normalización Unicode/HTML, remoción de URLs, @menciones, #hashtags completos,
     puntuación, dígitos y acentos (preservando ñ).
  2. Detección de idioma (ES/EN por trigramas de caracteres), intersección léxica y
     filtro morfológico anti-OCR (separa ruido ilegible sin hardcodear palabras).
  3. Lematización dependiente del idioma detectado (spaCy ES/EN) y eliminación de
     stopwords multilingües (ES + EN) y tokens <= 2 letras.
  4. Corte estricto de longitud (>= 6 palabras útiles DESPUÉS de stopwords) y
     deduplicación exacta sobre texto_limpio.
"""
import html
from pathlib import Path
import os
import re
import unicodedata
from collections import Counter
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xidootopic import TextChunker

BASE_DIR = Path(__file__).resolve().parent
_base_dir = str(BASE_DIR)
_local_nltk = os.path.join(_base_dir, "nltk_data")
if os.path.exists(_local_nltk) and _local_nltk not in nltk.data.path:
    nltk.data.path.insert(0, _local_nltk)

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

LANG_MAP = {
    "es": "spanish",
    "en": "english",
    "fr": "french",
}

CUSTOM_STOPWORDS = {
    "es": {
        "asi", "aqui", "ahi", "si", "no", "tambien", "vez", "veces", "cada",
        "puede", "pueden", "ser", "estar", "tener", "haber", "solo", "solamente",
        "donde", "cuando", "como", "cual", "quien", "mas", "muy", "tan", "bien",
        "uno", "una", "unos", "unas", "todo", "toda", "todos", "todas", "hacia",
        "hasta", "desde", "sobre", "tras", "ir", "ver", "hacer", "dia", "dias",
        # Muletillas, valorativos genéricos y verbos comodín sin carga temática
        "vale", "pena", "valer", "excelente", "bueno", "buena", "increible",
        "hermoso", "hermosa", "bonito", "bonita", "impresionante", "interesante",
        "maravilloso", "maravillosa", "espectacular", "genial", "lindo", "linda",
        "poder", "pasar", "encontrar", "llegar", "tomar", "recomendar", "recomeir",
        "recomendable", "adema", "ademas", "lugar", "lugares", "cosa", "cosas",
        "gran", "grande", "volver", "visitar", "visita",
    },
    "en": {
        "also", "very", "can", "could", "would", "get", "just", "one", "all",
        "from", "into", "onto", "about", "like", "really", "much", "many",
        "day", "days",
    },
    "fr": {
        "aussi", "tres", "plus", "tout", "tous", "toutes", "avoir", "etre",
        "pouvoir", "faire", "bien", "comme", "dans", "pour", "avec",
        "jour", "jours",
    },
}


def _pre_normalize_for_xidoo(text, lang_code="es"):
    """
    Limpieza estándar de PLN sin reglas hardcodeadas:
    decodifica entidades HTML, normaliza Unicode (NFKC), remueve URLs, @menciones,
    #hashtags completos, dígitos, puntuación y símbolos no alfabéticos.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = "".join(
        ch if ch in {"ñ", "Ñ"} else unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
        for ch in text
    )
    text = text.lower()
    text = re.sub(r"[^a-zñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_ocr_noise(norm_text):
    """Detecta ruido morfológico de OCR: tokens <=2 letras, sin vocales o repetitivos."""
    toks = norm_text.split()
    if not toks:
        return True
    short_ratio = sum(1 for t in toks if len(t) <= 2) / len(toks)
    no_vowel = (
        sum(
            1
            for t in toks
            if len(t) >= 2 and not re.search(r"[aeiou]", t) and t not in {"cdmx", "df", "km", "hrs", "mr", "st"}
        )
        / len(toks)
    )
    consec = sum(1 for i in range(1, len(toks)) if toks[i] == toks[i - 1]) / max(len(toks) - 1, 1)
    avg_len = sum(len(t) for t in toks) / len(toks)
    return len(toks) >= 3 and (short_ratio > 0.50 or no_vowel > 0.25 or consec > 0.25 or avg_len < 2.9)


class TextPreprocessor:
    def __init__(self, lang_code="es", use_lemmatization=True, apply_stemming=False, min_words=6):
        self.lang_code = lang_code.lower()
        self.lang_name = LANG_MAP.get(self.lang_code, "spanish")
        self.use_lemmatization = use_lemmatization
        self.apply_stemming = apply_stemming
        self.min_words = min_words
        self.xidoo_chunker = TextChunker(max_palabras=512, overlap=50)

        self.nlp_es = load_spacy_nlp("es") if self.use_lemmatization else None
        self.nlp_en = load_spacy_nlp("en") if self.use_lemmatization else None
        try:
            self.stemmer = SnowballStemmer(self.lang_name)
        except Exception:
            self.stemmer = None

        base_stops = set()
        for lname in ["spanish", "english"]:
            try:
                base_stops.update(stopwords.words(lname))
            except LookupError:
                pass
        if self.lang_name not in {"spanish", "english"}:
            try:
                base_stops.update(stopwords.words(self.lang_name))
            except LookupError:
                pass

        # Respaldo offline integrado: stopwords nativas de spaCy (no depende de nltk_data/)
        try:
            from spacy.lang.es.stop_words import STOP_WORDS as SPACY_ES_STOPS
            from spacy.lang.en.stop_words import STOP_WORDS as SPACY_EN_STOPS
            base_stops.update(SPACY_ES_STOPS)
            base_stops.update(SPACY_EN_STOPS)
        except Exception:
            pass
        if self.nlp_es is not None:
            base_stops.update(self.nlp_es.Defaults.stop_words)
        if self.nlp_en is not None:
            base_stops.update(self.nlp_en.Defaults.stop_words)

        norm_stops = set()
        combined_custom = CUSTOM_STOPWORDS.get("es", set()).union(CUSTOM_STOPWORDS.get("en", set()))
        for w in base_stops.union(combined_custom):
            norm_stops.add(w)
            norm_stops.add(self._strip_accents(w))
        self.stop_words = norm_stops

    @staticmethod
    def _strip_accents(text):
        return (
            text.replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ü", "u")
        )

    def _lemmatize_with_nlp(self, texts, nlp_model):
        token_lists = []
        if nlp_model is not None and self.use_lemmatization:
            for doc in nlp_model.pipe(texts, batch_size=512):
                toks = []
                for token in doc:
                    lemma = token.lemma_.lower().strip() if token.lemma_ else token.text.lower()
                    lemma = self._strip_accents(lemma)
                    if not lemma or len(lemma) <= 2 or not lemma.isalpha():
                        continue
                    if lemma in self.stop_words or self._strip_accents(token.text.lower()) in self.stop_words:
                        continue
                    if self.apply_stemming and self.stemmer is not None:
                        lemma = self.stemmer.stem(lemma)
                    toks.append(lemma)
                token_lists.append(toks)
        else:
            for text in texts:
                toks = []
                for raw_tok in text.split():
                    t = self._strip_accents(raw_tok.lower())
                    if len(t) <= 2 or t in self.stop_words:
                        continue
                    if self.apply_stemming and self.stemmer is not None:
                        t = self.stemmer.stem(t)
                    toks.append(t)
                token_lists.append(toks)
        return token_lists

    def tokenize_texts_by_lang(self, cleaned_texts, langs):
        """Aplica lematización por idioma detectado (ES/EN), stopwords y tokens > 2."""
        out_lists = [[] for _ in range(len(cleaned_texts))]
        es_idx = [i for i, lg in enumerate(langs) if lg != "en"]
        en_idx = [i for i, lg in enumerate(langs) if lg == "en"]

        if es_idx:
            es_toks = self._lemmatize_with_nlp([cleaned_texts[i] for i in es_idx], self.nlp_es)
            for pos, orig_i in enumerate(es_idx):
                out_lists[orig_i] = es_toks[pos]
        if en_idx:
            en_toks = self._lemmatize_with_nlp([cleaned_texts[i] for i in en_idx], self.nlp_en)
            for pos, orig_i in enumerate(en_idx):
                out_lists[orig_i] = en_toks[pos]
        return out_lists


def preprocess_corpus(
    df,
    text_col,
    source_col,
    lang="es",
    min_words=6,
    output_path=None,
    discarded_path="data/processed/cdmx_descartados.csv",
):
    """
    Ejecuta limpieza unificada:
      1. Normalización base + remoción completa de #hashtags y símbolos.
      2. Detección de idioma (ES/EN), intersección de vocabulario y filtro anti-OCR.
      3. Lematización ES/EN + eliminación de stopwords multilingües.
      4. Corte estricto num_tokens >= min_words (6) DESPUÉS de stopwords + deduplicación.
    Retorna: (df_clean, df_descartados)
    """
    preprocessor = TextPreprocessor(lang_code=lang, use_lemmatization=True, apply_stemming=False, min_words=min_words)
    n_raw = len(df)
    print(f"[Preprocesamiento Unificado] Procesando {n_raw:,} documentos (corte >= {min_words} palabras post-stopwords)...")

    raw_orig = df[text_col].fillna("").astype(str).tolist()
    sources = df[source_col].astype(str).tolist()
    norm_texts = [_pre_normalize_for_xidoo(t, lang_code=lang) for t in raw_orig]

    # Vocabulario de referencia construido desde reseñas de visitantes (evita ruido OCR de PDFs)
    doc_freq = Counter()
    for nt, src in zip(norm_texts, sources):
        if src.lower() != "sectur":
            doc_freq.update(set(w for w in nt.split() if len(w) >= 2))
    if not doc_freq:
        for nt in norm_texts:
            doc_freq.update(set(w for w in nt.split() if len(w) >= 2))
    corpus_vocab = {w for w, c in doc_freq.items() if c >= 3}

    # Perfiles de idioma por trigramas de caracteres (ES vs EN)
    es_samples = [nt for nt, src in zip(norm_texts, sources) if src.lower() in {"tripadvisor", "instagram"} and len(nt.split()) >= 5][:2000]
    en_samples = [nt for nt, src in zip(norm_texts, sources) if src.lower() == "reddit" and len(nt.split()) >= 5]
    if not es_samples:
        es_samples = [nt for nt in norm_texts if len(nt.split()) >= 5][:1000]
    if not en_samples:
        en_samples = ["the city museum park great place visit"]

    vec_lang = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=3)
    vec_lang.fit(es_samples + en_samples)
    prof_es = np.asarray(vec_lang.transform(es_samples).mean(axis=0))
    prof_en = np.asarray(vec_lang.transform(en_samples).mean(axis=0))

    V_all = vec_lang.transform(norm_texts)
    sim_es_all = cosine_similarity(V_all, prof_es).ravel()
    sim_en_all = cosine_similarity(V_all, prof_en).ravel()

    # Etapa 1: Auditoría de idioma, vocabulario y anti-OCR
    valid_mask = np.ones(n_raw, dtype=bool)
    overlaps = np.zeros(n_raw, dtype=float)
    lang_scores = np.zeros(n_raw, dtype=float)
    langs = []
    motivos = [""] * n_raw

    for i in range(n_raw):
        toks = [w for w in norm_texts[i].split() if len(w) >= 2]
        n_tok = len(toks)
        common_n = sum(1 for t in toks if t in corpus_vocab) if n_tok > 0 else 0
        ov = (common_n / n_tok) if n_tok > 0 else 0.0
        overlaps[i] = round(ov * 100.0, 1)

        s_es, s_en = float(sim_es_all[i]), float(sim_en_all[i])
        best_sim = max(s_es, s_en)
        lang_scores[i] = round(best_sim, 3)

        if V_all[i].nnz == 0 or best_sim < 0.12:
            lg = "desconocido"
        else:
            lg = "es" if s_es >= s_en else "en"
        langs.append(lg)

        ocr_flag = _is_ocr_noise(norm_texts[i])
        if (
            n_tok == 0
            or (common_n == 0 and lg == "desconocido")
            or (ov < 0.35 and (lg == "desconocido" or best_sim < 0.18))
            or (ocr_flag and ov < 0.70)
        ):
            valid_mask[i] = False
            motivos[i] = "Ilegible / Ruido OCR"

    # Etapa 2: Lematización por idioma + stopwords ES/EN sobre los que pasaron Etapa 1
    kept_idx_1 = np.where(valid_mask)[0]
    texts_for_spacy = [norm_texts[i][:2500] for i in kept_idx_1]
    langs_1 = [langs[i] for i in kept_idx_1]
    token_lists_1 = preprocessor.tokenize_texts_by_lang(texts_for_spacy, langs_1)

    # Etapa 3: Corte estricto num_tokens >= min_words DESPUÉS de stopwords + deduplicación
    clean_rows = []
    seen_clean = set()
    for pos, orig_i in enumerate(kept_idx_1):
        toks = token_lists_1[pos]
        clean_str = " ".join(toks)
        if len(toks) < min_words:
            valid_mask[orig_i] = False
            motivos[orig_i] = f"Menos de {min_words} palabras"
            continue
        if clean_str in seen_clean:
            valid_mask[orig_i] = False
            motivos[orig_i] = "Duplicado"
            continue
        seen_clean.add(clean_str)
        clean_rows.append((orig_i, clean_str, len(toks), langs[orig_i], overlaps[orig_i], lang_scores[orig_i]))

    # Construir df_clean
    sel_indices = [r[0] for r in clean_rows]
    df_clean = df.iloc[sel_indices].copy().reset_index(drop=True)
    df_clean["id"] = df_clean["id"] if "id" in df_clean.columns else np.arange(len(df_clean))
    df_clean["fuente"] = [sources[i] for i in sel_indices]
    df_clean["texto_original"] = [raw_orig[i] for i in sel_indices]
    df_clean["texto_limpio"] = [r[1] for r in clean_rows]
    df_clean["num_tokens"] = [r[2] for r in clean_rows]
    df_clean["idioma_detectado"] = [r[3] for r in clean_rows]
    df_clean["vocab_overlap_pct"] = [r[4] for r in clean_rows]
    df_clean["lang_score"] = [r[5] for r in clean_rows]
    # texto_chunked solo en memoria para Transformer/Embeddings (no se exportará en CSV final)
    df_clean["texto_chunked"] = preprocessor.xidoo_chunker.recortaTodo([norm_texts[i] for i in sel_indices])

    # Construir df_descartados
    disc_idx = np.where(~valid_mask)[0]
    df_descartados = pd.DataFrame({
        "id": [df.iloc[i]["id"] if "id" in df.columns else i for i in disc_idx],
        "fuente": [sources[i] for i in disc_idx],
        "vocab_overlap_pct": [overlaps[i] for i in disc_idx],
        "idioma_detectado": [langs[i] for i in disc_idx],
        "lang_score": [lang_scores[i] for i in disc_idx],
        "motivo_descarte": [motivos[i] for i in disc_idx],
        "texto_original": [raw_orig[i][:220] for i in disc_idx],
    })

    if discarded_path:
        os.makedirs(os.path.dirname(discarded_path), exist_ok=True)
        df_descartados.to_csv(discarded_path, index=False, encoding="utf-8-sig")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

    free_memory(preprocessor)
    print(f"[Preprocesamiento Unificado] Válidos limpios: {len(df_clean):,} | Descartados: {len(df_descartados):,}")
    if len(df_descartados) > 0:
        print(df_descartados["motivo_descarte"].value_counts().to_string())

    return df_clean, df_descartados


# ============================================================================
# SECCIÓN INTEGRADA: src/outlier_detector.py
# ============================================================================
"""
Módulo de Detección y Análisis de Outliers (Valores Atípicos) con XidooTopic.
Integra xidootopic.UMAPReducer + xidootopic.IsolationForestOutlierRemover
sobre embeddings semánticos densos (TextEmbedder), evaluando anomalías de forma
estratificada por fuente (agnóstica a nombres) para no sesgar canales minoritarios,
extrae unigramas, bigramas y trigramas sobre los outliers y retorna los inliers.
"""
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from xidootopic import UMAPReducer, IsolationForestOutlierRemover


def extract_ngrams(texts, n=1, top_k=20):
    """
    Extrae los top_k n-gramas más frecuentes de una lista de textos.
    """
    valid_texts = [str(t).strip() for t in texts if str(t).strip()]
    if not valid_texts:
        return []

    vec = CountVectorizer(ngram_range=(n, n), min_df=1)
    try:
        X = vec.fit_transform(valid_texts)
    except ValueError:
        return []

    sums = np.asarray(X.sum(axis=0)).ravel()
    vocab = vec.vocabulary_
    words_freq = [(word, int(sums[idx])) for word, idx in vocab.items()]
    words_freq.sort(key=lambda x: x[1], reverse=True)

    total_count = sum(freq for _, freq in words_freq) or 1
    return [
        {
            "ngram": word,
            "frecuencia": freq,
            "porcentaje": round((freq / total_count) * 100, 2),
        }
        for word, freq in words_freq[:top_k]
    ]


def _fit_umap_once(embeddings, n_neighbors, n_components, random_state=42):
    """
    Ejecuta xidootopic.UMAPReducer ajustando una sola vez (fit + embedding_)
    y liberando el modelo inmediatamente para ahorrar memoria RAM.
    """
    reducer = UMAPReducer(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=random_state,
    )
    reducer.fit(embeddings)
    coords = np.asarray(reducer.model_.embedding_, dtype=np.float32)
    free_memory(reducer)
    return coords


def detect_and_analyze_outliers(
    df,
    embeddings,
    text_clean_col="texto_limpio",
    text_embed_col="texto_chunked",
    source_col="fuente",
    contamination=0.05,
    random_state=42,
    output_dir="data/processed",
):
    """
    Detecta valores atípicos mediante XidooTopic:
      1. Reducción dimensional UMAP (5D y 2D) con xidootopic.UMAPReducer.
      2. Detección estratificada por fuente con xidootopic.IsolationForestOutlierRemover
         (preserva canales minoritarios y detecta anomalías reales dentro de cada fuente).
      3. Extracción de unigramas, bigramas y trigramas exclusivamente sobre outliers.
    """
    n_docs = len(df)
    print(f"[Outliers - XidooTopic] Analizando {n_docs:,} embeddings ({embeddings.shape[1]}D)...")

    # 1. Reducción UMAP 5D (para Isolation Forest y grafo Leiden)
    if n_docs > 12:
        n_neighbors_5d = min(25, max(2, n_docs - 2))
        n_comp_5d = min(5, max(2, n_docs - 3))
        emb_umap_5d = _fit_umap_once(embeddings, n_neighbors_5d, n_comp_5d, random_state=random_state)
    else:
        emb_umap_5d = embeddings.astype(np.float32)

    # 2. Proyección UMAP 2D (para dispersión interactiva de comentarios)
    if n_docs > 8:
        n_neighbors_2d = min(25, max(2, n_docs - 2))
        emb_umap_2d = _fit_umap_once(embeddings, n_neighbors_2d, 2, random_state=random_state)
    else:
        emb_umap_2d = embeddings[:, :2].astype(np.float32)

    # 3. Detección con xidootopic.IsolationForestOutlierRemover (estratificada por fuente)
    mask_inliers = np.ones(n_docs, dtype=bool)
    scores = np.zeros(n_docs, dtype=np.float32)
    texts_list = df[text_embed_col].tolist() if text_embed_col in df.columns else df[text_clean_col].tolist()

    sources = df[source_col].astype(str).values
    unique_sources = np.unique(sources)

    if len(unique_sources) > 1 and n_docs >= 100:
        for src in unique_sources:
            idx = np.where(sources == src)[0]
            if len(idx) < 20:
                continue
            iso = IsolationForestOutlierRemover(
                contamination=contamination,
                random_state=random_state,
                n_jobs=1,
            )
            sub_texts = [texts_list[i] for i in idx]
            out_res = iso.fit_transform(emb_umap_5d[idx], sub_texts)
            mask_inliers[idx] = np.asarray(out_res["mask"], dtype=bool)
            scores[idx] = np.asarray(out_res["scores"], dtype=np.float32)
            free_memory(iso)
    else:
        iso = IsolationForestOutlierRemover(
            contamination=contamination,
            random_state=random_state,
            n_jobs=1,
        )
        out_res = iso.fit_transform(emb_umap_5d, texts_list)
        mask_inliers = np.asarray(out_res["mask"], dtype=bool)
        scores = np.asarray(out_res["scores"], dtype=np.float32)
        free_memory(iso)

    df_result = df.copy()
    df_result["is_outlier"] = (~mask_inliers).astype(int)
    df_result["outlier_score"] = np.round(scores, 4)
    df_result["coord_x"] = np.round(emb_umap_2d[:, 0], 4)
    df_result["coord_y"] = np.round(emb_umap_2d[:, 1], 4)

    df_outliers = df_result[~mask_inliers].copy().reset_index(drop=True)
    df_inliers = df_result[mask_inliers].copy().reset_index(drop=True)

    emb_inliers = embeddings[mask_inliers].astype(np.float32)
    emb_umap5_inliers = emb_umap_5d[mask_inliers].astype(np.float32)

    print(
        f"[Outliers - XidooTopic] Detectados {len(df_outliers):,} outliers "
        f"({round(len(df_outliers) / max(n_docs, 1) * 100, 2)}%). "
        f"Continúan {len(df_inliers):,} inliers."
    )

    # 4. N-gramas (n=1, 2, 3) exclusivamente sobre outliers
    outlier_texts = df_outliers[text_clean_col].tolist() if len(df_outliers) > 0 else []
    unigrams = extract_ngrams(outlier_texts, n=1, top_k=20)
    bigrams = extract_ngrams(outlier_texts, n=2, top_k=20)
    trigrams = extract_ngrams(outlier_texts, n=3, top_k=20)

    os.makedirs(output_dir, exist_ok=True)
    outliers_csv = os.path.join(output_dir, "cdmx_outliers.csv")
    df_outliers.to_csv(outliers_csv, index=False)

    ngrams_rows = []
    for tipo, items in [("unigrama", unigrams), ("bigrama", bigrams), ("trigrama", trigrams)]:
        for it in items:
            ngrams_rows.append(
                {
                    "tipo": tipo,
                    "expresion": it["ngram"],
                    "frecuencia": it["frecuencia"],
                    "pct": it["porcentaje"],
                }
            )
    ngrams_df = pd.DataFrame(ngrams_rows, columns=["tipo", "expresion", "frecuencia", "pct"])
    ngrams_csv = os.path.join(output_dir, "outlier_ngrams.csv")
    ngrams_df.to_csv(ngrams_csv, index=False)

    print(f"[Outliers - XidooTopic] Evidencias exportadas a: {outliers_csv} y {ngrams_csv}")
    print("\n--- Top 5 Unigramas / Bigramas / Trigramas en Outliers ---")
    for label, grp in [("Unigramas", unigrams), ("Bigramas", bigrams), ("Trigramas", trigrams)]:
        top_str = ", ".join(f"'{g['ngram']}' ({g['frecuencia']})" for g in grp[:5])
        print(f"  * {label}: {top_str}")
    print("----------------------------------------------------------\n")

    free_memory(df_result, emb_umap_5d, emb_umap_2d)

    return (
        df_inliers,
        df_outliers,
        emb_inliers,
        emb_umap5_inliers,
        {"unigrams": unigrams, "bigrams": bigrams, "trigrams": trigrams},
    )


# ============================================================================
# SECCIÓN INTEGRADA: src/sentiment_analyzer.py
# ============================================================================
"""
Módulo de Análisis de Sentimientos Multilingüe (Offline).
Clasifica cada comentario en polaridad binaria ('positivo' vs 'negativo')
mediante el modelo Transformer multilingüe (nlptown/bert-base-multilingual-uncased-sentiment)
cargado desde caché local, con respaldo automático basado en similitud semántica
de embeddings XidooTopic + lexicón multilingüe (es, en, fr).
Preserva la etiqueta de fuente en todo momento.
"""
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

SENTIMENT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

MULTILINGUAL_LEXICON_POS = {
    # Español
    "excelente", "increible", "maravilloso", "hermoso", "bellisimo", "bonito", "lindo",
    "delicioso", "rico", "sabroso", "espectacular", "amable", "limpio", "seguro",
    "recomendado", "recomiendo", "encanto", "fascinante", "magico", "joya", "imprescindible",
    "agradable", "perfecto", "exquisito", "divertido", "autentico", "maravilla", "comodo",
    # Inglés
    "excellent", "amazing", "wonderful", "beautiful", "delicious", "great", "awesome",
    "friendly", "clean", "safe", "recommended", "recommend", "fantastic", "perfect",
    "fun", "authentic", "lovely", "enjoyed", "best", "favorite", "incredible", "worth",
    # Francés
    "excellent", "incroyable", "merveilleux", "magnifique", "beau", "belle", "delicieux",
    "agreable", "propre", "recommande", "parfait", "fantastique", "genial", "super",
}

MULTILINGUAL_LEXICON_NEG = {
    # Español
    "pesimo", "horrible", "terrible", "malo", "mala", "malos", "malas", "sucio",
    "caro", "carisimo", "sobreprecio", "estafa", "robo", "peligro", "peligroso",
    "inseguro", "decepcion", "decepcionante", "demora", "grosero", "ruido", "caos",
    "desorden", "asqueroso", "incomodo", "evitar", "fraude", "lastima", "peor",
    "asco", "queja", "abuso", "deficiente", "cerrado", "abandono", "desorganizado",
    # Inglés
    "terrible", "horrible", "awful", "bad", "worst", "dirty", "expensive", "overpriced",
    "scam", "rip", "dangerous", "unsafe", "disappointing", "disappointed", "rude",
    "noise", "crowded", "chaos", "avoid", "fraud", "waste", "poor", "closed", "rude",
    # Francés
    "terrible", "horrible", "mauvais", "pire", "sale", "cher", "arnaque", "dangereux",
    "decevant", "decu", "attente", "bruit", "eviter", "fraude", "mediocre", "ferme",
}


class TransformerSentimentClassifier:
    """
    Clasificador Transformer multilingüe (ES/EN/FR) 100% offline sobre caché local.
    """

    def __init__(self, model_name=SENTIMENT_MODEL_NAME, batch_size=64, max_length=256):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=True
        ).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()

    def predict_scores(self, texts):
        scores = []
        star_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=self.device)
        for i in range(0, len(texts), self.batch_size):
            batch = [str(t)[:1500] for t in texts[i : i + self.batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1)
                expected_stars = (probs * star_weights).sum(dim=-1)
                # Mapear [1, 5] -> [-1.0, +1.0]
                batch_scores = ((expected_stars - 3.0) / 2.0).cpu().numpy()
                scores.extend(batch_scores.tolist())
        return np.array(scores, dtype=float)


def _fallback_semantic_lexicon_scores(texts, text_clean_list, embedder=None):
    """
    Respaldo completamente offline si el modelo Transformer de sentimiento no está
    previamente cacheado: combina similitud semántica con prototipos de sentimiento
    (usando xidootopic.TextEmbedder) y lexicón multilingüe (es, en, fr).
    """
    lex_scores = []
    for txt in text_clean_list:
        words = str(txt).lower().split()
        pos_c = sum(1 for w in words if w in MULTILINGUAL_LEXICON_POS)
        neg_c = sum(1 for w in words if w in MULTILINGUAL_LEXICON_NEG)
        tot = pos_c + neg_c
        lex_scores.append((pos_c - neg_c) / tot if tot > 0 else 0.05)
    lex_scores = np.array(lex_scores, dtype=float)

    if embedder is None:
        return lex_scores

    pos_anchor = (
        "excelente experiencia hermoso lugar recomendado increíble servicio maravilloso "
        "amazing experience beautiful place highly recommended great value"
    )
    neg_anchor = (
        "pésima experiencia sucio caro estafa mala atención horrible decepción peligroso "
        "terrible experience dirty overpriced scam rude staff disappointing avoid"
    )
    anchor_embs = embedder.embed_texts([pos_anchor, neg_anchor])
    text_embs = embedder.embed_texts([str(t) for t in texts])
    sim_pos = cosine_similarity(text_embs, anchor_embs[0:1]).flatten()
    sim_neg = cosine_similarity(text_embs, anchor_embs[1:2]).flatten()
    sem_scores = np.clip((sim_pos - sim_neg) * 3.0, -1.0, 1.0)
    return 0.6 * lex_scores + 0.4 * sem_scores


def analyze_sentiments(
    df,
    embeddings=None,
    emb_umap5=None,
    text_col="texto_chunked",
    text_clean_col="texto_limpio",
    source_col="fuente",
    threshold=-0.35,
    embedder=None,
    output_path="data/processed/cdmx_sentiment.csv",
):
    """
    Clasifica los comentarios inliers en positivos y negativos preservando
    la etiqueta de fuente y alineando sus embeddings para modelado de tópicos.
    Retorna: (df_res, df_pos, df_neg, emb_pos, emb_neg, umap5_pos, umap5_neg)
    """
    n_docs = len(df)
    print(f"[Sentimiento] Clasificando polaridad de {n_docs:,} comentarios...")
    raw_for_model = (
        df[text_col].tolist() if text_col in df.columns else df["texto_original"].tolist()
    )

    try:
        clf = TransformerSentimentClassifier()
        scores = clf.predict_scores(raw_for_model)
        free_memory(clf)
        print("[Sentimiento] Modelo utilizado: Transformer multilingüe offline (nlptown).")
    except Exception as exc:
        print(f"[Sentimiento] Aviso ({exc}). Usando respaldo semántico-léxico offline...")
        scores = _fallback_semantic_lexicon_scores(
            raw_for_model, df[text_clean_col].tolist(), embedder=embedder
        )

    sentiments = np.where(scores >= threshold, "positivo", "negativo")

    df_res = df.copy()
    df_res["sentimiento"] = sentiments
    df_res["score_polaridad"] = np.round(scores, 4)

    mask_pos = (df_res["sentimiento"] == "positivo").to_numpy()
    mask_neg = ~mask_pos

    df_pos = df_res[mask_pos].copy().reset_index(drop=True)
    df_neg = df_res[mask_neg].copy().reset_index(drop=True)

    emb_pos = embeddings[mask_pos] if embeddings is not None else None
    emb_neg = embeddings[mask_neg] if embeddings is not None else None
    umap5_pos = emb_umap5[mask_pos] if emb_umap5 is not None else None
    umap5_neg = emb_umap5[mask_neg] if emb_umap5 is not None else None

    total = max(len(df_res), 1)
    n_pos = len(df_pos)
    n_neg = len(df_neg)
    print("[Sentimiento] Distribución Global:")
    print(f"  * Positivos: {n_pos:,} ({round(n_pos / total * 100, 1)}%)")
    print(f"  * Negativos: {n_neg:,} ({round(n_neg / total * 100, 1)}%)")
    print("[Sentimiento] Distribución por Fuente:")
    print(df_res.groupby([source_col, "sentimiento"]).size().unstack(fill_value=0))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        to_save = df_res.drop(columns=["tokens"]) if "tokens" in df_res.columns else df_res
        to_save.to_csv(output_path, index=False)
        print(f"[Sentimiento] Resultados exportados a: {output_path}")

    return df_res, df_pos, df_neg, emb_pos, emb_neg, umap5_pos, umap5_neg


# ============================================================================
# SECCIÓN INTEGRADA: src/topic_modeler.py
# ============================================================================
"""
Módulo de Modelado de Tópicos por Sentimiento basado en XidooTopic.
Ejecuta el pipeline grafo-semántico de xidootopic por polaridad (positivos y negativos):
  - xidootopic.KNNGraphBuilder (grafo KNN con FAISS coseno)
  - xidootopic.LeidenResolutionOptimizer + xidootopic.LeidenClusterer
  - xidootopic.ClusterFilter + xidootopic.ClusterMerger (fusión semántica)
  - xidootopic.KeyWordExtractor (KeyBERT + spaCy) + BM25KeywordExtractor
  - xidootopic.RepresentativeDocsExtractor (comentarios más cercanos al centroide)
  - xidootopic.modeling.presentation.TopicFormatter

Incluye salvaguarda automática por umbral mínimo de partición: si un subconjunto
tiene menos de `min_partition_size` documentos, aplica enfoque de frecuencia léxica.
"""
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from xidootopic import (
    UMAPReducer,
    KNNGraphBuilder,
    LeidenResolutionOptimizer,
    LeidenClusterer,
    ClusterFilter,
    ClusterMerger,
    KeyWordExtractor,
    RepresentativeDocsExtractor,
)
from xidootopic.modeling.representation import BM25KeywordExtractor
from xidootopic.modeling.text_processing import get_tokenized_grouped_texts
from xidootopic.modeling.presentation import TopicFormatter


class BoundedKeyWordExtractor(KeyWordExtractor):
    """
    Extiende xidootopic.KeyWordExtractor limitando el máximo de documentos
    representativos por clúster para mantener tiempos ágiles en corpus grandes
    (>20,000 documentos) sin perder fidelidad semántica al centroide.
    """

    def __init__(self, kw_model, stopwords, nlp, top_docs_ratio=0.2, top_n=10, max_rep_docs=15):
        super().__init__(
            kw_model=kw_model,
            stopwords=stopwords,
            nlp=nlp,
            top_docs_ratio=top_docs_ratio,
            top_n=top_n,
        )
        self.max_rep_docs = max_rep_docs

    def _select_representative_docs(self, texts, embeddings):
        docs = super()._select_representative_docs(texts, embeddings)
        if len(docs) > self.max_rep_docs:
            return docs[-self.max_rep_docs :]
        return docs

    def _postprocess_keywords(self, keywords):
        from collections import Counter
        counts = Counter(keywords)
        deduped = []
        seen = set()
        for word, _ in counts.most_common():
            w = str(word).strip().lower()
            if len(w) > 2 and w not in seen:
                seen.add(w)
                deduped.append(w)
        return deduped


def _frequency_fallback(df_subset, emb_subset, subset_name, top_n_words=10,
                        text_clean_col="texto_limpio", text_orig_col="texto_original"):
    """
    Salvaguarda obligatoria por umbral: cuando la partición es demasiado pequeña
    (< min_partition_size), utiliza frecuencia de palabras en lugar de modelado de tópicos.
    """
    n_docs = len(df_subset)
    print(f"  [Salvaguarda] Partición '{subset_name}' ({n_docs} docs) bajo umbral. Activando frecuencia léxica.")
    if n_docs == 0:
        return df_subset.copy(), []

    texts_clean = [str(t) for t in df_subset[text_clean_col].tolist() if str(t).strip()]
    top_words = []
    if texts_clean:
        vec = CountVectorizer(max_features=top_n_words * 2)
        try:
            X = vec.fit_transform(texts_clean)
            counts = np.asarray(X.sum(axis=0)).ravel()
            words = np.array(vec.get_feature_names_out())
            top_words = words[np.argsort(counts)[::-1][:top_n_words]].tolist()
        except ValueError:
            top_words = []

    if emb_subset is not None and len(emb_subset) > 0:
        centroid = emb_subset.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(centroid) + 1e-9
        sims = cosine_similarity(emb_subset, centroid / norm).flatten()
        best_idx = int(np.argmax(sims))
    else:
        sims = np.ones(n_docs, dtype=float)
        best_idx = 0

    rep_text = str(df_subset.iloc[best_idx][text_orig_col])
    rep_source = str(df_subset.iloc[best_idx]["fuente"])
    topic_name = f"Frecuencia_{subset_name.capitalize()}"

    topic_info = [{
        "topico_id": 0,
        "nombre_topico": topic_name,
        "polaridad": subset_name,
        "palabras_clave": top_words,
        "fuentes_origen": df_subset["fuente"].value_counts().to_dict(),
        "fuente_dominante": df_subset["fuente"].value_counts().index[0],
        "n_documentos": n_docs,
        "peso_relativo": 100.0,
        "comentario_representativo": rep_text,
        "fuente_representativa": rep_source,
        "similitud_rep": round(float(sims[best_idx]), 4),
    }]

    df_annotated = df_subset.copy()
    df_annotated["topico_id"] = 0
    df_annotated["nombre_topico"] = topic_name
    df_annotated["similitud_centroide"] = np.round(sims, 4)
    return df_annotated, topic_info


def _cap_clusters_if_needed(embeddings, labels, max_clusters=8):
    """
    Si tras ClusterMerger aún quedan demasiados micro-tópicos para una lectura
    ejecutiva clara, fusiona iterativamente el par de centroides más cercano
    hasta alcanzar a lo sumo `max_clusters`.
    """
    labels_curr = labels.copy()
    while len(np.unique(labels_curr)) > max_clusters:
        unique_c = np.unique(labels_curr)
        centroids = []
        for cid in unique_c:
            c_emb = embeddings[labels_curr == cid].mean(axis=0)
            c_emb = c_emb / (np.linalg.norm(c_emb) + 1e-9)
            centroids.append(c_emb)
        centroids = np.vstack(centroids)
        sim_mat = cosine_similarity(centroids)
        np.fill_diagonal(sim_mat, -np.inf)
        i_max, j_max = np.unravel_index(np.argmax(sim_mat), sim_mat.shape)
        cid_keep = unique_c[min(i_max, j_max)]
        cid_drop = unique_c[max(i_max, j_max)]
        labels_curr[labels_curr == cid_drop] = cid_keep

    # Reindexar 0..K-1 ordenados por tamaño descendente
    unique_c, counts = np.unique(labels_curr, return_counts=True)
    order = unique_c[np.argsort(-counts)]
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[c] for c in labels_curr], dtype=int)


def model_topics_for_subset(
    df_subset,
    emb_subset,
    umap5_subset=None,
    subset_name="positivo",
    lang="es",
    stopwords_list=None,
    kw_model=None,
    nlp_spacy=None,
    min_partition_size=30,
    top_n_words=10,
    max_clusters=8,
    text_clean_col="texto_limpio",
    text_orig_col="texto_original",
):
    """
    Ejecuta modelado de tópicos con componentes de xidootopic sobre una partición
    de sentimiento ('positivo' o 'negativo').
    """
    n_docs = len(df_subset)
    print(f"\n[Tópicos - XidooTopic] Procesando partición '{subset_name}' ({n_docs:,} comentarios)...")

    if n_docs < min_partition_size:
        return _frequency_fallback(
            df_subset,
            emb_subset,
            subset_name=subset_name,
            top_n_words=top_n_words,
            text_clean_col=text_clean_col,
            text_orig_col=text_orig_col,
        )

    # 1. Embeddings reducidos (UMAP 5D) para construir grafo KNN estable
    if umap5_subset is not None and len(umap5_subset) == n_docs:
        emb_red = umap5_subset
    else:
        n_neighbors = min(20, max(2, n_docs - 2))
        n_comp = min(5, max(2, n_docs - 3))
        emb_red = UMAPReducer(
            n_neighbors=n_neighbors, n_components=n_comp, metric="cosine", random_state=42
        ).fit_transform(emb_subset)

    # 2. Construcción del grafo KNN (xidootopic.KNNGraphBuilder)
    k_knn = min(15, max(2, n_docs - 1))
    graph = KNNGraphBuilder(k=k_knn, metrics="cosine", verbose=True).fit_transform(emb_red)

    # 3. Optimización de resolución Leiden + Clustering (xidootopic)
    min_avg = max(5, min(25, n_docs // 10))
    step_res = 0.2 if n_docs > 2000 else 0.1
    opt = LeidenResolutionOptimizer(
        res_min=0.3,
        res_max=1.3,
        step=step_res,
        alpha=0.25,
        min_avg_size=min_avg,
        verbose=False,
        seed=42,
    )
    try:
        _, best_res = opt.fit_predict(graph, emb_red)
        print(f"  * Resolución Leiden óptima encontrada: {best_res:.2f}")
    except ValueError:
        best_res = 0.8
        print(f"  * Aviso: usando resolución Leiden por defecto: {best_res:.2f}")

    clusterer = LeidenClusterer(resolution=float(best_res), use_weights=True, random_state=42, verbose=False)
    labels_raw = clusterer.fit_predict(graph)
    print(f"  * Comunidades Leiden iniciales: {len(np.unique(labels_raw))}")
    free_memory(graph, opt, clusterer)

    # 4. Filtrado de micro-clústeres (xidootopic.ClusterFilter)
    min_c_size = max(3, min(15, n_docs // 35))
    texts_for_kw = (
        df_subset["texto_chunked"].tolist()
        if "texto_chunked" in df_subset.columns
        else df_subset[text_clean_col].tolist()
    )
    c_filter = ClusterFilter(min_cluster_size=min_c_size, reindex=True, verbose=False)
    emb_f, texts_f, labels_f, _, filter_stats = c_filter.fit_transform(emb_subset, texts_for_kw, labels_raw)
    if len(labels_f) == 0 or len(np.unique(labels_f)) == 0:
        # Si todos los clústeres eran menores que min_c_size, conservar labels_raw
        emb_f, texts_f, labels_f = emb_subset, texts_for_kw, labels_raw
    else:
        print(
            f"  * ClusterFilter: {filter_stats['original_clusters']} -> "
            f"{filter_stats['remaining_clusters']} clústeres válidos"
        )

    # 5. Fusión semántica de clústeres (xidootopic.ClusterMerger)
    if len(np.unique(labels_f)) >= 3:
        try:
            merger = ClusterMerger(percentile=85, alpha=0.55, max_iter=50, verbose=False, reindex=True)
            labels_merged = merger.fit(emb_f, labels_f)
        except (IndexError, ValueError):
            labels_merged = labels_f
    else:
        labels_merged = labels_f

    # Consolidación a macro-tópicos interpretables si excede max_clusters
    labels_merged = _cap_clusters_if_needed(emb_f, labels_merged, max_clusters=max_clusters)

    # Calcular centroides finales sobre los clústeres fusionados y reasignar cualquier
    # documento filtrado por ClusterFilter al centroide más cercano para no perder cobertura DMO/UGC
    unique_cids = sorted(np.unique(labels_merged).tolist())
    centroids = []
    for cid in unique_cids:
        c_vec = emb_f[labels_merged == cid].mean(axis=0)
        c_vec = c_vec / (np.linalg.norm(c_vec) + 1e-9)
        centroids.append(c_vec)
    centroid_matrix = np.vstack(centroids)

    # Similitud de TODOS los documentos de la partición vs centroides
    sim_to_centroids = cosine_similarity(emb_subset, centroid_matrix)
    if len(labels_merged) == n_docs:
        labels_all = labels_merged
    else:
        # Conservar etiqueta directa de los no filtrados y reasignar los residuales al centroide más cercano
        valid_mask = np.isin(labels_raw, list(c_filter.valid_clusters))
        labels_all = np.zeros(n_docs, dtype=int)
        labels_all[valid_mask] = labels_merged
        if np.any(~valid_mask):
            labels_all[~valid_mask] = np.argmax(sim_to_centroids[~valid_mask], axis=1)

    doc_sims = sim_to_centroids[np.arange(n_docs), labels_all]

    # 6. Extracción de palabras clave con KeyWordExtractor (KeyBERT + spaCy) y BM25KeywordExtractor
    if kw_model is None:
        kw_model = build_offline_keybert()
    if nlp_spacy is None:
        nlp_spacy = load_spacy_nlp(lang)

    kw_extractor = BoundedKeyWordExtractor(
        kw_model=kw_model,
        stopwords=stopwords_list or [],
        nlp=nlp_spacy,
        top_docs_ratio=0.2,
        top_n=top_n_words,
        max_rep_docs=15,
    )
    keybert_topics = kw_extractor.extract(texts_for_kw, emb_subset, labels_all)

    # Enriquecimiento / respaldo léxico con xidootopic BM25KeywordExtractor
    clean_texts_for_bm25 = [str(t) for t in df_subset[text_clean_col].tolist()]
    grouped_tokens = get_tokenized_grouped_texts(clean_texts_for_bm25, labels_all)
    bm25_scores = BM25KeywordExtractor().extract(grouped_tokens)

    # 7. Extracción de documentos representativos con xidootopic.RepresentativeDocsExtractor
    orig_texts = [str(t) for t in df_subset[text_orig_col].tolist()]
    rep_extractor = RepresentativeDocsExtractor(top_k=3)
    rep_docs_dict = rep_extractor.extract(emb_subset, labels_all, orig_texts)

    # Formateo interno con TopicFormatter de xidootopic
    topic_info = []
    topic_name_map = {}
    formatter_payload = {}

    for cid in unique_cids:
        kb_keys, _ = keybert_topics.get(cid, ([], ""))
        bm25_keys = [w for w, _ in bm25_scores.get(cid, [])[:top_n_words]]

        # Combinar palabras clave de KeyBERT (lematizadas) + BM25 limpio sin duplicados
        merged_keys = []
        seen_k = set()
        for w in list(kb_keys) + list(bm25_keys):
            w_clean = str(w).strip().lower()
            if len(w_clean) > 2 and w_clean not in seen_k:
                if stopwords_list and w_clean in stopwords_list:
                    continue
                seen_k.add(w_clean)
                merged_keys.append(w_clean)
        final_keys = merged_keys[:top_n_words] if merged_keys else bm25_keys[:top_n_words]

        prefix = f"T{cid + 1}_{subset_name[:3].upper()}"
        slug = "_".join(final_keys[:2]) if len(final_keys) >= 2 else (final_keys[0] if final_keys else "tema")
        topic_name = f"{prefix}_{slug}"
        topic_name_map[cid] = topic_name

        mask_c = labels_all == cid
        c_sources = df_subset.loc[mask_c, "fuente"]
        source_counts = c_sources.value_counts().to_dict()
        dominant_source = c_sources.value_counts().index[0] if len(c_sources) > 0 else "N/A"
        n_c = int(np.sum(mask_c))
        pct = round(n_c / max(n_docs, 1) * 100, 2)

        rep_list = rep_docs_dict.get(cid, [])
        rep_text = rep_list[0][0] if rep_list else ""
        rep_sim = round(float(rep_list[0][1]), 4) if rep_list else 0.0

        # Encontrar fuente exacta del documento representativo más cercano
        c_indices = np.where(mask_c)[0]
        best_pos_in_c = int(np.argmax(doc_sims[c_indices]))
        rep_source = str(df_subset.iloc[c_indices[best_pos_in_c]]["fuente"])

        formatter_payload[topic_name] = (final_keys, rep_text[:220])
        topic_info.append(
            {
                "topico_id": int(cid),
                "nombre_topico": topic_name,
                "polaridad": subset_name,
                "palabras_clave": final_keys,
                "fuentes_origen": source_counts,
                "fuente_dominante": dominant_source,
                "n_documentos": n_c,
                "peso_relativo": pct,
                "comentario_representativo": rep_text,
                "fuente_representativa": rep_source,
                "similitud_rep": rep_sim,
            }
        )
        print(f"  * {topic_name} ({pct}% | n={n_c:,} | dom={dominant_source}): {', '.join(final_keys[:6])}")

    # Usar TopicFormatter de xidootopic para validar el bloque formateado
    _kw_list, _formatted_str = TopicFormatter().format(formatter_payload)

    df_annotated = df_subset.copy()
    df_annotated["topico_id"] = labels_all
    df_annotated["nombre_topico"] = [topic_name_map[c] for c in labels_all]
    df_annotated["similitud_centroide"] = np.round(doc_sims, 4)

    return df_annotated, topic_info


def model_all_topics(
    df_pos,
    df_neg,
    emb_pos,
    emb_neg,
    umap5_pos=None,
    umap5_neg=None,
    embedder=None,
    lang="es",
    stopwords_list=None,
    min_partition_size=30,
    max_clusters_pos=7,
    max_clusters_neg=5,
    output_path="data/processed/cdmx_topics.csv",
):
    """
    Modela tópicos independientes para comentarios positivos y negativos usando XidooTopic.
    Reutiliza el embedder de XidooTopic para KeyBERT evitando duplicar modelos en RAM.
    """
    kw_model = build_offline_keybert(embedder=embedder)
    nlp_spacy = load_spacy_nlp(lang)

    df_pos_ann, topics_pos = model_topics_for_subset(
        df_pos,
        emb_pos,
        umap5_subset=umap5_pos,
        subset_name="positivo",
        lang=lang,
        stopwords_list=stopwords_list,
        kw_model=kw_model,
        nlp_spacy=nlp_spacy,
        min_partition_size=min_partition_size,
        max_clusters=max_clusters_pos,
    )
    df_neg_ann, topics_neg = model_topics_for_subset(
        df_neg,
        emb_neg,
        umap5_subset=umap5_neg,
        subset_name="negativo",
        lang=lang,
        stopwords_list=stopwords_list,
        kw_model=kw_model,
        nlp_spacy=nlp_spacy,
        min_partition_size=min_partition_size,
        max_clusters=max_clusters_neg,
    )

    df_all = pd.concat([df_pos_ann, df_neg_ann], ignore_index=True)
    emb_parts = [e for e in [emb_pos, emb_neg] if e is not None and len(e) > 0]
    emb_all = np.vstack(emb_parts) if emb_parts else None

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        to_save = df_all.drop(columns=["tokens"]) if "tokens" in df_all.columns else df_all
        to_save.to_csv(output_path, index=False)
        print(f"\n[Tópicos - XidooTopic] Matriz final de tópicos exportada a: {output_path}")

    free_memory(kw_model, nlp_spacy)
    return df_all, emb_all, {"positivos": topics_pos, "negativos": topics_neg}


# ============================================================================
# SECCIÓN INTEGRADA: src/comparative_analyzer.py
# ============================================================================
"""
Módulo de Comparación Estratégica: Imagen Proyectada (DMO) vs. Imagen Percibida.
Usa los embeddings semánticos densos generados con XidooTopic para calcular:
  - Similitud semántica global (Índice de Alineación) entre DMO y fuentes percibidas.
  - Similitud semántica promedio a nivel documento respecto al centroide DMO.
  - Similitud individual de cada fuente frente a la DMO.
  - Comparación de tópicos y brechas temáticas cuantitativas.
  - Comparación de sentimiento (polaridad positiva/negativa).
  - Tabla comparativa por fuente: tópicos principales y tono predominante frente a DMO.
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_semantic_alignment(df, embeddings, dmo_label="sectur", source_col="fuente"):
    """
    Calcula la similitud semántica coseno sobre embeddings densos XidooTopic
    entre el corpus DMO y el corpus percibido (global y por fuente individual).
    """
    src_series = df[source_col].astype(str).str.lower()
    is_dmo = (src_series == dmo_label.lower()).to_numpy()

    if not np.any(is_dmo):
        raise ValueError(f"No se encontraron registros para la etiqueta DMO '{dmo_label}'.")
    if not np.any(~is_dmo):
        raise ValueError("Se requiere al menos un registro de fuentes percibidas (distintas a DMO).")

    emb_dmo = embeddings[is_dmo]
    emb_perc = embeddings[~is_dmo]

    centroid_dmo = emb_dmo.mean(axis=0, keepdims=True)
    centroid_dmo = centroid_dmo / (np.linalg.norm(centroid_dmo) + 1e-9)

    centroid_perc = emb_perc.mean(axis=0, keepdims=True)
    centroid_perc = centroid_perc / (np.linalg.norm(centroid_perc) + 1e-9)

    # Similitud global de centroides + similitud promedio documento-a-centroide DMO
    global_centroid_sim = float(cosine_similarity(centroid_dmo, centroid_perc)[0, 0])
    doc_level_sims = cosine_similarity(emb_perc, centroid_dmo).flatten()
    mean_doc_sim = float(np.mean(doc_level_sims))

    alignment_index = round(global_centroid_sim * 100, 2)
    doc_alignment_index = round(mean_doc_sim * 100, 2)

    # Similitud individual de cada fuente frente al centroide DMO (agnóstica a nombres)
    source_sims = {}
    for src in sorted(df[source_col].astype(str).unique()):
        if src.lower() == dmo_label.lower():
            continue
        mask = (df[source_col].astype(str) == src).to_numpy()
        if np.any(mask):
            c_src = embeddings[mask].mean(axis=0, keepdims=True)
            c_src = c_src / (np.linalg.norm(c_src) + 1e-9)
            sim_src = float(cosine_similarity(centroid_dmo, c_src)[0, 0])
            source_sims[src] = round(sim_src * 100, 2)

    return alignment_index, doc_alignment_index, source_sims, centroid_dmo


def analyze_topic_and_sentiment_gaps(df, dmo_label="sectur", source_col="fuente"):
    """
    Contrasta distribución de tópicos y sentimiento entre DMO y fuentes percibidas,
    generando además la tabla comparativa por fuente individual.
    """
    is_dmo = df[source_col].astype(str).str.lower() == dmo_label.lower()

    # 1. Comparación de Sentimiento por fuente
    sent_by_source = pd.crosstab(df[source_col], df["sentimiento"], normalize="index") * 100
    sent_by_source = sent_by_source.round(2)

    dmo_pos_pct = round(float(np.mean(df.loc[is_dmo, "sentimiento"] == "positivo") * 100), 2)
    perc_pos_pct = round(float(np.mean(df.loc[~is_dmo, "sentimiento"] == "positivo") * 100), 2)
    sentiment_gap = round(dmo_pos_pct - perc_pos_pct, 2)

    # 2. Comparación de Tópicos (DMO vs Percibido agregado)
    topic_dist_dmo = (df.loc[is_dmo, "nombre_topico"].value_counts(normalize=True) * 100).round(2).to_dict()
    topic_dist_perc = (df.loc[~is_dmo, "nombre_topico"].value_counts(normalize=True) * 100).round(2).to_dict()

    all_topics = sorted(set(topic_dist_dmo.keys()).union(set(topic_dist_perc.keys())))
    gaps_table = []
    for top in all_topics:
        pct_dmo = topic_dist_dmo.get(top, 0.0)
        pct_perc = topic_dist_perc.get(top, 0.0)
        gap = round(pct_dmo - pct_perc, 2)
        if gap > 5.0:
            diag = "Sobreproyectado por DMO (menor presencia en usuarios)"
        elif gap < -5.0:
            diag = "Subatendido por DMO (alta conversación en usuarios)"
        else:
            diag = "Alineado temáticamente"
        gaps_table.append(
            {
                "topico": top,
                "pct_dmo": pct_dmo,
                "pct_percibido": pct_perc,
                "brecha_puntos": gap,
                "diagnostico": diag,
            }
        )

    df_gaps = pd.DataFrame(gaps_table).sort_values("brecha_puntos", ascending=False).reset_index(drop=True)

    # 3. Análisis comparativo por fuente (tópicos principales + tono predominante frente a DMO)
    source_rows = []
    for src in sorted(df[source_col].astype(str).unique()):
        sub = df[df[source_col].astype(str) == src]
        n_src = len(sub)
        role = "DMO (Proyectada)" if src.lower() == dmo_label.lower() else "Percibida (UGC)"
        pos_rate = round(float(np.mean(sub["sentimiento"] == "positivo") * 100), 2)
        mean_pol = round(float(sub["score_polaridad"].mean()), 3)
        tone = "Positivo" if pos_rate >= 50.0 else "Negativo"
        top_counts = sub["nombre_topico"].value_counts(normalize=True) * 100
        top_3_str = "; ".join(f"{t} ({round(float(p), 1)}%)" for t, p in top_counts.head(3).items())
        source_rows.append(
            {
                "fuente": src,
                "universo": role,
                "registros": n_src,
                "pct_positivo": pos_rate,
                "score_polaridad_medio": mean_pol,
                "tono_predominante": tone,
                "topicos_principales": top_3_str,
            }
        )
    df_source_summary = pd.DataFrame(source_rows)

    return {
        "dmo_pos_pct": dmo_pos_pct,
        "perc_pos_pct": perc_pos_pct,
        "sentiment_gap": sentiment_gap,
        "sent_by_source": sent_by_source,
        "gaps_table": df_gaps,
        "source_summary": df_source_summary,
    }


def run_comparative_analysis(df, embeddings, dmo_label="sectur", source_col="fuente", output_dir="data/processed"):
    """
    Ejecuta la comparación completa DMO vs Imagen Percibida y exporta tablas CSV.
    """
    print("\n=======================================================")
    print("  COMPARACIÓN CENTRAL: IMAGEN PROYECTADA VS. PERCIBIDA")
    print("=======================================================")

    alignment_idx, doc_alignment_idx, source_sims, centroid_dmo = compute_semantic_alignment(
        df, embeddings=embeddings, dmo_label=dmo_label, source_col=source_col
    )
    gaps_summary = analyze_topic_and_sentiment_gaps(df, dmo_label=dmo_label, source_col=source_col)

    print(f"\n>>> ÍNDICE DE ALINEACIÓN SEMÁNTICA GLOBAL (Centroides): {alignment_idx}%")
    print(f">>> SIMILITUD COSENO PROMEDIO A NIVEL DOCUMENTO vs DMO: {doc_alignment_idx}%")
    print(">>> Similitud Semántica por Fuente Individual vs. DMO:")
    for src, sim in source_sims.items():
        print(f"    - {src}: {sim}% de alineación con DMO")

    print(f"\n>>> Brecha de Polaridad Positiva: {gaps_summary['sentiment_gap']} puntos porcentuales.")
    print(f"    - DMO ({dmo_label}): {gaps_summary['dmo_pos_pct']}% positiva.")
    print(f"    - Fuentes Percibidas: {gaps_summary['perc_pos_pct']}% positiva.")

    print("\n>>> Tabla Comparativa por Fuente (Tópicos Principales y Tono):")
    print(gaps_summary["source_summary"].to_string(index=False))

    print("\n>>> Brechas Temáticas Detectadas (Top):")
    print(gaps_summary["gaps_table"].to_string(index=False))
    print("=======================================================\n")

    os.makedirs(output_dir, exist_ok=True)
    gaps_csv = os.path.join(output_dir, "cdmx_alignment_dmo.csv")
    gaps_summary["gaps_table"].to_csv(gaps_csv, index=False)

    src_csv = os.path.join(output_dir, "cdmx_source_comparison.csv")
    gaps_summary["source_summary"].to_csv(src_csv, index=False)
    print(f"[Comparativo] Tablas exportadas a: {gaps_csv} y {src_csv}")

    return {
        "alignment_index": alignment_idx,
        "doc_alignment_index": doc_alignment_idx,
        "source_sims": source_sims,
        "sentiment_summary": gaps_summary,
        "centroid_dmo": centroid_dmo,
    }


# ============================================================================
# SECCIÓN INTEGRADA: src/price_analyzer.py
# ============================================================================
"""
Módulo de Análisis Específico del Concepto 'Precio, Valor, Costo' con XidooTopic.
Calcula la similitud semántica coseno sobre los embeddings densos (TextEmbedder)
de cada comentario frente al vector conceptual sintético de 'precio / valor / costo',
contrasta frecuencia, tono y contexto temático entre la DMO y las fuentes percibidas,
y extrae los 5 comentarios más relevantes indicando su fuente.
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PRICE_QUERIES_BY_LANG = {
    "es": [
        "El precio es muy caro o barato, cuánto cuesta la entrada, el boleto o la tarifa.",
        "Relación calidad precio, vale la pena pagar el costo, dinero, cobro excesivo o gratis.",
    ],
    "en": [
        "The price is expensive or cheap, ticket cost, admission fee and entrance charges.",
        "Good or bad value for money, worth the price, overpriced or affordable budget.",
    ],
    "fr": [
        "Le prix est cher ou abordable, coût du billet, tarif d'entrée et dépenses.",
        "Rapport qualité prix, vaut le coût, gratuit ou trop cher pour le service.",
    ],
}


def analyze_price_concept(
    df,
    embeddings,
    embedder,
    dmo_label="sectur",
    lang="es",
    source_col="fuente",
    text_orig_col="texto_original",
    output_path="data/processed/cdmx_price_analysis.csv",
):
    """
    Evalúa la relevancia semántica del concepto 'Precio, Valor, Costo' usando
    los embeddings de XidooTopic sobre oraciones prototipo naturales.
    """
    print("\n=======================================================")
    print("  ANÁLISIS ESPECÍFICO: CONCEPTO 'PRECIO, VALOR, COSTO'")
    print("=======================================================")

    lang_queries = list(PRICE_QUERIES_BY_LANG.get(lang.lower(), PRICE_QUERIES_BY_LANG["es"]))
    if lang.lower() != "en":
        # Incluir prototipo en inglés para foros internacionales (ej. Reddit) en corpus mixtos
        lang_queries.append(PRICE_QUERIES_BY_LANG["en"][0])

    # Vectorizar oraciones prototipo con xidootopic.TextEmbedder y promediar centroide conceptual
    q_embs = embedder.embed_texts(lang_queries)
    query_emb = q_embs.mean(axis=0, keepdims=True)
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    similarities = cosine_similarity(embeddings, query_emb).flatten()

    df_result = df.copy()
    df_result["similitud_precio"] = np.round(similarities, 4)

    # Umbral estadístico adaptativo (media + 0.5 desviación estándar, mínimo 0.20)
    mean_sim = float(np.mean(similarities))
    std_sim = float(np.std(similarities))
    price_threshold = round(max(0.20, mean_sim + 0.5 * std_sim), 4)
    df_result["relacionado_precio"] = (df_result["similitud_precio"] >= price_threshold).astype(int)

    is_dmo = (df_result[source_col].astype(str).str.lower() == dmo_label.lower()).to_numpy()
    is_price = (df_result["relacionado_precio"] == 1).to_numpy()

    total_dmo = int(np.sum(is_dmo))
    total_perc = int(np.sum(~is_dmo))

    price_dmo = int(np.sum(is_dmo & is_price))
    price_perc = int(np.sum((~is_dmo) & is_price))

    pct_dmo = round(price_dmo / max(total_dmo, 1) * 100, 2)
    pct_perc = round(price_perc / max(total_perc, 1) * 100, 2)

    dmo_mean_sim = round(float(similarities[is_dmo].mean()) if total_dmo > 0 else 0.0, 4)
    perc_mean_sim = round(float(similarities[~is_dmo].mean()) if total_perc > 0 else 0.0, 4)

    dmo_price_tone = round(
        float(df_result.loc[is_dmo & is_price, "score_polaridad"].mean()) if price_dmo > 0 else 0.0,
        3,
    )
    perc_price_tone = round(
        float(df_result.loc[(~is_dmo) & is_price, "score_polaridad"].mean()) if price_perc > 0 else 0.0,
        3,
    )

    # Contexto temático predominante en comentarios sobre precio
    dmo_price_ctx = (
        df_result.loc[is_dmo & is_price, "nombre_topico"].value_counts().head(2).to_dict()
        if price_dmo > 0 and "nombre_topico" in df_result.columns
        else {}
    )
    perc_price_ctx = (
        df_result.loc[(~is_dmo) & is_price, "nombre_topico"].value_counts().head(3).to_dict()
        if price_perc > 0 and "nombre_topico" in df_result.columns
        else {}
    )

    print(f">>> Umbral semántico adaptativo (media + 1σ): {price_threshold}")
    print(f">>> Frecuencia y Tono en el eje Precio / Valor / Costo:")
    print(
        f"    - DMO ({dmo_label}): {price_dmo:,} de {total_dmo:,} ({pct_dmo}%) | "
        f"Similitud media: {dmo_mean_sim} | Tono promedio: {dmo_price_tone}"
    )
    print(
        f"    - Percibido (UGC): {price_perc:,} de {total_perc:,} ({pct_perc}%) | "
        f"Similitud media: {perc_mean_sim} | Tono promedio: {perc_price_tone}"
    )
    print(f"    - Contexto DMO: {dmo_price_ctx}")
    print(f"    - Contexto Percibido: {perc_price_ctx}")

    # Top 5 comentarios más relevantes al concepto precio/valor/costo
    top_price_indices = np.argsort(similarities)[::-1][:5]
    top_5 = df_result.iloc[top_price_indices][
        [source_col, "sentimiento", "score_polaridad", "similitud_precio", text_orig_col]
    ].copy()

    print("\n>>> Top 5 Comentarios Más Cercanos al Concepto 'Precio/Valor/Costo':")
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        txt_preview = str(row[text_orig_col])[:140].replace("\n", " ")
        print(
            f"  {idx}. [{str(row[source_col]).upper()} | {row['sentimiento']} | "
            f"Sim: {row['similitud_precio']}]"
        )
        print(f"     \"{txt_preview}...\"")
    print("=======================================================\n")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        to_save = df_result.drop(columns=["tokens"]) if "tokens" in df_result.columns else df_result
        to_save.to_csv(output_path, index=False)
        print(f"[Precio] Corpus con métricas de precio exportado a: {output_path}")

    price_stats = {
        "threshold": price_threshold,
        "price_dmo_count": price_dmo,
        "total_dmo": total_dmo,
        "price_perc_count": price_perc,
        "total_perc": total_perc,
        "pct_dmo": pct_dmo,
        "pct_perc": pct_perc,
        "dmo_mean_sim": dmo_mean_sim,
        "perc_mean_sim": perc_mean_sim,
        "dmo_price_tone": dmo_price_tone,
        "perc_price_tone": perc_price_tone,
        "dmo_price_context": dmo_price_ctx,
        "perc_price_context": perc_price_ctx,
        "top_5": top_5,
    }
    return df_result, price_stats



from dashboard_html import generate_dashboard

# ============================================================================
# SECCIÓN INTEGRADA: src/report_generator.py
# ============================================================================
"""
Módulo de Extracción y Presentación de Resultados Clave (Punto 6 del Proyecto).
Genera el reporte estructurado (consola, Markdown y JSON) con:
  - Palabras clave representativas por tópico (positivo y negativo) y fuentes de origen.
  - Comentarios representativos más cercanos al centroide de cada tópico (RepresentativeDocsExtractor).
  - Resumen de brechas entre imagen proyectada (DMO) e imagen percibida, incluyendo tabla por fuente.
  - Síntesis de outliers (n-gramas) y análisis específico de 'Precio, Valor, Costo'.
"""
import os
import json
from datetime import datetime


def generate_key_results_report(
    report_title,
    dmo_label,
    language,
    palette,
    n_total_clean,
    n_outliers,
    ngrams_dict,
    topic_catalog,
    comp_results,
    price_stats,
    source_col="fuente",
    md_output_path="docs/reporte_resultados_xidootopic.md",
    json_output_path="data/processed/resumen_xidootopic.json",
):
    """
    Construye e imprime el reporte de resultados clave del pipeline XidooTopic.
    """
    os.makedirs(os.path.dirname(md_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    gaps_summary = comp_results["sentiment_summary"]
    gaps_df = gaps_summary["gaps_table"]
    src_df = gaps_summary["source_summary"]

    lines = [
        f"# {report_title}",
        "",
        f"- **Fecha de ejecución:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Motor de modelado:** `xidootopic` (Embeddings Transformer + UMAP + IsolationForest + KNN/FAISS + Leiden + ClusterMerger + KeyBERT/BM25)",
        f"- **Etiqueta DMO:** `{dmo_label}` | **Idioma:** `{language}` | **Paleta:** `{palette}`",
        f"- **Registros válidos preprocesados:** {n_total_clean:,} | **Outliers aislados:** {n_outliers:,}",
        "",
        "## 1. Índice de Alineación Semántica y Brechas Globales (DMO vs. Percibido)",
        f"- **Índice de Alineación Semántica Global (Centroides):** `{comp_results['alignment_index']}%`",
        f"- **Similitud Coseno Promedio a Nivel Documento vs. DMO:** `{comp_results['doc_alignment_index']}%`",
        f"- **Polaridad Positiva:** DMO `{gaps_summary['dmo_pos_pct']}%` vs. Percibido `{gaps_summary['perc_pos_pct']}%` (Brecha: `{gaps_summary['sentiment_gap']} pts`)",
        "",
        "### Similitud Individual por Fuente vs. DMO",
    ]
    for src, sim in comp_results["source_sims"].items():
        lines.append(f"- `{src}`: **{sim}%**")

    lines.extend(["", "### Tabla Comparativa por Fuente (Tópicos Principales y Tono Predominante)"])
    lines.append("| Fuente | Universo | Registros | % Positivo | Score Medio | Tono Predominante | Tópicos Principales |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    for _, r in src_df.iterrows():
        lines.append(
            f"| `{r['fuente']}` | {r['universo']} | {r['registros']:,} | {r['pct_positivo']}% | "
            f"{r['score_polaridad_medio']} | {r['tono_predominante']} | {r['topicos_principales']} |"
        )

    lines.extend(["", "## 2. Tópicos Identificados por Sentimiento (Palabras Clave, Fuentes y Comentario Representativo)"])
    for pol_label, pol_key in [("Tópicos Positivos", "positivos"), ("Tópicos Negativos", "negativos")]:
        lines.append(f"\n### {pol_label}")
        for t in topic_catalog.get(pol_key, []):
            kw_str = ", ".join(t.get("palabras_clave", []))
            fuentes_str = ", ".join(f"{k}: {v}" for k, v in t.get("fuentes_origen", {}).items())
            rep_txt = str(t.get("comentario_representativo", "")).replace("\n", " ")[:260]
            lines.append(f"#### `{t['nombre_topico']}` ({t['peso_relativo']}% | n={t['n_documentos']:,})")
            lines.append(f"- **Palabras clave:** {kw_str}")
            lines.append(f"- **Fuentes de origen en el tópico:** {fuentes_str} (Dominante: `{t.get('fuente_dominante', 'N/A')}`)")
            lines.append(
                f"- **Comentario representativo (centroide, sim={t.get('similitud_rep', 0.0)}, fuente=`{t.get('fuente_representativa', 'N/A')}`):** "
                f"> \"{rep_txt}...\""
            )

    lines.extend(["", "## 3. Resumen de Brechas Temáticas (DMO vs. Percibido)"])
    lines.append("| Tópico | DMO % | Percibido % | Brecha (pts) | Diagnóstico |")
    lines.append("|---|---:|---:|---:|---|")
    for _, r in gaps_df.iterrows():
        lines.append(
            f"| `{r['topico']}` | {r['pct_dmo']}% | {r['pct_percibido']}% | {r['brecha_puntos']} | {r['diagnostico']} |"
        )

    lines.extend(["", "## 4. N-Gramas en Comentarios Atípicos (Outliers)"])
    for titulo, key in [("Unigramas", "unigrams"), ("Bigramas", "bigrams"), ("Trigramas", "trigrams")]:
        items = ngrams_dict.get(key, [])[:10]
        formatted = ", ".join(f"`{it['ngram']}` ({it['frecuencia']})" for it in items)
        lines.append(f"- **{titulo} Top 10:** {formatted}")

    lines.extend([
        "",
        "## 5. Análisis Específico del Concepto 'Precio / Valor / Costo'",
        f"- **Umbral de similitud semántica:** `{price_stats['threshold']}`",
        f"- **DMO (`{dmo_label}`):** {price_stats['price_dmo_count']:,} de {price_stats['total_dmo']:,} ({price_stats['pct_dmo']}%) | Tono medio: `{price_stats['dmo_price_tone']}` | Similitud media: `{price_stats['dmo_mean_sim']}`",
        f"- **Percibido (UGC):** {price_stats['price_perc_count']:,} de {price_stats['total_perc']:,} ({price_stats['pct_perc']}%) | Tono medio: `{price_stats['perc_price_tone']}` | Similitud media: `{price_stats['perc_mean_sim']}`",
        f"- **Contexto temático DMO:** `{price_stats['dmo_price_context']}`",
        f"- **Contexto temático Percibido:** `{price_stats['perc_price_context']}`",
        "",
        "### Top 5 Comentarios Más Cercanos al Concepto 'Precio / Valor / Costo'",
    ])
    for idx, (_, row) in enumerate(price_stats["top_5"].iterrows(), 1):
        txt = str(row["texto_original"]).replace("\n", " ")[:240]
        lines.append(
            f"{idx}. **[{str(row[source_col]).upper()} | {row['sentimiento']} | Sim: {row['similitud_precio']}]** "
            f"\"{txt}...\""
        )

    md_content = "\n".join(lines) + "\n"
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Exportar JSON estructurado
    json_payload = {
        "report_title": report_title,
        "dmo_label": dmo_label,
        "language": language,
        "palette": palette,
        "n_total_clean": int(n_total_clean),
        "n_outliers": int(n_outliers),
        "alignment_index": comp_results["alignment_index"],
        "doc_alignment_index": comp_results["doc_alignment_index"],
        "source_sims": comp_results["source_sims"],
        "sentiment_gap": gaps_summary["sentiment_gap"],
        "dmo_pos_pct": gaps_summary["dmo_pos_pct"],
        "perc_pos_pct": gaps_summary["perc_pos_pct"],
        "topics": topic_catalog,
        "gaps": gaps_df.to_dict(orient="records"),
        "source_summary": src_df.to_dict(orient="records"),
        "outlier_ngrams": ngrams_dict,
        "price_analysis": {
            k: v for k, v in price_stats.items() if k != "top_5"
        },
        "price_top_5": price_stats["top_5"].to_dict(orient="records"),
    }
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[Reporte Clave] Reporte Markdown exportado a: {md_output_path}")
    print(f"[Reporte Clave] Resumen JSON exportado a: {json_output_path}")
    return md_output_path, json_output_path


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL (CLI)
# ============================================================================
import argparse
import time

def parse_arguments():
    """
    Configura y valida los 7 parámetros obligatorios por línea de comandos mediante argparse.
    """
    parser = argparse.ArgumentParser(
        description="Sistema de Análisis Comparativo (XidooTopic): Imagen Proyectada (DMO) vs. Imagen Percibida.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 1. Dataset CSV unificado (opcional, por defecto carga data/raw)
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="data/processed/cdmx_unified.csv",
        help="Ruta al archivo CSV unificado o crudo.",
    )
    # 2. Columna de comentarios textuales
    parser.add_argument(
        "-c", "--text_column",
        type=str,
        default="texto",
        help="Nombre de la columna que contiene los comentarios textuales (default: texto)",
    )
    # 3. Columna de fuentes
    parser.add_argument(
        "-s", "--source_column",
        type=str,
        default="fuente",
        help="Nombre de la columna que identifica la fuente del comentario (default: fuente)",
    )
    # 4. Etiqueta DMO
    parser.add_argument(
        "-m", "--dmo_label",
        type=str,
        default="sectur",
        help="Valor/etiqueta que identifica a la DMO en la columna de fuentes (default: sectur)",
    )
    # 5. Idioma objetivo
    parser.add_argument(
        "-l", "--language",
        type=str,
        choices=["es", "en", "fr"],
        default="es",
        help="Idioma objetivo del análisis (es: Español, en: Inglés, fr: Francés). Default: es",
    )
    # 6. Título del reporte
    parser.add_argument(
        "-t", "--report_title",
        type=str,
        default="Análisis Comparativo DMO vs Imagen Percibida",
        help="Título a desplegar en reportes y visualizaciones generadas.",
    )
    # 7. Paleta de colores accesible
    parser.add_argument(
        "-p", "--palette",
        type=str,
        choices=["okabe-ito", "viridis"],
        default="okabe-ito",
        help="Paleta de colores accesible apta para daltonismo. Default: okabe-ito",
    )

    # Opcionales auxiliares de configuración
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Modelo HuggingFace para xidootopic.TextEmbedder. Default: {DEFAULT_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--min_partition_size",
        type=int,
        default=30,
        help="Umbral mínimo para modelado de tópicos; por debajo usa frecuencia léxica. Default: 30",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Proporción esperada de outliers en IsolationForestOutlierRemover. Default: 0.05",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Límite opcional de muestreo estratificado por fuente para pruebas rápidas.",
    )
    # Modo dual/multi-fuente: uno o varios CSV de UGC y uno o varios CSV de DMO
    parser.add_argument(
        "--ugc_dataset",
        nargs="+",
        default=None,
        help="Uno o más CSVs de comentarios UGC (requiere --dmo_dataset).",
    )
    parser.add_argument(
        "--ugc_text_col",
        nargs="+",
        default=["texto"],
        help="Columna(s) de texto en --ugc_dataset (1 global o 1 por archivo; default: texto).",
    )
    parser.add_argument(
        "--ugc_label",
        nargs="+",
        default=None,
        help="Etiqueta(s) de fuente para --ugc_dataset (1 por archivo; por defecto usa el nombre del archivo).",
    )
    parser.add_argument(
        "--dmo_dataset",
        nargs="+",
        default=None,
        help="Uno o más CSVs de comentarios DMO (se fusionan en una sola fuente --dmo_label).",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="Directorio de salida donde se crearán los resultados (default: ./resultados junto a main.py).",
    )
    parser.add_argument(
        "--dmo_text_col",
        nargs="+",
        default=["texto"],
        help="Columna(s) de texto en --dmo_dataset (1 global o 1 por archivo; default: texto).",
    )

    return parser.parse_args()


def main():
    start_time = time.time()
    args = parse_arguments()

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (BASE_DIR / "resultados")
    csv_dir = out_dir / "csv"
    rep_dir = out_dir / "reportes"
    csv_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_input_path(p_str):
        p = Path(p_str)
        if p.exists():
            return str(p)
        rel = BASE_DIR / p_str
        return str(rel) if rel.exists() else p_str

    print("\n" + "=" * 80)
    print("  ANÁLISIS COMPARATIVO DMO VS. IMAGEN PERCIBIDA (IMPULSADO POR XIDOOTOPIC)")
    print("  Visualización de Información | Posgrado LIDIA 2026")
    print("=" * 80)
    print(f"  • Archivo de Entrada : {args.dataset}")
    print(f"  • Columna de Texto   : {args.text_column}")
    print(f"  • Columna de Fuente  : {args.source_column}")
    print(f"  • Etiqueta DMO       : {args.dmo_label}")
    print(f"  • Idioma Objetivo    : {args.language}")
    print(f"  • Título del Reporte : {args.report_title}")
    print(f"  • Paleta Accesible   : {args.palette}")
    print(f"  • Modelo Embeddings  : {args.embedding_model}")
    print("=" * 80 + "\n")

    # 1. Carga del dataset unificado o modo dual (UGC + DMO separados)
    if args.ugc_dataset or args.dmo_dataset:
        if not (args.ugc_dataset and args.dmo_dataset):
            raise ValueError("Debes especificar ambos parámetros: --ugc_dataset y --dmo_dataset.")
        df = load_dual_datasets(
            ugc_path=[_resolve_input_path(p) for p in args.ugc_dataset],
            ugc_text_col=args.ugc_text_col,
            dmo_path=[_resolve_input_path(p) for p in args.dmo_dataset],
            dmo_text_col=args.dmo_text_col,
            ugc_label=args.ugc_label,
            dmo_label=args.dmo_label,
            out_text_col=args.text_column,
            out_source_col=args.source_column,
        )
    else:
        ds_path = _resolve_input_path(args.dataset)
        if not os.path.exists(ds_path):
            raw_dir = BASE_DIR / "data" / "raw"
            ex_dmo = BASE_DIR / "data" / "dmo_ejemplo.csv"
            ex_ugc1 = BASE_DIR / "data" / "ugc_resenas_ejemplo.csv"
            ex_ugc2 = BASE_DIR / "data" / "ugc_redes_ejemplo.csv"
            if "cdmx_unified.csv" in ds_path and raw_dir.exists():
                ds_path = str(csv_dir / "cdmx_unified.csv")
                print(f"[Dataset] Compilando corpus base desde {raw_dir}...")
                build_unified_dataset(raw_dir=str(raw_dir), output_path=ds_path)
                df = load_dataset(ds_path, args.text_column, args.source_column)
            elif "cdmx_unified.csv" in ds_path and ex_dmo.exists() and ex_ugc1.exists() and ex_ugc2.exists():
                print("[Dataset] Usando los 3 datasets sintéticos de ejemplo en data/...")
                df = load_dual_datasets(
                    ugc_path=[str(ex_ugc1), str(ex_ugc2)],
                    ugc_text_col=["texto", "texto"],
                    dmo_path=[str(ex_dmo)],
                    dmo_text_col=["comentario"],
                    ugc_label=["resenas_ejemplo", "redes_ejemplo"],
                    dmo_label=args.dmo_label,
                    out_text_col=args.text_column,
                    out_source_col=args.source_column,
                )
            else:
                raise FileNotFoundError(f"No se encontró el archivo CSV especificado: {args.dataset}")
        else:
            df = load_dataset(ds_path, args.text_column, args.source_column)

    if args.sample_limit and len(df) > args.sample_limit:
        print(f"[CLI] Aplicando muestreo estratificado a {args.sample_limit:,} registros...")
        df = (
            df.groupby(args.source_column, group_keys=False)
            .apply(lambda x: x.sample(max(1, int(round(len(x) / len(df) * args.sample_limit))), random_state=42))
            .reset_index(drop=True)
        )

    print(f"[Dataset] Registros cargados: {len(df):,}")
    print(f"[Dataset] Distribución por fuente:\n{df[args.source_column].value_counts()}\n")

    # 2. Preprocesamiento unificado (idioma ES/EN, anti-OCR, lematización, stopwords, >= 6 palabras)
    df_clean, df_descartados = preprocess_corpus(
        df,
        text_col=args.text_column,
        source_col=args.source_column,
        lang=args.language,
        min_words=6,
        output_path=None,
        discarded_path=str(csv_dir / "cdmx_descartados.csv"),
    )
    free_memory(df)
    print(f"[RAM] Uso actual tras preprocesamiento: {get_rss_mb()} MB")

    # 3. Generación de embeddings densos con xidootopic.TextEmbedder (HFencoder + MeanPooling)
    embedder = build_xidoo_embedder(model_name=args.embedding_model, batch_size=64, max_length=256)
    print(f"[XidooTopic-Embedder] Calculando embeddings para {len(df_clean):,} documentos...")
    embeddings_all = embedder.embed_texts(df_clean["texto_chunked"].tolist()).astype(np.float32)
    print(f"[XidooTopic-Embedder] Matriz de embeddings generada: {embeddings_all.shape} | RAM: {get_rss_mb()} MB\n")

    # 4. Detección de Outliers con xidootopic.UMAPReducer + IsolationForestOutlierRemover
    df_inliers, df_outliers, emb_inliers, umap5_inliers, ngrams_dict = (
        detect_and_analyze_outliers(
            df_clean,
            embeddings=embeddings_all,
            text_clean_col="texto_limpio",
            text_embed_col="texto_chunked",
            source_col="fuente",
            contamination=args.contamination,
            output_dir=str(csv_dir),
        )
    )
    n_total_clean = len(df_clean)
    n_outliers = len(df_outliers)
    free_memory(df_clean, embeddings_all)
    print(f"[RAM] Uso actual tras Outliers + UMAP: {get_rss_mb()} MB")

    # 5. Análisis de Sentimientos sobre Inliers (conservando fuente y embeddings)
    df_sent, df_pos, df_neg, emb_pos, emb_neg, umap5_pos, umap5_neg = analyze_sentiments(
        df_inliers,
        embeddings=emb_inliers,
        emb_umap5=umap5_inliers,
        text_col="texto_chunked",
        text_clean_col="texto_limpio",
        source_col="fuente",
        embedder=embedder,
        output_path=None,
    )
    free_memory(df_inliers, df_sent, emb_inliers, umap5_inliers)
    print(f"[RAM] Uso actual tras Sentimiento: {get_rss_mb()} MB")

    # 6. Modelado de Tópicos por Sentimiento con XidooTopic (KNN + Leiden + ClusterMerger + KeyBERT/BM25)
    stop_list = list(TextPreprocessor(lang_code=args.language, use_lemmatization=False).stop_words)
    df_topics, emb_topics, topic_catalog = model_all_topics(
        df_pos,
        df_neg,
        emb_pos=emb_pos,
        emb_neg=emb_neg,
        umap5_pos=umap5_pos,
        umap5_neg=umap5_neg,
        embedder=embedder,
        lang=args.language,
        stopwords_list=stop_list,
        min_partition_size=args.min_partition_size,
        output_path=None,
    )
    free_memory(df_pos, df_neg, emb_pos, emb_neg, umap5_pos, umap5_neg)
    print(f"[RAM] Uso actual tras Modelado de Tópicos: {get_rss_mb()} MB")

    # 7. Comparación DMO vs. Imagen Percibida (similitud semántica global y por fuente)
    comp_results = run_comparative_analysis(
        df_topics,
        embeddings=emb_topics,
        dmo_label=args.dmo_label,
        source_col="fuente",
        output_dir=str(csv_dir),
    )

    # 8. Análisis Específico del Concepto "Precio, Valor, Costo"
    df_final, price_stats = analyze_price_concept(
        df_topics,
        embeddings=emb_topics,
        embedder=embedder,
        dmo_label=args.dmo_label,
        lang=args.language,
        source_col="fuente",
        output_path=None,
    )
    free_memory(df_topics, emb_topics, embedder)

    # Ejes temáticos generales (k=6 NMF) para mantener compatibilidad con tablero DMO vs UGC
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVec
    from sklearn.decomposition import NMF as _NMF
    _vec6 = _TfidfVec(max_features=3000, min_df=3)
    _X6 = _vec6.fit_transform(df_final["texto_limpio"].fillna("").astype(str))
    _nmf6 = _NMF(n_components=6, random_state=42, init="nndsvda", max_iter=300)
    df_final["topico_6"] = np.argmax(_nmf6.fit_transform(_X6), axis=1)

    # Limpiar columnas redundantes antes de exportar datasets finales (utf-8-sig para Excel)
    drop_cols = [c for c in ["texto", "texto_xidoo", "texto_chunked", "tokens"] if c in df_final.columns]
    if drop_cols:
        df_final = df_final.drop(columns=drop_cols)
    drop_out = [c for c in ["texto", "texto_xidoo", "texto_chunked", "tokens"] if c in df_outliers.columns]
    if drop_out:
        df_outliers = df_outliers.drop(columns=drop_out)

    final_csv = str(csv_dir / "cdmx_final_processed.csv")
    df_final.to_csv(final_csv, index=False, encoding="utf-8-sig")
    df_outliers.to_csv(csv_dir / "cdmx_outliers.csv", index=False, encoding="utf-8-sig")
    print(f"\n[Exportación] Matriz analítica final (inliers): {final_csv} ({len(df_final):,} filas)")
    print(f"[Exportación] Outliers semánticos: {csv_dir / 'cdmx_outliers.csv'} ({len(df_outliers):,} filas)")

    # 9. Generación directa del Tablero Único HTML (con datos reales en vivo de esta corrida)
    f_board = generate_dashboard(
        df_final=df_final,
        df_outliers=df_outliers,
        df_descartados=df_descartados,
        ngrams_dict=ngrams_dict,
        topic_catalog=topic_catalog,
        comp_results=comp_results,
        palette=args.palette,
        output_path=str(out_dir / "tablero_2pantallas_8puntos.html"),
    )

    # 10. Reporte de resultados clave
    f_md, f_json = generate_key_results_report(
        report_title=args.report_title,
        dmo_label=args.dmo_label,
        language=args.language,
        palette=args.palette,
        n_total_clean=n_total_clean,
        n_outliers=n_outliers,
        ngrams_dict=ngrams_dict,
        topic_catalog=topic_catalog,
        comp_results=comp_results,
        price_stats=price_stats,
        source_col="fuente",
        md_output_path=str(rep_dir / "reporte_resultados_xidootopic.md"),
        json_output_path=str(csv_dir / "resumen_xidootopic.json"),
    )

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print("  EJECUCIÓN UNIFICADA FINALIZADA CON ÉXITO")
    print("=" * 80)
    print(f"  • Tiempo de cómputo total : {elapsed} segundos")
    print(f"  • Documentos válidos      : {n_total_clean:,} (Inliers: {len(df_final):,} | Outliers: {n_outliers:,} | Descartados limpieza: {len(df_descartados):,})")
    print(f"  • Índice de Alineación DMO: {comp_results['alignment_index']}%")
    print("  • Artefactos Generados:")
    print(f"    1. {final_csv}")
    print(f"    2. {csv_dir / 'cdmx_outliers.csv'}")
    print(f"    3. {csv_dir / 'cdmx_descartados.csv'}")
    print(f"    4. {f_board}")
    print(f"    5. {f_md}")
    print(f"    6. {f_json}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

"""
Librería de generación del tablero interactivo HTML (2 pantallas × 4 cuadrantes).
Llamada exclusivamente desde main.py.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# PALETA OFICIAL OKABE-ITO (ESTÁNDAR UNIVERSAL DE 8 COLORES DALTÓNICO-SEGUROS)
# Nature Methods (2011) / Color Universal Design (CUD)
# ==============================================================================
OI_SKY_BLUE     = "#56B4E9"  # Azul Cielo
OI_ORANGE       = "#E69F00"  # Naranja Cálido
OI_BLUISH_GREEN = "#009E73"  # Verde Azulado / Esmeralda Accesible
OI_YELLOW       = "#F0E442"  # Amarillo Suave / Trigo
OI_BLUE         = "#0072B2"  # Azul Profundo
OI_VERMILION    = "#D55E00"  # Bermellón / Terracota
OI_PURPLE       = "#CC79A7"  # Púrpura Rojizo
OI_GREY         = "#999999"  # Gris Pizarra Neutro

PALETTES = {
    "okabe-ito": {
        "name": "Okabe-Ito (8 Colores CUD)",
        "sky_blue": "#56B4E9",
        "orange": "#E69F00",
        "bluish_green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermilion": "#D55E00",
        "purple": "#CC79A7",
        "grey": "#999999",
    },
    "viridis": {
        "name": "Viridis (Perceptual CVD-Safe)",
        "sky_blue": "#22A884",
        "orange": "#BBDF27",
        "bluish_green": "#7AD151",
        "yellow": "#8A64B8",
        "blue": "#2A788E",
        "vermilion": "#FDE725",
        "purple": "#A577D6",
        "grey": "#999999",
    },
}


def get_palette_colors(palette_name="okabe-ito"):
    """Devuelve el diccionario de colores para la paleta solicitada."""
    key = (palette_name or "okabe-ito").strip().lower()
    return PALETTES.get(key, PALETTES["okabe-ito"])


def _hex_to_rgba(hex_color, alpha):
    """Convierte #RRGGBB a cadena rgba(r,g,b,alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_clean_wordcloud_full(words, total_w=420, total_h=210):
    """
    Genera un panel SVG de nube de palabras limpia para el subcuadrante 1.1.
    Aplica el algoritmo espiral de Arquímedes con paleta Okabe-Ito de 8 colores.
    """
    cx, cy = total_w / 2.0, total_h / 2.0
    vals = [item[1] for item in words]
    vmin, vmax = min(vals), max(vals)
    placed = []
    elems = [
        f'<rect x="0" y="0" width="{total_w}" height="{total_h}" fill="rgba(0, 158, 115, 0.04)" rx="6"/>',
    ]
    for idx, (word, count, col, reason) in enumerate(words):
        norm = (count - vmin) / (vmax - vmin + 1e-9)
        fsize = 9.5 + (norm ** 0.55) * 15.0
        tw = len(word) * fsize * 0.56 + 5
        th = fsize * 1.12 + 3
        angle = idx * 1.618 * 2.0
        step = 0.30
        r = 0.0
        fx, fy = cx, cy
        for _ in range(2200):
            tx = cx + r * math.cos(angle) * 1.55
            ty = cy + r * math.sin(angle) * 0.85
            x0, x1 = tx - tw / 2, tx + tw / 2
            y0, y1 = ty - th / 2, ty + th / 2
            if x0 >= 8 and x1 <= total_w - 8 and y0 >= 8 and y1 <= total_h - 8:
                colls = False
                for px0, py0, px1, py1 in placed:
                    if not (x1 < px0 or x0 > px1 or y1 < py0 or y0 > py1):
                        colls = True
                        break
                if not colls:
                    placed.append((x0, y0, x1, y1))
                    fx, fy = tx, ty
                    break
            angle += step
            r += 0.22
        elems.append(
            f'<text x="{fx:.1f}" y="{fy + fsize*0.32:.1f}" text-anchor="middle" '
            f'font-size="{fsize:.1f}" font-weight="bold" fill="{col}">'
            f'<title>{word.upper()}: {count:,} menciones ({reason})</title>{word}</text>'
        )
    return f"""<svg viewBox="0 0 {total_w} {total_h}" preserveAspectRatio="xMidYMid meet" style="width:100%; height:100%; display:block;" xmlns="http://www.w3.org/2000/svg">
    {''.join(elems)}
    </svg>"""


def generate_dashboard(df_final=None, df_outliers=None, df_descartados=None, ngrams_dict=None, topic_catalog=None, comp_results=None, palette="okabe-ito", output_path=None):
    _pal = get_palette_colors(palette)
    pal_name = _pal["name"]
    OI_SKY_BLUE     = _pal["sky_blue"]
    OI_ORANGE       = _pal["orange"]
    OI_BLUISH_GREEN = _pal["bluish_green"]
    OI_YELLOW       = _pal["yellow"]
    OI_BLUE         = _pal["blue"]
    OI_VERMILION    = _pal["vermilion"]
    OI_PURPLE       = _pal["purple"]
    OI_GREY         = _pal["grey"]

    if df_final is None:
        raise ValueError("df_final es requerido para generar el tablero HTML.")
    df = df_final.copy()

    # --------------------------------------------------------------------------
    # CÁLCULO DE SIMILITUD CON EL CONCEPTO SINTÉTICO 'PRECIO, VALOR, COSTO' (PUNTO 8)
    # Conserva la similitud de embeddings densos si ya viene calculada en df
    # --------------------------------------------------------------------------
    if "similitud_precio" in df.columns and df["similitud_precio"].notna().any():
        df["similitud_precio"] = pd.to_numeric(df["similitud_precio"], errors="coerce").fillna(0.0)
    else:
        price_query = (
            "precio valor costo caro carisimo barato baratisimo economico accesible "
            "tarifa cobro pagar billete propina gasto entrada desquitar vale pena dinero "
            "sobreprecio estafa cuenta presupuesto abuso ticket price cost expensive cheap affordable"
        )
        vec = TfidfVectorizer(max_features=2500)
        X_tfidf = vec.fit_transform(df["texto_limpio"].fillna(""))
        q_vec = vec.transform([price_query])
        df["similitud_precio"] = cosine_similarity(X_tfidf, q_vec).flatten()

    _p_mean = float(df["similitud_precio"].mean())
    _p_std = float(df["similitud_precio"].std())
    _p_thr = round(max(0.20, _p_mean + 0.5 * _p_std), 4)
    df["relacionado_precio"] = (df["similitud_precio"] >= _p_thr).astype(int)

    # --------------------------------------------------------------------------
    # PUNTO 1 (1.1 - 1.4): NUBE, SENTIMIENTO GLOBAL, BRECHA DMO VS UGC, AFINIDAD
    # --------------------------------------------------------------------------
    from collections import Counter
    oi_cycle = [OI_SKY_BLUE, OI_BLUISH_GREEN, OI_BLUE, OI_PURPLE, OI_ORANGE, OI_VERMILION, OI_YELLOW]

    def _build_group_cloud(sub_df, label_desc):
        wc = Counter()
        for txt in sub_df["texto_limpio"].fillna("").astype(str):
            wc.update(txt.split())
        words = [
            (w, int(c), oi_cycle[idx % len(oi_cycle)], f"Frecuencia en {label_desc}")
            for idx, (w, c) in enumerate(wc.most_common(26))
        ]
        return build_clean_wordcloud_full(words, total_w=300, total_h=200)

    svg_wordcloud_ugc = _build_group_cloud(df[df["fuente"] != "sectur"], "UGC")
    svg_wordcloud_dmo = _build_group_cloud(df[df["fuente"] == "sectur"], "DMO (SECTUR)")

    # 1.2: Porcentaje global de comentarios positivos vs negativos
    sent_global = df["sentimiento"].value_counts(normalize=True) * 100
    global_sent_pcts = [
        round(float(sent_global.get("positivo", 0.0)), 1),
        round(float(sent_global.get("negativo", 0.0)), 1),
    ]

    # 1.4: Similitud coseno DMO vs cada canal UGC (sobre texto_limpio)
    _present_ugc = [str(s) for s in ["tripadvisor", "instagram", "reddit"] if s in df["fuente"].values]
    _extra_ugc = sorted([str(s) for s in df["fuente"].dropna().unique() if str(s) not in _present_ugc and str(s) != "sectur"])
    ugc_channels_all = _present_ugc + _extra_ugc

    vec_ch = TfidfVectorizer(max_features=5000, min_df=1 if len(df) < 300 else 3)
    X_ch = vec_ch.fit_transform(df["texto_limpio"].fillna(""))
    dmo_mask_ch = (df["fuente"] == "sectur").to_numpy()
    c_dmo_vec = np.asarray(X_ch[dmo_mask_ch].mean(axis=0)) if dmo_mask_ch.any() else None
    channel_sim_names = [s.capitalize() if s != "tripadvisor" else "TripAdvisor" for s in ugc_channels_all]
    channel_sim_vals = []
    for src_k in ugc_channels_all:
        src_mask_ch = (df["fuente"] == src_k).to_numpy()
        if c_dmo_vec is not None and src_mask_ch.any():
            c_src_vec = np.asarray(X_ch[src_mask_ch].mean(axis=0))
            sim_v = float(cosine_similarity(c_dmo_vec, c_src_vec)[0, 0]) * 100.0
        else:
            sim_v = 0.0
        channel_sim_vals.append(round(sim_v, 1))

    # --------------------------------------------------------------------------
    # PUNTO 2: ANÁLISIS DE N-GRAMAS (DESDE OUTLIERS REALES DE ESTA CORRIDA)
    # --------------------------------------------------------------------------
    df_outliers_use = df_outliers if df_outliers is not None else pd.DataFrame()

    def _top_ngrams_from_outliers(n_val, *keys):
        if ngrams_dict:
            for k in keys:
                if k in ngrams_dict and len(ngrams_dict[k]) > 0:
                    return [
                        {"ngram": r.get("expresion") or r.get("ngram", ""), "frec": int(r["frecuencia"])}
                        for r in ngrams_dict[k][:7]
                    ]
        if len(df_outliers_use) > 0 and "texto_limpio" in df_outliers_use.columns:
            from sklearn.feature_extraction.text import CountVectorizer
            texts_o = df_outliers_use["texto_limpio"].fillna("").astype(str).tolist()
            cv = CountVectorizer(ngram_range=(n_val, n_val), min_df=1 if len(df_outliers_use) < 20 else 2)
            try:
                Xo = cv.fit_transform(texts_o)
                freqs = np.asarray(Xo.sum(axis=0)).ravel()
                terms = np.array(cv.get_feature_names_out())
                top_i = np.argsort(freqs)[::-1][:7]
                return [{"ngram": str(terms[i]), "frec": int(freqs[i])} for i in top_i if freqs[i] > 0]
            except ValueError:
                pass
        return []

    uni_ngrams = _top_ngrams_from_outliers(1, "unigrams", "unigramas")
    bi_ngrams = _top_ngrams_from_outliers(2, "bigrams", "bigramas")
    tri_ngrams = _top_ngrams_from_outliers(3, "trigrams", "trigramas")

    # --------------------------------------------------------------------------
    # PUNTO 3: SENTIMIENTO POR FUENTE
    # --------------------------------------------------------------------------
    sent_by_src = pd.crosstab(df["fuente"], df["sentimiento"], normalize="index") * 100
    sources_list = (["sectur"] if "sectur" in df["fuente"].values else []) + ugc_channels_all
    pos_pcts = [round(float(sent_by_src.loc[s, "positivo"]), 1) if s in sent_by_src.index and "positivo" in sent_by_src.columns else 90.0 for s in sources_list]
    neg_pcts = [round(float(sent_by_src.loc[s, "negativo"]), 1) if s in sent_by_src.index and "negativo" in sent_by_src.columns else 10.0 for s in sources_list]

    # --------------------------------------------------------------------------
    # PUNTO 4: MODELADO DE TÓPICOS POR SENTIMIENTO (100% ALGORÍTMICO, SIN NOMBRES MANUALES)
    # --------------------------------------------------------------------------
    def _top_kw_for_topic(sub_df, col_name, val, k=5):
        wc = Counter()
        for txt in sub_df[sub_df[col_name] == val]["texto_limpio"].fillna("").astype(str):
            wc.update(txt.split())
        return ", ".join([w for w, _ in wc.most_common(k)])

    def _src_dist_for_topic(sub_df, col_name, val):
        s_df = sub_df[sub_df[col_name] == val]
        if len(s_df) == 0:
            return ""
        vc = s_df["fuente"].astype(str).str.upper().value_counts(normalize=True) * 100
        return " · ".join([f"{k} {v:.1f}%" for k, v in vc.items()])

    def _summarize_partition_for_p4(sub_df, fallback_name, max_bars):
        if len(sub_df) == 0:
            return [], [], [], [], [], []
        counts = sub_df["nombre_topico"].value_counts(normalize=True) * 100
        # Condición especial: si la partición cayó en modo frecuencia (< min_partition_size),
        # desplegar los términos más frecuentes como barras individuales en lugar de 1 barra genérica.
        if len(counts) == 1 and str(counts.index[0]) == fallback_name:
            wc_f = Counter()
            doc_sets = sub_df["texto_limpio"].fillna("").astype(str).str.split().map(set).tolist()
            for s in doc_sets:
                wc_f.update(s)
            top_terms = wc_f.most_common(max_bars)
            names, ids, pcts, kws, srcs, match_terms = [], [], [], [], [], []
            n_sub = max(len(sub_df), 1)
            for term, cnt in top_terms:
                pct_term = round(float(cnt) / n_sub * 100.0, 2)
                mask_term = sub_df["texto_limpio"].fillna("").astype(str).str.split().map(lambda ws, t=term: t in ws)
                sub_t = sub_df[mask_term]
                vc_s = sub_t["fuente"].astype(str).str.upper().value_counts(normalize=True) * 100
                src_str = " · ".join([f"{k} {v:.1f}%" for k, v in vc_s.items()])
                names.append(f"Frec_{term} ({pct_term:.1f}%)")
                ids.append(f"F:{term}")
                pcts.append(pct_term)
                kws.append(f"Modo frecuencia (partición < umbral): {cnt} docs con '{term}'")
                srcs.append(src_str)
                match_terms.append(("term", term))
            return names, ids, pcts, kws, srcs, match_terms

        top_c = counts
        names = [f"{k} ({v:.1f}%)" for k, v in top_c.items()]
        ids = [str(k).split("_")[0] for k in top_c.index]
        pcts = [round(float(v), 2) for v in top_c.values]
        kws = [_top_kw_for_topic(sub_df, "nombre_topico", k, 5) for k in top_c.index]
        srcs = [_src_dist_for_topic(sub_df, "nombre_topico", k) for k in top_c.index]
        match_terms = [("topic", k) for k in top_c.index]
        return names, ids, pcts, kws, srcs, match_terms

    df_pos = df[df["sentimiento"] == "positivo"]
    topics_pos_names, topics_pos_ids, topics_pos_pcts, topics_pos_kw, topics_pos_src, pos_match_specs = _summarize_partition_for_p4(
        df_pos, "Frecuencia_Positivo", 5
    )
    n_pos_total = len(df_pos)

    df_neg = df[df["sentimiento"] == "negativo"]
    topics_neg_names, topics_neg_ids, topics_neg_pcts, topics_neg_kw, topics_neg_src, neg_match_specs = _summarize_partition_for_p4(
        df_neg, "Frecuencia_Negativo", 5
    )
    n_neg_total = len(df_neg)

    # --------------------------------------------------------------------------
    # PUNTO 5: COMPARACIÓN DMO VS UGC (NOMBRES ALGORÍTMICOS POR TOP-2 PALABRAS)
    # --------------------------------------------------------------------------
    dmo_dist = df[df["fuente"] == "sectur"]["topico_6"].value_counts(normalize=True) * 100
    ugc_dist = df[df["fuente"] != "sectur"]["topico_6"].value_counts(normalize=True) * 100
    dmo_shares = [round(float(dmo_dist.get(i, 0.0)), 2) for i in range(6)]
    ugc_shares = [round(float(ugc_dist.get(i, 0.0)), 2) for i in range(6)]

    topic_gap_ids = [f"T{i}" for i in range(6)]
    topic_gap_labels = []
    radar_short_labels = []
    for i in range(6):
        wc_t = Counter()
        for txt in df[df["topico_6"] == i]["texto_limpio"].fillna("").astype(str):
            wc_t.update(txt.split())
        top2 = [w for w, _ in wc_t.most_common(2)]
        slug_t = "_".join(top2) if top2 else f"eje_{i}"
        topic_gap_labels.append(f"T{i}: {slug_t}")
        radar_short_labels.append(f"T{i}: {top2[0]}" if top2 else f"T{i}")

    from sklearn.decomposition import NMF as _NMF_13
    _vec6_13 = TfidfVectorizer(max_features=3000, min_df=3)
    _X6_13 = _vec6_13.fit_transform(df["texto_limpio"].fillna("").astype(str))
    _nmf6_13 = _NMF_13(n_components=6, random_state=42, init="nndsvda", max_iter=300).fit(_X6_13)
    _terms6_13 = _vec6_13.get_feature_names_out()
    topic_gap_words = [
        ", ".join([str(_terms6_13[j]) for j in np.argsort(comp)[::-1][:6]])
        for comp in _nmf6_13.components_
    ]

    topic_gap_vals = [round(ugc_shares[i] - dmo_shares[i], 2) for i in range(6)]

    # 2 comentarios representativos por tópico (para despliegue al hacer clic en 1.3)
    # Criterio: 50% cercanía al centroide del grupo en el tópico + 50% vocabulario único (sin truncar texto)
    _vec_13 = TfidfVectorizer(sublinear_tf=True, min_df=3)
    X_cent = _vec_13.fit_transform(df["texto_limpio"].fillna("").astype(str))
    uniq_vocab_df = (
        df["texto_limpio"].fillna("").astype(str).str.split().map(lambda w: len(set(w))).to_numpy(dtype=float)
    )

    def _norm_01(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return arr
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 0 else np.ones_like(arr)

    def _pick_best_rep(mask_bool, label_fn, color_hex):
        idx_arr = np.where(mask_bool)[0]
        if len(idx_arr) == 0:
            return None
        sub_X = X_cent[idx_arr]
        cent = np.asarray(sub_X.mean(axis=0))
        sims = cosine_similarity(sub_X, cent).ravel()
        vocs = uniq_vocab_df[idx_arr]
        scores = 0.5 * _norm_01(sims) + 0.5 * _norm_01(vocs)
        best_pos = int(np.argmax(scores))
        r = df.iloc[int(idx_arr[best_pos])]
        txt = str(r["texto_limpio"]).replace('"', '&quot;').replace("\n", " ").strip()
        return {
            "fuente": label_fn(r),
            "sentimiento": f"{str(r['sentimiento']).capitalize()} · Sim: {sims[best_pos]:.3f} · Vocab: {int(vocs[best_pos])} pal.",
            "color": color_hex,
            "texto": txt,
        }

    topic_gap_comments = []
    is_dmo_arr = (df["fuente"] == "sectur").to_numpy()
    topico_arr = df["topico_6"].to_numpy()
    for i in range(6):
        pair = []
        dmo_item = _pick_best_rep(
            (topico_arr == i) & is_dmo_arr,
            lambda _r: "SECTUR (DMO)",
            OI_VERMILION,
        )
        if dmo_item is not None:
            pair.append(dmo_item)
        ugc_item = _pick_best_rep(
            (topico_arr == i) & (~is_dmo_arr),
            lambda r: f"{str(r['fuente']).upper()} (UGC)",
            OI_SKY_BLUE,
        )
        if ugc_item is not None:
            pair.append(ugc_item)
        topic_gap_comments.append(pair)

    # 2 comentarios más representativos por tópico de sentimiento (para despliegue al hacer clic en Cuadro 5)
    topic_sent_comments = {}
    sent_arr = df["sentimiento"].to_numpy()
    nom_top_arr = df["nombre_topico"].to_numpy()
    doc_word_sets = df["texto_limpio"].fillna("").astype(str).str.split().map(set).tolist()
    for sent_val, match_specs, disp_names, col_hex in [
        ("positivo", pos_match_specs, topics_pos_names, OI_BLUISH_GREEN),
        ("negativo", neg_match_specs, topics_neg_names, OI_YELLOW),
    ]:
        for (kind_m, key_m), disp_name in zip(match_specs, disp_names):
            if kind_m == "term":
                idx_arr = np.where([
                    (sent_arr[i] == sent_val) and (key_m in doc_word_sets[i])
                    for i in range(len(df))
                ])[0]
            else:
                idx_arr = np.where((sent_arr == sent_val) & (nom_top_arr == key_m))[0]
            items = []
            if len(idx_arr) > 0:
                sub_X = X_cent[idx_arr]
                cent = np.asarray(sub_X.mean(axis=0))
                sims = cosine_similarity(sub_X, cent).ravel()
                vocs = uniq_vocab_df[idx_arr]
                scores = 0.5 * _norm_01(sims) + 0.5 * _norm_01(vocs)
                for rank_i, best_pos in enumerate(np.argsort(scores)[::-1][:2]):
                    r = df.iloc[int(idx_arr[int(best_pos)])]
                    txt = str(r["texto_limpio"]).replace('"', '&quot;').replace("\n", " ").strip()
                    items.append({
                        "titulo": f"#{rank_i+1} · {disp_name}",
                        "fuente": f"{str(r['fuente']).upper()} · {str(r['sentimiento']).capitalize()} · Sim: {sims[best_pos]:.3f} · Vocab: {int(vocs[best_pos])} pal.",
                        "color": col_hex,
                        "texto": txt,
                    })
            topic_sent_comments[disp_name] = items

    # 2 comentarios más cercanos al centroide + mayor vocabulario único (UGC y DMO, 4 total en 1.4)
    vec_cent = TfidfVectorizer(sublinear_tf=True, min_df=3)
    X_cent = vec_cent.fit_transform(df["texto_limpio"].fillna("").astype(str))
    mask_dmo_c = (df["fuente"] == "sectur").to_numpy()
    mask_ugg_c = ~mask_dmo_c
    if comp_results and "alignment_index" in comp_results:
        alignment_idx = float(comp_results["alignment_index"])
        doc_alignment_idx = float(comp_results.get("doc_alignment_index", alignment_idx))
    elif np.any(mask_dmo_c) and np.any(mask_ugg_c):
        _c_dmo_g = np.asarray(X_cent[mask_dmo_c].mean(axis=0))
        _c_ugc_g = np.asarray(X_cent[mask_ugg_c].mean(axis=0))
        alignment_idx = round(float(cosine_similarity(_c_dmo_g, _c_ugc_g)[0, 0]) * 100.0, 1)
        doc_alignment_idx = round(float(cosine_similarity(X_cent[mask_ugg_c], _c_dmo_g).mean()) * 100.0, 1)
    else:
        alignment_idx = 0.0
        doc_alignment_idx = 0.0

    # Modelado de tópicos INDEPENDIENTE por grupo (DMO vs UGC Global o fuente individual)
    _terms_cent = _vec_13.get_feature_names_out()

    def _fit_group_topics(mask_bool, prefix, label_fn, color_hex, k_target=6):
        idx_grp = np.where(mask_bool)[0]
        n_grp = len(idx_grp)
        if n_grp == 0:
            return [], np.zeros((0, X_cent.shape[1]))
        k_eff = min(k_target, max(1, n_grp // 3)) if n_grp < k_target * 3 else k_target
        sub_X = X_cent[idx_grp]
        W_g = _NMF_13(n_components=k_eff, random_state=42, init="nndsvda", max_iter=300).fit_transform(sub_X)
        lbls_g = np.argmax(W_g, axis=1)
        topics_out = []
        cents_out = []
        for k_i in range(k_eff):
            km = (lbls_g == k_i)
            pct_k = round(float(np.mean(km) * 100.0), 1)
            idx_k = idx_grp[km]
            if len(idx_k) > 0:
                cent_k = np.asarray(X_cent[idx_k].mean(axis=0))
                top_w = [str(_terms_cent[j]) for j in np.argsort(cent_k.ravel())[::-1][:5]]
                mask_k_full = np.zeros(len(df), dtype=bool)
                mask_k_full[idx_k] = True
                rep_k = _pick_best_rep(mask_k_full, label_fn, color_hex)
            else:
                cent_k = np.zeros((1, X_cent.shape[1]))
                top_w = [f"eje_{k_i}"]
                rep_k = None
            cents_out.append(cent_k)
            topics_out.append({
                "id": f"{prefix}{k_i}",
                "short": f"{prefix}{k_i}: {top_w[0]}",
                "pct": pct_k,
                "words": ", ".join(top_w),
                "rep": rep_k,
            })
        return topics_out, np.vstack(cents_out) if cents_out else np.zeros((0, X_cent.shape[1]))

    _dmo_topics_indep, _dmo_cents_indep = _fit_group_topics(
        mask_dmo_c, "D", lambda _r: "SECTUR (DMO)", OI_VERMILION, k_target=6
    )

    def _build_src_profile(mask_bool, label_str):
        sub_p = df[mask_bool]
        n_s = len(sub_p)
        s_dist = sub_p["sentimiento"].value_counts(normalize=True) * 100
        pos_p = round(float(s_dist.get("positivo", 0.0)), 1)
        neg_p = round(float(s_dist.get("negativo", 0.0)), 1)
        if np.any(mask_dmo_c) and np.any(mask_bool):
            _c_d = np.asarray(X_cent[mask_dmo_c].mean(axis=0))
            _c_s = np.asarray(X_cent[mask_bool].mean(axis=0))
            sim_d = round(float(cosine_similarity(_c_d, _c_s)[0, 0]) * 100.0, 1)
        else:
            sim_d = 0.0

        src_topics, src_cents = _fit_group_topics(
            mask_bool, "U", lambda r: f"{str(r['fuente']).upper()} (UGC)", OI_SKY_BLUE, k_target=6
        )

        # Matriz de similitud entre tópicos propios de esta fuente (U) y tópicos propios de DMO (D)
        if len(src_cents) > 0 and len(_dmo_cents_indep) > 0:
            sim_mat = cosine_similarity(src_cents, _dmo_cents_indep) * 100.0
        else:
            sim_mat = np.zeros((len(src_topics), len(_dmo_topics_indep)))

        for ui, ut in enumerate(src_topics):
            if sim_mat.shape[1] > 0:
                best_dj = int(np.argmax(sim_mat[ui]))
                ut["match_id"] = _dmo_topics_indep[best_dj]["id"]
                ut["match_words"] = _dmo_topics_indep[best_dj]["words"]
                ut["match_sim"] = round(float(sim_mat[ui, best_dj]), 1)
                ut["match_rep"] = _dmo_topics_indep[best_dj]["rep"]
            else:
                ut["match_id"] = "-"
                ut["match_words"] = ""
                ut["match_sim"] = 0.0
                ut["match_rep"] = None

        dmo_matches = []
        for dj, dt in enumerate(_dmo_topics_indep):
            if sim_mat.shape[0] > 0:
                best_ui = int(np.argmax(sim_mat[:, dj]))
                dmo_matches.append({
                    "match_id": src_topics[best_ui]["id"],
                    "match_words": src_topics[best_ui]["words"],
                    "match_sim": round(float(sim_mat[best_ui, dj]), 1),
                    "match_rep": src_topics[best_ui]["rep"],
                })
            else:
                dmo_matches.append({"match_id": "-", "match_words": "", "match_sim": 0.0, "match_rep": None})

        cent_pair = []
        src_cent = _pick_best_rep(
            mask_bool,
            lambda r: f"Centroide {label_str} ({str(r['fuente']).upper()})",
            OI_SKY_BLUE,
        )
        if src_cent is not None:
            cent_pair.append(src_cent)
        dmo_cent = _pick_best_rep(
            mask_dmo_c,
            lambda _r: "Centroide DMO (SECTUR)",
            OI_VERMILION,
        )
        if dmo_cent is not None:
            cent_pair.append(dmo_cent)

        return {
            "label": label_str,
            "n": int(n_s),
            "pos_pct": pos_p,
            "neg_pct": neg_p,
            "sim_dmo": sim_d,
            "topics": src_topics,
            "dmo_matches": dmo_matches,
            "centroid_comments": cent_pair,
        }

    src_lower_arr = df["fuente"].astype(str).str.lower().to_numpy()
    q1_dmo_profile = _build_src_profile(mask_dmo_c, "DMO (SECTUR)")
    q1_dmo_profile["topics"] = _dmo_topics_indep
    q1_ugc_sources = [
        s for s in df.loc[~mask_dmo_c, "fuente"].astype(str).str.lower().value_counts().index
    ]
    q1_source_profiles = {
        "GLOBAL": _build_src_profile(mask_ugg_c, "UGC (Global)")
    }
    for s_name in q1_ugc_sources:
        q1_source_profiles[s_name] = _build_src_profile(src_lower_arr == s_name, s_name.upper())
    centroid_pair_comments = []
    for label_c, m_c, col_c in [
        ("Centroide UGC (Imagen Percibida)", mask_ugg_c, OI_SKY_BLUE),
        ("Centroide DMO (Imagen Proyectada)", mask_dmo_c, OI_VERMILION),
    ]:
        if np.any(m_c):
            X_sub_c = X_cent[m_c]
            cent_vec = np.asarray(X_sub_c.mean(axis=0))
            sims_c = cosine_similarity(X_sub_c, cent_vec).ravel()
            orig_indices = np.where(m_c)[0]
            uniq_vocab = np.array([
                len(set(str(df.iloc[idx]["texto_limpio"]).split()))
                for idx in orig_indices
            ], dtype=float)
            sim_n = (sims_c - sims_c.min()) / (sims_c.max() - sims_c.min() + 1e-9)
            voc_n = (uniq_vocab - uniq_vocab.min()) / (uniq_vocab.max() - uniq_vocab.min() + 1e-9)
            comb_score = 0.5 * sim_n + 0.5 * voc_n
            top1_pos = int(np.argmax(comb_score))
            r_best = df.iloc[orig_indices[top1_pos]]
            txt_b = str(r_best["texto_limpio"]).replace('"', '&quot;').replace("\n", " ").strip()
            centroid_pair_comments.append({
                "titulo": f"{label_c} · Sim: {float(sims_c[top1_pos]):.3f} · Vocab: {int(uniq_vocab[top1_pos])}",
                "fuente": str(r_best["fuente"]).upper(),
                "sentimiento": str(r_best["sentimiento"]).capitalize(),
                "color": col_c,
                "texto": txt_b,
            })

    # --------------------------------------------------------------------------
    # PUNTO 6: ANÁLISIS DE OUTLIERS SEMÁNTICOS VÁLIDOS (TEXTO LIMPIO)
    # --------------------------------------------------------------------------
    outlier_audit_pts = []
    if len(df_outliers_use) > 0:
        for _, r in df_outliers_use.iterrows():
            full_txt = str(r.get("texto_limpio", "")).replace('"', '&quot;').replace("\n", " ").strip()
            clean_txt = full_txt
            if len(clean_txt) > 110:
                clean_txt = clean_txt[:110].rsplit(" ", 1)[0] + "..."
            outlier_audit_pts.append({
                "x": round(float(r.get("vocab_overlap_pct", 90.0)), 1),
                "y": round(float(r.get("lang_score", 0.35)), 3),
                "fuente": str(r.get("fuente", "")).lower(),
                "idioma": str(r.get("idioma_detectado", "es")),
                "texto": clean_txt,
                "texto_full": full_txt,
            })

    # --------------------------------------------------------------------------
    # DATOS UMAP 2D REALES COMPLETOS PARA UGG VS DMO (DATASET LIMPIO FINAL)
    # --------------------------------------------------------------------------
    df_umap = df

    dmo_sub = df_umap[df_umap["fuente"] == "sectur"]
    ugg_sub = df_umap[df_umap["fuente"] != "sectur"]

    raw_ugg_x = [round(float(v), 2) for v in ugg_sub["coord_x"]]
    raw_ugg_y = [round(float(v), 2) for v in ugg_sub["coord_y"]]
    raw_ugg_t = [int(v) for v in ugg_sub["topico_6"]]
    raw_dmo_x = [round(float(v), 2) for v in dmo_sub["coord_x"]]
    raw_dmo_y = [round(float(v), 2) for v in dmo_sub["coord_y"]]
    raw_dmo_t = [int(v) for v in dmo_sub["topico_6"]]

    # --------------------------------------------------------------------------
    # PUNTO 7 (CUADRO 7): 7.1 DMO VS UGC + 7.2 TOP 5 PARES UGC / DMO
    # --------------------------------------------------------------------------
    is_price_arr = (df["relacionado_precio"].to_numpy() == 1)
    sim_price_all = df["similitud_precio"].to_numpy(dtype=float)

    def _top_k_price_group(group_mask, k=5, is_dmo=False):
        idx_arr = np.where(is_price_arr & group_mask)[0]
        if len(idx_arr) == 0:
            return []
        sims = sim_price_all[idx_arr]
        vocs = uniq_vocab_df[idx_arr]
        scores = 0.5 * _norm_01(sims) + 0.5 * _norm_01(vocs)
        top_pos = np.argsort(scores)[::-1][:k]
        col = OI_VERMILION if is_dmo else OI_SKY_BLUE
        tag = "DMO" if is_dmo else "UGC"
        out = []
        for rank_i, loc_i in enumerate(top_pos):
            pos_p = int(idx_arr[int(loc_i)])
            r = df.iloc[pos_p]
            clean_t = str(r["texto_limpio"]).replace('"', '&quot;').replace("\n", " ").strip()
            out.append({
                "titulo": f"#{rank_i+1} {tag} · Sim: {float(r['similitud_precio']):.3f} · {int(uniq_vocab_df[pos_p])} pal.",
                "fuente": str(r["fuente"]).upper(),
                "sentimiento": str(r["sentimiento"]).capitalize(),
                "color": col,
                "texto": clean_t,
            })
        return out

    ugc_top5 = _top_k_price_group(~is_dmo_arr, k=5, is_dmo=False)
    dmo_top5 = _top_k_price_group(is_dmo_arr, k=5, is_dmo=True)
    has_price_data = (len(ugc_top5) + len(dmo_top5)) > 0

    price_top5 = []
    price_rows_html = []
    n_price_rows = max(len(ugc_top5), len(dmo_top5))
    for r_i in range(n_price_rows):
        row_cards = []
        if r_i < len(ugc_top5):
            row_cards.append(ugc_top5[r_i])
        if r_i < len(dmo_top5):
            row_cards.append(dmo_top5[r_i])
        cols_css = "1fr 1fr" if len(row_cards) == 2 else "1fr"
        cards_markup = []
        for c_item in row_cards:
            c_idx = len(price_top5)
            price_top5.append(c_item)
            cards_markup.append(f'''
                <div class="doc-card" onclick="openPriceComment({c_idx})" title="Clic para ver el comentario completo" style="min-width:0; min-height:0; border-left:3px solid {c_item['color']};">
                    <div class="doc-header">
                        <span style="color:{c_item['color']}; font-weight:800; font-size:10.5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{c_item["titulo"]}</span>
                        <span class="doc-source-tag" style="font-size:8.5px; flex-shrink:0;">{c_item["fuente"]}</span>
                    </div>
                    <div class="doc-text" style="font-size:11px;">"{c_item['texto']}"</div>
                </div>
            ''')
        price_rows_html.append(
            f'<div style="display:grid; grid-template-columns:{cols_css}; gap:5px; min-height:0; min-width:0;">{"".join(cards_markup)}</div>'
        )
    price_rows_joined = "".join(price_rows_html)

    # 7.1 Comparativa DMO vs UGC en Precio/Valor/Costo: Frecuencia, Tono y Contexto
    def _price_group_stats(sub_df):
        n_tot = len(sub_df)
        p_df = sub_df[sub_df["relacionado_precio"] == 1] if "relacionado_precio" in sub_df.columns else sub_df.iloc[0:0]
        n_p = len(p_df)
        freq_pct = round(n_p / max(n_tot, 1) * 100.0, 1)
        pos_pct = round(float((p_df["sentimiento"] == "positivo").mean() * 100.0), 1) if n_p > 0 else 0.0
        sim_mean = round(float(sub_df["similitud_precio"].mean()), 3) if n_tot > 0 else 0.0
        wc_p = Counter()
        for t in p_df["texto_limpio"].fillna("").astype(str):
            wc_p.update(t.split())
        ctx_str = ", ".join([w for w, _ in wc_p.most_common(6)]) if n_p > 0 else f"Sin menciones sobre umbral (Sim. media: {sim_mean:.3f})"
        return {
            "n_price": int(n_p),
            "n_total": int(n_tot),
            "freq_pct": freq_pct,
            "pos_pct": pos_pct,
            "sim_mean": sim_mean,
            "ctx": ctx_str,
        }

    price_comp_data = {
        "dmo": _price_group_stats(df[df["fuente"] == "sectur"]),
        "ugc": _price_group_stats(df[df["fuente"] != "sectur"]),
    }

    if has_price_data:
        p7_quadrant_html = f'''
                <div id="p7-split-wrap" style="display:grid; grid-template-columns:1fr 1.25fr; gap:10px; width:100%; height:100%; min-height:0;">
                    <div style="display:flex; flex-direction:column; min-width:0; min-height:0; overflow:hidden; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.10); border-radius:6px; padding:6px 8px;">
                        <div style="font-size:10.5px; font-weight:800; color:#FFFFFF; margin-bottom:2px; flex-shrink:0;">
                            7.1 DMO vs UGC: Frecuencia, Tono y Contexto
                        </div>
                        <div id="chart_p7_1" style="width:100%; flex:1; min-height:0;"></div>
                        <div style="display:flex; flex-direction:column; gap:4px; margin-top:4px; flex-shrink:0; font-size:10px; line-height:1.25;">
                            <div style="background:{_hex_to_rgba(OI_VERMILION, 0.10)}; border-left:3px solid {OI_VERMILION}; padding:3px 6px; border-radius:3px;">
                                <span style="color:{OI_VERMILION}; font-weight:800;">Contexto DMO (Sim. media {price_comp_data['dmo']['sim_mean']:.3f}):</span>
                                <span style="color:#FFFFFF;"> {price_comp_data['dmo']['ctx']}</span>
                            </div>
                            <div style="background:{_hex_to_rgba(OI_SKY_BLUE, 0.10)}; border-left:3px solid {OI_SKY_BLUE}; padding:3px 6px; border-radius:3px;">
                                <span style="color:{OI_SKY_BLUE}; font-weight:800;">Contexto UGC (Sim. media {price_comp_data['ugc']['sim_mean']:.3f}):</span>
                                <span style="color:#FFFFFF;"> {price_comp_data['ugc']['ctx']}</span>
                            </div>
                        </div>
                    </div>
                    <div id="price-grid-wrap" class="price-docs-grid" style="min-width:0; min-height:0;">
                        {price_rows_joined}
                    </div>
                </div>
                <div id="price-full-view" class="q1-comments-view"></div>
        '''
    else:
        p7_quadrant_html = '''
                <div id="p7-empty-msg" style="display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; width:100%; height:100%; padding:24px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.14); border-radius:8px;">
                    <div style="font-size:15px; font-weight:800; color:#FFFFFF; margin-bottom:8px;">
                        Sin comentarios relevantes sobre Precio, Valor o Costo
                    </div>
                    <div style="font-size:12.5px; color:#CBD5E1; max-width:520px; line-height:1.45;">
                        Ningún comentario en DMO ni en UGC superó el umbral semántico del concepto "precio / valor / costo" en este dataset, por lo que no es posible generar ni el comparativo 7.1 ni el top 5 del 7.2.
                    </div>
                </div>
        '''

    # --------------------------------------------------------------------------
    # PUNTOS 3 Y 8: TODOS LOS REGISTROS LIMPIOS PARA LOS DOS SCATTER PLOTS (UGC -> SECTUR ENCIMA)
    # --------------------------------------------------------------------------
    _present_srcs = [str(s) for s in df["fuente"].dropna().unique().tolist()]
    _pref_ugc = [s for s in ["tripadvisor", "instagram", "reddit"] if s in _present_srcs]
    _other_ugc = sorted([s for s in _present_srcs if s not in _pref_ugc and s != "sectur"])
    draw_order = _pref_ugc + _other_ugc + (["sectur"] if "sectur" in _present_srcs else [])
    sample_pts = []
    for src in draw_order:
        sub = df[df["fuente"] == src]
        for _, r in sub.iterrows():
            full_txt_s = str(r["texto_limpio"]).replace('"', '&quot;').replace('\n', ' ').strip()
            prev_txt_s = (full_txt_s[:130].rsplit(" ", 1)[0] + "...") if len(full_txt_s) > 130 else full_txt_s
            sample_pts.append({
                "x": round(float(r["coord_x"]), 4),
                "y": round(float(r["coord_y"]), 4),
                "sim_precio": round(float(r["similitud_precio"]), 4),
                "fuente": src,
                "is_dmo": (src == "sectur"),
                "topico": int(r["topico_6"]),
                "sentimiento": r["sentimiento"],
                "menciona_precio": int(bool(r["relacionado_precio"])),
                "texto": prev_txt_s,
                "texto_full": full_txt_s,
            })

    # ==========================================================================
    # CÓDIGO HTML INTEGRAL: PALETA CIENTÍFICA OKABE-ITO (8 COLORES)
    # ==========================================================================
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tablero CDMX: Análisis DMO vs. UGC ({pal_name})</title>
    <!-- Plotly.js para interactividad preatentiva -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {{
            --bg-slate: #0B0F19;      /* Fondo grafito mate antirreflejo */
            --text-main: #FFFFFF;     /* Texto blanco puro de alta visibilidad */
            --text-muted: #FFFFFF;    /* Texto secundario blanco de alta visibilidad */
            --oi-sky: {OI_SKY_BLUE};
            --oi-orange: {OI_ORANGE};
            --oi-green: {OI_BLUISH_GREEN};
            --oi-purple: {OI_PURPLE};
            --oi-vermilion: {OI_VERMILION};
            --oi-blue: {OI_BLUE};
        }}
        html {{
            scroll-behavior: smooth;
            height: 100%;
            background-color: var(--bg-slate);
        }}
        body {{
            margin: 0;
            padding: 0;
            height: 100%;
            background-color: var(--bg-slate);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            overflow-y: scroll;
            scroll-snap-type: y mandatory;
        }}
        * {{
            box-sizing: border-box;
        }}

        /* Header Fijo Ergonómico Antifatiga */
        header {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 42px;
            z-index: 1000;
            background: rgba(11, 15, 25, 0.94);
            color: #FFFFFF;
            padding: 0 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(8px);
        }}
        .brand-title {{
            font-size: 16.5px;
            font-weight: 700;
            letter-spacing: 0.3px;
            color: #FFFFFF;
        }}
        .alignment-pill {{
            background: rgba(86, 180, 233, 0.16);
            border: 1.5px solid {OI_SKY_BLUE};
            color: #FFFFFF;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.2px;
            white-space: nowrap;
        }}
        .alignment-pill strong {{
            color: {OI_YELLOW};
            font-size: 14px;
            font-weight: 900;
        }}
        .q1-ctrl-bar {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 4px;
            flex-shrink: 0;
        }}
        .q1-ctrl-btn {{
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.22);
            color: #FFFFFF;
            padding: 3px 9px;
            border-radius: 5px;
            font-size: 10.5px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
        }}
        .q1-ctrl-btn.active {{
            background: rgba(86, 180, 233, 0.24);
            border-color: {OI_SKY_BLUE};
            color: #FFFFFF;
        }}
        .q1-ctrl-select {{
            background: #131B2E;
            border: 1px solid rgba(255, 255, 255, 0.22);
            color: #FFFFFF;
            color-scheme: dark;
            padding: 3px 7px;
            border-radius: 5px;
            font-size: 10.5px;
            font-weight: 700;
            cursor: pointer;
            max-width: 190px;
        }}
        .q1-ctrl-select option {{
            background-color: #0B0F19;
            color: #FFFFFF;
        }}
        .q1-ctrl-select.active {{
            border-color: {OI_SKY_BLUE};
            background: rgba(86, 180, 233, 0.16);
        }}
        .q1-ctrl-sim {{
            margin-left: auto;
            font-size: 10px;
            font-weight: 700;
            color: {OI_YELLOW};
            white-space: nowrap;
        }}
        .header-btn {{
            background: rgba(255, 255, 255, 0.12);
            color: #FFFFFF;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 4px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
        }}
        .header-btn:hover {{
            background: rgba(255, 255, 255, 0.22);
            border-color: #FFFFFF;
            color: #FFFFFF;
        }}

        /* Contenedor de Pantalla Completa (Scroll Snap: 100vh) */
        .screen-view {{
            height: 100vh;
            min-height: 100vh;
            max-height: 100vh;
            scroll-snap-align: start;
            scroll-snap-stop: always;
            padding: 48px 14px 10px 14px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr auto 1fr;
            gap: 8px 12px;
            box-sizing: border-box;
            background-color: var(--bg-slate);
        }}
        #screen2 {{
            grid-template-rows: auto 1fr auto 1fr;
        }}
        .row-divider-title {{
            grid-column: 1 / -1;
            text-align: center;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 1.4px;
            text-transform: uppercase;
            color: #FFFFFF;
            padding: 2px 0;
            border-top: 1px solid rgba(255, 255, 255, 0.10);
            border-bottom: 1px solid rgba(255, 255, 255, 0.10);
            background: rgba(255, 255, 255, 0.03);
        }}

        /* Sub-sección Visual Sin Delimitar */
        .quadrant {{
            background: transparent;
            border: none;
            box-shadow: none;
            padding: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            height: 100%;
        }}

        /* Contenido del Sub-espacio: Gráfica al 100% */
        .quadrant-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
            height: 100%;
            width: 100%;
            background: transparent;
        }}

        /* Estilos Cuadro 2: Dividido en 3 por el Eje X (1-G, 2-G, 3-G) con Drilldown */
        .ngram-3col-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 8px;
            width: 100%;
            height: 100%;
            box-sizing: border-box;
            transition: all 0.25s ease;
        }}
        .ngram-col-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            padding: 4px 6px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            cursor: pointer;
            transition: background 0.2s, border-color 0.2s, box-shadow 0.2s;
            position: relative;
        }}
        .ngram-col-card:hover {{
            background: rgba(255, 255, 255, 0.07);
            border-color: rgba(255, 255, 255, 0.28);
        }}
        .ngram-col-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12.5px;
            font-weight: 800;
            padding: 2px 6px;
            letter-spacing: 0.3px;
            height: 24px;
            flex-shrink: 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .expand-hint {{
            font-size: 10.5px;
            color: #FFFFFF;
            opacity: 0.85;
            font-weight: 700;
        }}
        .ngram-reset-pill {{
            margin-left: auto;
            background: {_hex_to_rgba(OI_SKY_BLUE, 0.25)};
            color: {OI_SKY_BLUE};
            border: 1px solid {OI_SKY_BLUE};
            border-radius: 4px;
            padding: 3px 10px;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .ngram-reset-pill:hover {{
            background: {OI_SKY_BLUE};
            color: #0B0F19;
        }}

        /* Subdivisión 2x2 limpia para Cuadrante 1 (sin títulos) */
        .q1-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 6px;
            width: 100%;
            height: 100%;
            box-sizing: border-box;
        }}
        .q1-cell {{
            width: 100%;
            height: 100%;
            min-width: 0;
            min-height: 0;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 5px;
        }}
        .q1-comments-view {{
            display: none;
            flex-direction: column;
            gap: 10px;
            width: 100%;
            height: 100%;
            box-sizing: border-box;
        }}
        .q1-comment-card {{
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            cursor: pointer;
            transition: background 0.2s, border-color 0.2s;
            box-sizing: border-box;
        }}
        .q1-comment-card:hover {{
            background: rgba(255, 255, 255, 0.09);
            border-color: {OI_SKY_BLUE};
        }}

        /* Estilo para las Fichas de Citas en Cuadrante 6 (Grid 2×3 espacioso y letra grande) */
        .rep-docs-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: repeat(3, 1fr);
            gap: 5px;
            height: 100%;
            width: 100%;
            box-sizing: border-box;
            overflow: hidden;
        }}
        /* Estilo para las 5 Citas más cercanas a Precio en Cuadrante 7 */
        .price-docs-grid {{
            display: grid;
            grid-template-rows: repeat(5, 1fr);
            gap: 4px;
            height: 100%;
            width: 100%;
            box-sizing: border-box;
            overflow: hidden;
        }}
        .price-docs-grid .doc-card {{
            padding: 4px 10px;
        }}
        .price-docs-grid .doc-header {{
            font-size: 12px;
            margin-bottom: 1px;
        }}
        .price-docs-grid .doc-text {{
            font-size: 12.2px;
            line-height: 1.25;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        .doc-card {{
            position: relative;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 6px;
            padding: 6px 10px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            cursor: pointer;
            transition: background 0.2s, border-color 0.2s, box-shadow 0.2s;
            box-sizing: border-box;
            user-select: none;
        }}
        .doc-card:hover {{
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.35);
        }}
        .doc-card.active {{
            border-color: {OI_SKY_BLUE};
            box-shadow: 0 0 14px {_hex_to_rgba(OI_SKY_BLUE, 0.45)};
        }}
        .doc-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 800;
            margin-bottom: 3px;
            color: #FFFFFF;
            font-size: 13.5px;
        }}
        .doc-source-tag {{
            font-size: 10.5px;
            font-weight: 800;
            color: #FFFFFF;
            background: rgba(255, 255, 255, 0.15);
            padding: 2px 8px;
            border-radius: 3px;
            border: 1px solid rgba(255, 255, 255, 0.25);
        }}
        .doc-text {{
            color: #FFFFFF;
            font-style: italic;
            font-size: 14px;
            line-height: 1.38;
            white-space: normal;
        }}
        .doc-tooltip {{
            display: none;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(11, 15, 25, 0.96);
            border: 1.5px solid {OI_SKY_BLUE};
            border-radius: 8px;
            padding: 8px 14px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.92);
            z-index: 30;
            text-align: center;
            white-space: nowrap;
            backdrop-filter: blur(10px);
            animation: fadeInTooltip 0.16s ease-out;
            pointer-events: none;
        }}
        @keyframes fadeInTooltip {{
            from {{ opacity: 0; transform: translate(-50%, -45%) scale(0.96); }}
            to {{ opacity: 1; transform: translate(-50%, -50%) scale(1); }}
        }}
        .doc-card.active .doc-tooltip {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
    </style>
</head>
<body>

    <!-- ======================================================================= -->
    <!-- HEADER FIJO CON BOTÓN PARA ACTIVAR / OCULTAR LEYENDAS                  -->
    <!-- ======================================================================= -->
    <header>
        <div style="display:flex; align-items:center; gap:14px; min-width:0;">
            <span class="brand-title">TABLERO DECISIONAL CDMX: IMAGEN PROYECTADA VS. IMAGEN PERCIBIDA</span>
            <span class="alignment-pill" title="Similitud coseno global entre el centroide DMO y el centroide percibido (UGC) · Promedio doc: {doc_alignment_idx:.1f}%">
                Índice de Alineación DMO vs UGC: <strong>{alignment_idx:.1f}%</strong>
            </span>
        </div>
        <button id="btn-toggle-legend" onclick="toggleLegends()" class="header-btn" title="Activar o desactivar leyendas en todos los gráficos">
            👁 Ocultar Leyendas
        </button>
    </header>

    <!-- ======================================================================= -->
    <!-- PANTALLA 1 (VISTA 1 - 100vh): PUNTOS 1 AL 4                             -->
    <!-- ======================================================================= -->
    <section id="screen1" class="screen-view">

        <!-- CUADRANTE 1: SUBDIVIDIDO EN 4 (1.1 NUBE UGC, 1.2 NUBE DMO, 1.3 BRECHA DMO VS UGC, 1.4 AFINIDAD) -->
        <article class="quadrant">
            <div class="quadrant-content">
                <div id="q1-grid-wrap" style="display:flex; flex-direction:column; width:100%; height:100%; min-height:0;">
                    <div class="q1-ctrl-bar">
                        <button id="btn-q1-global" class="q1-ctrl-btn active" onclick="renderQ1TopCharts('GLOBAL')">
                            DMO vs UGC (Global)
                        </button>
                        <select id="sel-q1-source" class="q1-ctrl-select" onchange="if(this.value) renderQ1TopCharts(this.value);">
                            <option value="">Fuente UGC...</option>
                            {"".join([f'<option value="{s}">{s.upper()} ({q1_source_profiles[s]["n"]:,})</option>' for s in q1_ugc_sources])}
                        </select>
                        <span id="q1-ctrl-sim-badge" class="q1-ctrl-sim"></span>
                    </div>
                    <div style="flex:1; display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr; gap:6px; min-height:0;">
                        <div class="q1-cell" style="grid-column:1; grid-row:1 / 3;"><div id="chart_p1_3" style="width:100%; height:100%; cursor:pointer;"></div></div>
                        <div class="q1-cell" style="grid-column:2; grid-row:1;"><div id="chart_p1_2_src" style="width:100%; height:100%;"></div></div>
                        <div class="q1-cell" id="q1-gap-summary" onclick="openCentroidComments()" style="grid-column:2; grid-row:2; padding:6px 8px; overflow-y:auto; display:flex; flex-direction:column; justify-content:space-between; gap:4px; cursor:pointer;" title="Clic para ver los comentarios más cercanos al centroide UGC y DMO"></div>
                    </div>
                </div>
                <div id="q1-comments-view" class="q1-comments-view"></div>
            </div>
        </article>

        <!-- CUADRANTE 2: 1/3 NUBES (UGC ARRIBA, DMO ABAJO) | 2/3 SCATTER DMO VS UGC -->
        <article class="quadrant">
            <div class="quadrant-content">
                <div id="q2-split-wrap" style="display:grid; grid-template-columns:1fr 2fr; gap:8px; width:100%; height:100%; min-height:0;">
                    <div style="display:grid; grid-template-rows:1fr 1fr; gap:6px; min-width:0; min-height:0;">
                        <div class="q1-cell" onclick="openCloudCentroid('UGC')" style="display:flex; flex-direction:column; min-width:0; min-height:0; overflow:hidden; padding:3px; cursor:pointer;" title="Clic para ver el comentario centroide UGC (50% vocab + 50% similitud)">
                            <div style="font-size:10px; font-weight:800; color:{OI_SKY_BLUE}; text-align:center; line-height:1.1; flex-shrink:0;">UGC</div>
                            <div style="flex:1; min-height:0;">{svg_wordcloud_ugc}</div>
                        </div>
                        <div class="q1-cell" onclick="openCloudCentroid('DMO')" style="display:flex; flex-direction:column; min-width:0; min-height:0; overflow:hidden; padding:3px; cursor:pointer;" title="Clic para ver el comentario centroide DMO (50% vocab + 50% similitud)">
                            <div style="font-size:10px; font-weight:800; color:{OI_VERMILION}; text-align:center; line-height:1.1; flex-shrink:0;">DMO (SECTUR)</div>
                            <div style="flex:1; min-height:0;">{svg_wordcloud_dmo}</div>
                        </div>
                    </div>
                    <div style="position:relative; min-width:0; min-height:0; width:100%; height:100%;">
                        <div id="chart_p5" style="width:100%; height:100%;"></div>
                        <div id="chart_p5-comment-view" class="q1-comments-view"></div>
                    </div>
                </div>
                <div id="q2-cloud-comment-view" class="q1-comments-view"></div>
            </div>
        </article>

        <div class="row-divider-title">Análisis de Outliers</div>

        <!-- CUADRANTE 3 (ANTES 2): ANÁLISIS DE N-GRAMAS EN OUTLIERS (1-G, 2-G, 3-G) -->
        <article class="quadrant">
            <div class="quadrant-content">
                <div id="ngram_grid_wrap" class="ngram-3col-grid" style="position: relative;">
                    <button id="ngram_badge_reset" class="ngram-reset-pill" onclick="resetNgrams(); event.stopPropagation();" style="display:none; position: absolute; top: 4px; right: 8px; z-index: 15;" title="Volver a ver las 3 columnas de n-gramas">
                        ← Ver los 3 N-gramas
                    </button>
                    <div id="card_uni" class="ngram-col-card" onclick="toggleNgramCol('uni')">
                        <div class="ngram-col-header" style="color:{OI_SKY_BLUE};">
                            <span>1-G UNIGRAMAS</span>
                        </div>
                        <div id="chart_p2_uni" style="width:100%; height:calc(100% - 22px);"></div>
                    </div>
                    <div id="card_bi" class="ngram-col-card" onclick="toggleNgramCol('bi')">
                        <div class="ngram-col-header" style="color:{OI_ORANGE};">
                            <span>2-G BIGRAMAS</span>
                        </div>
                        <div id="chart_p2_bi" style="width:100%; height:calc(100% - 22px);"></div>
                    </div>
                    <div id="card_tri" class="ngram-col-card" onclick="toggleNgramCol('tri')">
                        <div class="ngram-col-header" style="color:{OI_PURPLE};">
                            <span>3-G TRIGRAMAS</span>
                        </div>
                        <div id="chart_p2_tri" style="width:100%; height:calc(100% - 22px);"></div>
                    </div>
                </div>
            </div>
        </article>

        <!-- CUADRANTE 4: ANÁLISIS DE OUTLIERS SEMÁNTICOS -->
        <article class="quadrant">
            <div class="quadrant-content">
                <div id="chart_p6" style="width:100%; height:100%;"></div>
                <div id="chart_p6-comment-view" class="q1-comments-view"></div>
            </div>
        </article>

    </section>

    <!-- ======================================================================= -->
    <!-- PANTALLA 2 (VISTA 2 - 100vh): PUNTOS 5 AL 8                             -->
    <!-- ======================================================================= -->
    <section id="screen2" class="screen-view">

        <div class="row-divider-title">Análisis de Sentimientos</div>

        <!-- CUADRANTE 5 (ANTES 4): MODELADO DE TÓPICOS POR SENTIMIENTO -->
        <article class="quadrant">
            <div class="quadrant-content">
                <div id="p4-chart-wrap" style="display:flex; flex-direction:column; width:100%; height:100%;">
                    <div id="chart_p4" style="width:100%; flex:1; cursor:pointer;"></div>
                </div>
                <div id="p4-comments-view" class="q1-comments-view"></div>
            </div>
        </article>

        <!-- CUADRANTE 6: SCATTER PLOT 1 (ESPACIO SEMÁNTICO 2D) -->
        <article class="quadrant">
            <div class="quadrant-content">
                <div id="chart_p3" style="width:100%; height:100%;"></div>
                <div id="chart_p3-comment-view" class="q1-comments-view"></div>
            </div>
        </article>

        <div class="row-divider-title">Costo / Precio</div>

        <!-- CUADRANTE 7: 7.1 COMPARATIVA DMO VS UGC (FRECUENCIA, TONO, CONTEXTO) | 7.2 TOP 5 PARES UGC/DMO -->
        <article class="quadrant">
            <div class="quadrant-content">
                {p7_quadrant_html}
            </div>
        </article>

        <!-- CUADRANTE 8: SCATTER PLOT 2 (CONCEPTO PRECIO, VALOR, COSTO) -->
        <article class="quadrant">
            <div class="quadrant-content">
                <div id="chart_p8" style="width:100%; height:100%;"></div>
                <div id="chart_p8-comment-view" class="q1-comments-view"></div>
            </div>
        </article>

    </section>

    <!-- ======================================================================= -->
    <!-- RENDERIZADO DE GRÁFICOS PLOTLY EN PALETA OKABE-ITO (8 COLORES CVD-SAFE) -->
    <!-- ======================================================================= -->
    <script>
        const uni_data = {json.dumps(uni_ngrams)};
        const bi_data = {json.dumps(bi_ngrams)};
        const tri_data = {json.dumps(tri_ngrams)};

        const global_sent_pcts = {json.dumps(global_sent_pcts)};
        const topic_gap_ids = {json.dumps(topic_gap_ids)};
        const topic_gap_labels = {json.dumps(topic_gap_labels)};
        const topic_gap_words = {json.dumps(topic_gap_words)};
        const radar_short_labels = {json.dumps(radar_short_labels)};
        const topic_gap_vals = {json.dumps(topic_gap_vals)};
        const q1_dmo_profile = {json.dumps(q1_dmo_profile)};
        const q1_source_profiles = {json.dumps(q1_source_profiles)};
        const topics_pos_ids = {json.dumps(topics_pos_ids)};
        const topics_neg_ids = {json.dumps(topics_neg_ids)};
        const topic_sent_comments = {json.dumps(topic_sent_comments)};
        const topic_gap_comments = {json.dumps(topic_gap_comments)};
        const centroid_pair_comments = {json.dumps(centroid_pair_comments)};
        const channel_sim_names = {json.dumps(channel_sim_names)};
        const channel_sim_vals = {json.dumps(channel_sim_vals)};

        const sources = {json.dumps(sources_list)};
        const pos_pcts = {json.dumps(pos_pcts)};
        const neg_pcts = {json.dumps(neg_pcts)};

        const topics_pos_names = {json.dumps(topics_pos_names)};
        const topics_pos_pcts = {json.dumps(topics_pos_pcts)};
        const topics_pos_kw = {json.dumps(topics_pos_kw)};
        const topics_pos_src = {json.dumps(topics_pos_src)};

        const topics_neg_names = {json.dumps(topics_neg_names)};
        const topics_neg_pcts = {json.dumps(topics_neg_pcts)};
        const topics_neg_kw = {json.dumps(topics_neg_kw)};
        const topics_neg_src = {json.dumps(topics_neg_src)};

        const dmo_shares = {json.dumps(dmo_shares)};
        const ugc_shares = {json.dumps(ugc_shares)};

        const outlier_audit_pts = {json.dumps(outlier_audit_pts)};
        const sample_pts = {json.dumps(sample_pts)};
        const topic6_keywords = {json.dumps(topic_gap_words)};

        // Redimensionar gráficos al scrollear a Pantalla 2
        const obsOptions = {{ root: null, threshold: 0.2 }};
        const screenObserver = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting && entry.target.id === 'screen2') {{
                    Plotly.Plots.resize('chart_p4');
                    Plotly.Plots.resize('chart_p5');
                    Plotly.Plots.resize('chart_p8');
                }}
            }});
        }}, obsOptions);
        screenObserver.observe(document.getElementById('screen2'));

        // FUNCIÓN GLOBAL: CONMUTADOR (TOGGLE) DE LEYENDAS EN TODOS LOS GRÁFICOS
        let legendsVisible = true;
        function toggleLegends() {{
            legendsVisible = !legendsVisible;
            const charts = ['chart_p3', 'chart_p4', 'chart_p5', 'chart_p6', 'chart_p7_1', 'chart_p8'];
            charts.forEach(id => {{
                const el = document.getElementById(id);
                if (el && el.data) {{
                    Plotly.relayout(id, {{ showlegend: legendsVisible }});
                }}
            }});
            const btn = document.getElementById('btn-toggle-legend');
            if (btn) {{
                btn.innerHTML = legendsVisible ? '👁 Ocultar Leyendas' : '👁 Mostrar Leyendas';
            }}
        }}

        // FUNCIÓN GLOBAL: DRILLDOWN / EXPANDIR Y RESTAURAR COLUMNAS DE N-GRAMAS
        let activeNgram = null;
        function toggleNgramCol(type) {{
            if (activeNgram === type) {{
                resetNgrams();
                return;
            }}
            activeNgram = type;
            const cards = {{
                'uni': document.getElementById('card_uni'),
                'bi': document.getElementById('card_bi'),
                'tri': document.getElementById('card_tri')
            }};
            const pill = document.getElementById('ngram_badge_reset');
            if (pill) pill.style.display = 'inline-block';

            const grid = document.getElementById('ngram_grid_wrap');
            grid.style.gridTemplateColumns = '1fr';

            Object.keys(cards).forEach(k => {{
                if (k === type) {{
                    cards[k].style.display = 'flex';
                }} else {{
                    cards[k].style.display = 'none';
                }}
            }});

            setTimeout(() => {{
                Plotly.Plots.resize('chart_p2_' + type);
            }}, 60);
        }}

        function resetNgrams() {{
            activeNgram = null;
            const cards = {{
                'uni': document.getElementById('card_uni'),
                'bi': document.getElementById('card_bi'),
                'tri': document.getElementById('card_tri')
            }};
            const pill = document.getElementById('ngram_badge_reset');
            if (pill) pill.style.display = 'none';

            const grid = document.getElementById('ngram_grid_wrap');
            grid.style.gridTemplateColumns = '1fr 1fr 1fr';

            Object.keys(cards).forEach(k => {{
                cards[k].style.display = 'flex';
            }});

            setTimeout(() => {{
                Plotly.Plots.resize('chart_p2_uni');
                Plotly.Plots.resize('chart_p2_bi');
                Plotly.Plots.resize('chart_p2_tri');
            }}, 60);
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            const softFont = {{ family: "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size: 10, color: "#FFFFFF" }};
            const gridStyle = {{ gridcolor: 'rgba(255, 255, 255, 0.08)', zerolinecolor: 'rgba(255, 255, 255, 0.15)' }};

            let currentQ1Mode = 'GLOBAL';

            function renderQ1TopCharts(modeKey) {{
                currentQ1Mode = q1_source_profiles[modeKey] ? modeKey : 'GLOBAL';
                const prof = q1_source_profiles[currentQ1Mode];
                const dmoTopics = q1_dmo_profile.topics || [];
                const srcTopics = prof.topics || [];
                const dmoMatches = prof.dmo_matches || [];
                const isGlobal = (currentQ1Mode === 'GLOBAL');
                const btnG = document.getElementById('btn-q1-global');
                const selS = document.getElementById('sel-q1-source');
                const simBadge = document.getElementById('q1-ctrl-sim-badge');
                if (btnG) btnG.classList.toggle('active', isGlobal);
                if (selS) {{
                    selS.classList.toggle('active', !isGlobal);
                    if (isGlobal) selS.value = '';
                }}
                if (simBadge) {{
                    simBadge.textContent = `Sim. con DMO: ${{prof.sim_dmo.toFixed(1)}}% · n=${{prof.n.toLocaleString()}}`;
                }}

                // Coincidencia máxima, brecha propia de DMO y brecha propia de Fuente
                let bestU = srcTopics[0] || {{ id: '-', words: '', pct: 0, match_id: '-', match_words: '', match_sim: 0 }};
                let worstU = bestU;
                srcTopics.forEach(ut => {{
                    if (ut.match_sim > bestU.match_sim) bestU = ut;
                    if (ut.match_sim < worstU.match_sim) worstU = ut;
                }});
                let worstDIdx = 0;
                dmoMatches.forEach((dm, j) => {{
                    if (dm.match_sim < dmoMatches[worstDIdx].match_sim) worstDIdx = j;
                }});
                const worstD = dmoTopics[worstDIdx] || {{ id: '-', words: '', pct: 0 }};
                const worstDSim = (dmoMatches[worstDIdx] && dmoMatches[worstDIdx].match_sim) || 0;
                const toneDiff = Number((prof.pos_pct - q1_dmo_profile.pos_pct).toFixed(1));

                // 1.1 Resumen de coincidencias y brechas (tópicos independientes)
                const sumEl = document.getElementById('q1-gap-summary');
                if (sumEl) {{
                    sumEl.innerHTML = `
                        <div style="font-size:9.8px; font-weight:800; color:#FFFFFF; letter-spacing:0.2px;">
                            COINCIDENCIAS Y BRECHAS · DMO vs ${{prof.label}}
                        </div>
                        <div style="background:{_hex_to_rgba(OI_BLUISH_GREEN, 0.13)}; border-left:3px solid {OI_BLUISH_GREEN}; padding:2px 6px; border-radius:3px; font-size:9.2px; line-height:1.22;">
                            <span style="color:{OI_BLUISH_GREEN}; font-weight:800;">Coincidencia (${{bestU.id}} ↔ ${{bestU.match_id}} · ${{bestU.match_sim.toFixed(1)}}%):</span>
                            <span style="color:#FFFFFF;"> ${{bestU.words}}</span>
                        </div>
                        <div style="background:{_hex_to_rgba(OI_VERMILION, 0.12)}; border-left:3px solid {OI_VERMILION}; padding:2px 6px; border-radius:3px; font-size:9.2px; line-height:1.22;">
                            <span style="color:{OI_VERMILION}; font-weight:800;">Brecha DMO (${{worstD.id}} ${{worstD.pct}}% · eco ${{worstDSim.toFixed(1)}}%):</span>
                            <span style="color:#FFFFFF;"> ${{worstD.words}}</span>
                        </div>
                        <div style="background:{_hex_to_rgba(OI_SKY_BLUE, 0.12)}; border-left:3px solid {OI_SKY_BLUE}; padding:2px 6px; border-radius:3px; font-size:9.2px; line-height:1.22;">
                            <span style="color:{OI_SKY_BLUE}; font-weight:800;">Brecha ${{prof.label}} (${{worstU.id}} ${{worstU.pct}}% · eco ${{worstU.match_sim.toFixed(1)}}%):</span>
                            <span style="color:#FFFFFF;"> ${{worstU.words}}</span>
                        </div>
                        <div style="font-size:9px; color:#FFFFFF; border-top:1px solid rgba(255,255,255,0.1); padding-top:2px; display:flex; justify-content:space-between; align-items:center; gap:4px;">
                            <span>Tono +: <b style="color:{OI_VERMILION};">DMO ${{q1_dmo_profile.pos_pct.toFixed(1)}}%</b> vs <b style="color:{OI_SKY_BLUE};">${{prof.label}} ${{prof.pos_pct.toFixed(1)}}%</b> (${{toneDiff >= 0 ? '+' : ''}}${{toneDiff}} pts)</span>
                            <span style="color:#94A3B8; font-size:8.5px; white-space:nowrap;">Clic: ver comentarios ↗</span>
                        </div>
                    `;
                }}

                // 1.2 Gráfica de Barras agrupadas: Sentimiento Positivo vs Negativo (DMO vs UGC/Fuente)
                Plotly.newPlot('chart_p1_2_src', [
                    {{
                        x: ['Positivo (%)', 'Negativo (%)'],
                        y: [q1_dmo_profile.pos_pct, q1_dmo_profile.neg_pct],
                        name: 'DMO (SECTUR)',
                        type: 'bar',
                        marker: {{ color: '{OI_VERMILION}' }},
                        text: [q1_dmo_profile.pos_pct.toFixed(1) + '%', q1_dmo_profile.neg_pct.toFixed(1) + '%'],
                        textposition: 'outside',
                        textfont: {{ size: 9.5, color: '#FFFFFF' }},
                        hoverlabel: {{ bgcolor: '{OI_VERMILION}', bordercolor: '#FFFFFF', font: {{ color: '#FFFFFF', size: 11 }} }},
                        hovertemplate: '<b>DMO (SECTUR)</b><br>%{{x}}: <b>%{{y:.1f}}%</b><extra></extra>'
                    }},
                    {{
                        x: ['Positivo (%)', 'Negativo (%)'],
                        y: [prof.pos_pct, prof.neg_pct],
                        name: prof.label,
                        type: 'bar',
                        marker: {{ color: '{OI_SKY_BLUE}' }},
                        text: [prof.pos_pct.toFixed(1) + '%', prof.neg_pct.toFixed(1) + '%'],
                        textposition: 'outside',
                        textfont: {{ size: 9.5, color: '#FFFFFF' }},
                        hoverlabel: {{ bgcolor: '{OI_SKY_BLUE}', bordercolor: '#000000', font: {{ color: '#000000', size: 11 }} }},
                        hovertemplate: `<b>${{prof.label}}</b><br>%{{x}}: <b>%{{y:.1f}}%</b><extra></extra>`
                    }}
                ], {{
                    barmode: 'group',
                    dragmode: false,
                    margin: {{ l: 30, r: 8, t: 24, b: 22 }},
                    yaxis: {{ range: [0, 108], ticksuffix: '%', fixedrange: true, tickfont: {{ size: 8, color: '#FFFFFF' }}, ...gridStyle }},
                    xaxis: {{ fixedrange: true, tickfont: {{ size: 9, color: '#FFFFFF' }} }},
                    legend: {{ orientation: 'h', y: 1.22, x: 0.0, itemclick: false, itemdoubleclick: false, font: {{ size: 8.5, color: '#FFFFFF' }} }},
                    font: softFont,
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent'
                }}, {{ responsive: true, displayModeBar: false }});

                // 1.3 Tópicos independientes: U0..U5 (Fuente, derecha +) vs D0..D5 (DMO, izquierda -)
                const barIds13 = [...srcTopics.map(t => t.id), ...dmoTopics.map(t => t.id)];
                const barVals13 = [...srcTopics.map(t => t.pct), ...dmoTopics.map(t => -t.pct)];
                const barColors13 = [...srcTopics.map(() => '{OI_SKY_BLUE}'), ...dmoTopics.map(() => '{OI_VERMILION}')];
                const barHover13 = [
                    ...srcTopics.map(t => [
                        `${{t.id}} (${{prof.label}}: ${{t.pct.toFixed(1)}}%)`,
                        t.words,
                        `Mejor match DMO: ${{t.match_id}} (${{t.match_sim.toFixed(1)}}%) · ${{t.match_words}}`
                    ]),
                    ...dmoTopics.map((t, j) => {{
                        const dm = dmoMatches[j] || {{ match_id: '-', match_sim: 0, match_words: '' }};
                        return [
                            `${{t.id}} (DMO: ${{t.pct.toFixed(1)}}%)`,
                            t.words,
                            `Mejor match ${{prof.label}}: ${{dm.match_id}} (${{dm.match_sim.toFixed(1)}}%) · ${{dm.match_words}}`
                        ];
                    }})
                ];
                const maxAbs13 = Math.max(25, ...barVals13.map(v => Math.abs(v))) + 10;

                Plotly.newPlot('chart_p1_3', [{{
                    x: barVals13.map(() => maxAbs13 * 2),
                    base: -maxAbs13,
                    y: barIds13,
                    customdata: barHover13,
                    type: 'bar',
                    orientation: 'h',
                    marker: {{ color: 'rgba(255,255,255,0.001)' }},
                    showlegend: false,
                    hoverlabel: {{
                        bgcolor: barColors13,
                        bordercolor: barVals13.map(v => v >= 0 ? '#000000' : '#FFFFFF'),
                        font: {{ color: barVals13.map(v => v >= 0 ? '#000000' : '#FFFFFF'), size: 11 }}
                    }},
                    hovertemplate: '<b>%{{customdata[0]}}</b><br><b>%{{customdata[1]}}</b><br>%{{customdata[2]}}<extra></extra>'
                }}, {{
                    x: barVals13,
                    y: barIds13,
                    customdata: barHover13,
                    type: 'bar',
                    orientation: 'h',
                    marker: {{ color: barColors13 }},
                    text: barVals13.map(v => Math.abs(v).toFixed(1) + '%'),
                    textposition: 'outside',
                    textfont: {{ size: 8.5, color: '#FFFFFF' }},
                    hoverlabel: {{
                        bordercolor: barVals13.map(v => v >= 0 ? '#000000' : '#FFFFFF'),
                        font: {{ color: barVals13.map(v => v >= 0 ? '#000000' : '#FFFFFF'), size: 11 }}
                    }},
                    hovertemplate: '<b>%{{customdata[0]}}</b><br><b>%{{customdata[1]}}</b><br>%{{customdata[2]}}<extra></extra>'
                }}], {{
                    margin: {{ l: 36, r: 36, t: 6, b: 18 }},
                    barmode: 'overlay',
                    dragmode: false,
                    xaxis: {{ range: [-maxAbs13, maxAbs13], fixedrange: true, zeroline: true, zerolinecolor: '#FFFFFF', tickfont: {{ size: 8, color: '#FFFFFF' }}, ...gridStyle }},
                    yaxis: {{
                        autorange: 'reversed',
                        fixedrange: true,
                        tickvals: barIds13,
                        ticktext: barIds13.map(t => '<b>' + t + '</b>'),
                        ticks: 'outside',
                        ticklen: 4,
                        tickcolor: 'rgba(0,0,0,0)',
                        tickfont: {{ size: 9, color: '#FFFFFF' }}
                    }},
                    showlegend: false,
                    hovermode: 'closest',
                    font: softFont,
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent'
                }}, {{ responsive: true, displayModeBar: false }});

                // Al hacer clic en Ui o Dj de 1.3: despliega su comentario + el de su pareja más cercana en el otro grupo
                const el13 = document.getElementById('chart_p1_3');
                if (el13 && el13.removeAllListeners) el13.removeAllListeners('plotly_click');
                el13.on('plotly_click', function(ev) {{
                    if (!ev || !ev.points || ev.points.length === 0) return;
                    const idx = ev.points[0].pointIndex;
                    const pair = [];
                    if (idx < srcTopics.length) {{
                        const ut = srcTopics[idx];
                        if (ut.rep) pair.push({{ ...ut.rep, head: `${{ut.id}} (${{prof.label}} ${{ut.pct.toFixed(1)}}%): ${{ut.words}}` }});
                        if (ut.match_rep) pair.push({{ ...ut.match_rep, head: `Match DMO ${{ut.match_id}} (Sim ${{ut.match_sim.toFixed(1)}}%): ${{ut.match_words}}` }});
                    }} else {{
                        const dj = idx - srcTopics.length;
                        const dt = dmoTopics[dj];
                        const dm = dmoMatches[dj];
                        if (dt && dt.rep) pair.push({{ ...dt.rep, head: `${{dt.id}} (DMO ${{dt.pct.toFixed(1)}}%): ${{dt.words}}` }});
                        if (dm && dm.match_rep) pair.push({{ ...dm.match_rep, head: `Match ${{prof.label}} ${{dm.match_id}} (Sim ${{dm.match_sim.toFixed(1)}}%): ${{dm.match_words}}` }});
                    }}
                    if (pair.length === 0) return;
                    const view = document.getElementById('q1-comments-view');
                    view.style.gap = '8px';
                    view.innerHTML = pair.map(c => `
                        <div class="q1-comment-card" onclick="closeTopicComments()" style="padding:10px 14px; justify-content:flex-start; overflow-y:auto;" title="Clic para cerrar y volver a los 4 cuadrantes">
                            <div class="doc-header" style="margin-bottom:5px; font-size:11.5px; flex-shrink:0;">
                                <span style="color:${{c.color}}; font-weight:800;">${{c.head}}</span>
                                <span class="doc-source-tag">${{c.fuente}} · ${{c.sentimiento}}</span>
                            </div>
                            <div class="doc-text" style="font-size:18.8px; line-height:1.45;">"${{c.texto}}"</div>
                        </div>
                    `).join('');
                    document.getElementById('q1-grid-wrap').style.display = 'none';
                    view.style.display = 'flex';
                }});
            }}
            window.renderQ1TopCharts = renderQ1TopCharts;

            function closeTopicComments() {{
                document.getElementById('q1-comments-view').style.display = 'none';
                document.getElementById('q1-grid-wrap').style.display = 'flex';
                setTimeout(() => {{
                    Plotly.Plots.resize('chart_p1_2_src');
                    Plotly.Plots.resize('chart_p1_3');
                }}, 40);
            }}
            window.closeTopicComments = closeTopicComments;

            function openCentroidComments() {{
                const prof = q1_source_profiles[currentQ1Mode] || q1_source_profiles['GLOBAL'];
                const itemsC = (prof.centroid_comments && prof.centroid_comments.length > 0) ? prof.centroid_comments : centroid_pair_comments;
                const view = document.getElementById('q1-comments-view');
                view.style.gap = '8px';
                view.innerHTML = itemsC.map(c => `
                    <div class="q1-comment-card" onclick="closeTopicComments()" style="padding:10px 14px; justify-content:flex-start; overflow-y:auto;" title="Clic para cerrar y volver al cuadrante">
                        <div class="doc-header" style="margin-bottom:5px; font-size:12px; flex-shrink:0;">
                            <span style="color:${{c.color}}; font-weight:800;">${{c.titulo}}</span>
                            <span class="doc-source-tag">${{c.fuente}} · ${{c.sentimiento}}</span>
                        </div>
                        <div class="doc-text" style="font-size:18.8px; line-height:1.45;">"${{c.texto}}"</div>
                    </div>
                `).join('');
                document.getElementById('q1-grid-wrap').style.display = 'none';
                view.style.display = 'flex';
            }}
            window.openCentroidComments = openCentroidComments;

            function openCloudCentroid(kind) {{
                const prof = q1_source_profiles[currentQ1Mode] || q1_source_profiles['GLOBAL'];
                const itemsC = (prof.centroid_comments && prof.centroid_comments.length > 0) ? prof.centroid_comments : centroid_pair_comments;
                const c = (kind === 'DMO') ? (itemsC[1] || centroid_pair_comments[1]) : (itemsC[0] || centroid_pair_comments[0]);
                if (!c) return;
                const view = document.getElementById('q2-cloud-comment-view');
                view.innerHTML = `
                    <div class="q1-comment-card" onclick="closeCloudCentroid()" style="padding:14px 18px; justify-content:flex-start; overflow-y:auto;" title="Clic para cerrar y volver a las nubes">
                        <div class="doc-header" style="margin-bottom:8px; font-size:13px; flex-shrink:0;">
                            <span style="color:${{c.color}}; font-weight:800;">${{c.titulo}}</span>
                            <span class="doc-source-tag">${{c.fuente}} · ${{c.sentimiento}}</span>
                        </div>
                        <div class="doc-text" style="font-size:19.8px; line-height:1.48;">"${{c.texto}}"</div>
                    </div>
                `;
                document.getElementById('q2-split-wrap').style.display = 'none';
                view.style.display = 'flex';
            }}
            function closeCloudCentroid() {{
                document.getElementById('q2-cloud-comment-view').style.display = 'none';
                document.getElementById('q2-split-wrap').style.display = 'grid';
                setTimeout(() => Plotly.Plots.resize('chart_p5'), 40);
            }}
            window.openCloudCentroid = openCloudCentroid;
            window.closeCloudCentroid = closeCloudCentroid;

            renderQ1TopCharts('GLOBAL');

            const has_price_data = {json.dumps(has_price_data)};
            const price_top5 = {json.dumps(price_top5)};
            const price_comp_data = {json.dumps(price_comp_data)};

            if (has_price_data) {{
                Plotly.newPlot('chart_p7_1', [
                    {{
                        x: ['Frecuencia tema (%)', 'Tono Positivo (%)'],
                        y: [price_comp_data.dmo.freq_pct, price_comp_data.dmo.pos_pct],
                        name: `DMO (SECTUR · ${{price_comp_data.dmo.n_price}}/${{price_comp_data.dmo.n_total}})`,
                        type: 'bar',
                        marker: {{ color: '{OI_VERMILION}' }},
                        text: [price_comp_data.dmo.freq_pct.toFixed(1) + '%', price_comp_data.dmo.n_price > 0 ? price_comp_data.dmo.pos_pct.toFixed(1) + '%' : 'N/A'],
                        textposition: 'outside',
                        textfont: {{ size: 10, color: '#FFFFFF' }},
                        hoverlabel: {{ bgcolor: '{OI_VERMILION}', bordercolor: '#FFFFFF', font: {{ color: '#FFFFFF', size: 11.5 }} }},
                        hovertemplate: '<b>DMO (SECTUR)</b><br>%{{x}}: <b>%{{text}}</b><extra></extra>'
                    }},
                    {{
                        x: ['Frecuencia tema (%)', 'Tono Positivo (%)'],
                        y: [price_comp_data.ugc.freq_pct, price_comp_data.ugc.pos_pct],
                        name: `UGC (${{price_comp_data.ugc.n_price.toLocaleString()}}/${{price_comp_data.ugc.n_total.toLocaleString()}})`,
                        type: 'bar',
                        marker: {{ color: '{OI_SKY_BLUE}' }},
                        text: [price_comp_data.ugc.freq_pct.toFixed(1) + '%', price_comp_data.ugc.n_price > 0 ? price_comp_data.ugc.pos_pct.toFixed(1) + '%' : 'N/A'],
                        textposition: 'outside',
                        textfont: {{ size: 10, color: '#FFFFFF' }},
                        hoverlabel: {{ bgcolor: '{OI_SKY_BLUE}', bordercolor: '#000000', font: {{ color: '#000000', size: 11.5 }} }},
                        hovertemplate: '<b>UGC (Percibida)</b><br>%{{x}}: <b>%{{text}}</b><extra></extra>'
                    }}
                ], {{
                    barmode: 'group',
                    dragmode: false,
                    margin: {{ l: 34, r: 10, t: 28, b: 26 }},
                    yaxis: {{ range: [0, 105], ticksuffix: '%', fixedrange: true, tickfont: {{ size: 8.5, color: '#FFFFFF' }}, ...gridStyle }},
                    xaxis: {{ fixedrange: true, tickfont: {{ size: 9.5, color: '#FFFFFF' }} }},
                    legend: {{ orientation: 'h', y: 1.18, x: 0.0, itemclick: false, itemdoubleclick: false, font: {{ size: 9.5, color: '#FFFFFF' }} }},
                    font: softFont,
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent'
                }}, {{ responsive: true, displayModeBar: false }});
            }}

            function openPriceComment(idx) {{
                const c = price_top5[idx];
                if (!c) return;
                const view = document.getElementById('price-full-view');
                view.innerHTML = `
                    <div class="q1-comment-card" onclick="closePriceComment()" style="padding:14px 18px; justify-content:flex-start; overflow-y:auto;" title="Clic para cerrar y volver a 7.1 y 7.2">
                        <div class="doc-header" style="margin-bottom:8px; font-size:13px; flex-shrink:0;">
                            <span style="color:${{c.color}}; font-weight:800;">${{c.titulo}}</span>
                            <span class="doc-source-tag">${{c.fuente}} · ${{c.sentimiento}}</span>
                        </div>
                        <div class="doc-text" style="font-size:20px; line-height:1.48;">"${{c.texto}}"</div>
                    </div>
                `;
                document.getElementById('p7-split-wrap').style.display = 'none';
                view.style.display = 'flex';
            }}
            function closePriceComment() {{
                document.getElementById('price-full-view').style.display = 'none';
                document.getElementById('p7-split-wrap').style.display = 'grid';
                setTimeout(() => Plotly.Plots.resize('chart_p7_1'), 40);
            }}
            window.openPriceComment = openPriceComment;
            window.closePriceComment = closePriceComment;

            function _emptyAnn(msg) {{
                return [{{
                    text: msg,
                    xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
                    showarrow: false,
                    font: {{ size: 11, color: '#94A3B8' }}
                }}];
            }}

            // GRÁFICO P2 - UNIGRAMAS (1-G): Azul Cielo
            Plotly.newPlot('chart_p2_uni', [{{
                y: uni_data.map(d => d.ngram).reverse(),
                x: uni_data.map(d => d.frec).reverse(),
                type: 'bar',
                orientation: 'h',
                marker: {{ color: '{OI_SKY_BLUE}' }},
                text: uni_data.map(d => d.frec).reverse(),
                textposition: 'auto',
                textfont: {{ size: 10.5, color: '#FFFFFF' }},
                hovertemplate: '<b>%{{y}}</b>: %{{x}} menciones<extra></extra>'
            }}], {{
                margin: {{ l: 65, r: 12, t: 8, b: 24 }},
                dragmode: false,
                xaxis: {{ title: "Frecuencia", fixedrange: true, visible: uni_data.length > 0, titlefont: {{ size: 10, color: "#FFFFFF" }}, tickfont: {{ size: 9.5, color: "#FFFFFF" }}, ...gridStyle }},
                yaxis: {{ fixedrange: true, visible: uni_data.length > 0, tickfont: {{ size: 12, color: "#FFFFFF" }} }},
                annotations: uni_data.length === 0 ? _emptyAnn('Sin outliers para unigramas') : [],
                showlegend: false,
                hovermode: 'y',
                font: softFont,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent'
            }}, {{ responsive: true, displayModeBar: false }});



            // GRÁFICO P2 - BIGRAMAS (2-G): Naranja Cálido
            Plotly.newPlot('chart_p2_bi', [{{
                y: bi_data.map(d => d.ngram).reverse(),
                x: bi_data.map(d => d.frec).reverse(),
                type: 'bar',
                orientation: 'h',
                marker: {{ color: '{OI_ORANGE}' }},
                text: bi_data.map(d => d.frec).reverse(),
                textposition: 'auto',
                textfont: {{ size: 10.5, color: '#FFFFFF' }},
                hovertemplate: '<b>%{{y}}</b>: %{{x}} menciones<extra></extra>'
            }}], {{
                margin: {{ l: 95, r: 12, t: 8, b: 24 }},
                dragmode: false,
                xaxis: {{ title: "Frecuencia", fixedrange: true, visible: bi_data.length > 0, titlefont: {{ size: 10, color: "#FFFFFF" }}, tickfont: {{ size: 9.5, color: "#FFFFFF" }}, ...gridStyle }},
                yaxis: {{ fixedrange: true, visible: bi_data.length > 0, tickfont: {{ size: 11.5, color: "#FFFFFF" }} }},
                annotations: bi_data.length === 0 ? _emptyAnn('Sin outliers para bigramas') : [],
                showlegend: false,
                hovermode: 'y',
                font: softFont,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent'
            }}, {{ responsive: true, displayModeBar: false }});



            // GRÁFICO P2 - TRIGRAMAS (3-G): Púrpura Rojizo
            Plotly.newPlot('chart_p2_tri', [{{
                y: tri_data.map(d => d.ngram).reverse(),
                x: tri_data.map(d => d.frec).reverse(),
                type: 'bar',
                orientation: 'h',
                marker: {{ color: '{OI_PURPLE}' }},
                text: tri_data.map(d => d.frec).reverse(),
                textposition: 'auto',
                textfont: {{ size: 10.5, color: '#FFFFFF' }},
                hovertemplate: '<b>%{{y}}</b>: %{{x}} menciones<extra></extra>'
            }}], {{
                margin: {{ l: 135, r: 12, t: 8, b: 24 }},
                dragmode: false,
                xaxis: {{ title: "Frecuencia", fixedrange: true, visible: tri_data.length > 0, titlefont: {{ size: 10, color: "#FFFFFF" }}, tickfont: {{ size: 9.5, color: "#FFFFFF" }}, ...gridStyle }},
                yaxis: {{ fixedrange: true, visible: tri_data.length > 0, tickfont: {{ size: 11, color: "#FFFFFF" }} }},
                annotations: tri_data.length === 0 ? _emptyAnn('Sin outliers para trigramas') : [],
                showlegend: false,
                hovermode: 'y',
                font: softFont,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent'
            }}, {{ responsive: true, displayModeBar: false }});



            // GRÁFICO P3: (Movido a Scatter Plot Semántico 2D desde el Cuadro 7)

            // GRÁFICO P4: Modelado de Tópicos por Sentimiento (Positivos vs Negativos)
            const posHoverPairs = topics_pos_names.slice().reverse().map((nm, i) => [nm, topics_pos_kw.slice().reverse()[i], topics_pos_src.slice().reverse()[i]]);
            const negHoverPairs = topics_neg_names.slice().reverse().map((nm, i) => [nm, topics_neg_kw.slice().reverse()[i], topics_neg_src.slice().reverse()[i]]);
            Plotly.newPlot('chart_p4', [
                {{
                    y: topics_pos_names.slice().reverse(),
                    x: topics_pos_pcts.slice().reverse().map(() => 100),
                    type: 'bar',
                    orientation: 'h',
                    marker: {{ color: 'rgba(255,255,255,0.001)' }},
                    showlegend: false,
                    customdata: posHoverPairs,
                    hoverlabel: {{ bgcolor: '{OI_BLUISH_GREEN}', bordercolor: '#FFFFFF', font: {{ color: '#FFFFFF', size: 12 }} }},
                    hovertemplate: '<b>%{{customdata[0]}}</b><br><b>%{{customdata[1]}}</b><br>Fuentes: %{{customdata[2]}}<extra></extra>'
                }},
                {{
                    y: topics_neg_names.slice().reverse(),
                    x: topics_neg_pcts.slice().reverse().map(() => 100),
                    type: 'bar',
                    orientation: 'h',
                    marker: {{ color: 'rgba(255,255,255,0.001)' }},
                    showlegend: false,
                    customdata: negHoverPairs,
                    hoverlabel: {{ bgcolor: '{OI_YELLOW}', bordercolor: '#000000', font: {{ color: '#000000', size: 12 }} }},
                    hovertemplate: '<b>%{{customdata[0]}}</b><br><b>%{{customdata[1]}}</b><br>Fuentes: %{{customdata[2]}}<extra></extra>'
                }},
                {{
                    y: topics_pos_names.slice().reverse(),
                    x: topics_pos_pcts.slice().reverse(),
                    name: 'Tópicos Positivos ({global_sent_pcts[0]:.1f}% · n={n_pos_total:,})',
                    type: 'bar',
                    orientation: 'h',
                    marker: {{ color: '{OI_BLUISH_GREEN}' }},
                    text: topics_pos_pcts.slice().reverse().map(v => v.toFixed(1) + '%'),
                    textposition: 'outside',
                    textfont: {{ size: 10, color: '#FFFFFF' }},
                    customdata: posHoverPairs,
                    hoverlabel: {{ bgcolor: '{OI_BLUISH_GREEN}', bordercolor: '#FFFFFF', font: {{ color: '#FFFFFF', size: 12 }} }},
                    hovertemplate: '<b>%{{customdata[0]}}</b><br><b>%{{customdata[1]}}</b><br>Fuentes: %{{customdata[2]}}<extra></extra>'
                }},
                {{
                    y: topics_neg_names.slice().reverse(),
                    x: topics_neg_pcts.slice().reverse(),
                    name: 'Tópicos Negativos ({global_sent_pcts[1]:.1f}% · n={n_neg_total:,})',
                    type: 'bar',
                    orientation: 'h',
                    marker: {{ color: '{OI_YELLOW}' }},
                    text: topics_neg_pcts.slice().reverse().map(v => v.toFixed(1) + '%'),
                    textposition: 'outside',
                    textfont: {{ size: 10, color: '#FFFFFF' }},
                    customdata: negHoverPairs,
                    hoverlabel: {{ bgcolor: '{OI_YELLOW}', bordercolor: '#000000', font: {{ color: '#000000', size: 12 }} }},
                    hovertemplate: '<b>%{{customdata[0]}}</b><br><b>%{{customdata[1]}}</b><br>Fuentes: %{{customdata[2]}}<extra></extra>'
                }}
            ], {{
                margin: {{ l: 76, r: 45, t: 10, b: 25 }},
                barmode: 'overlay',
                dragmode: false,
                xaxis: {{ range: [0, 100], ticksuffix: '%', fixedrange: true, tickfont: {{ size: 9, color: "#FFFFFF" }}, ...gridStyle }},
                yaxis: {{
                    fixedrange: true,
                    automargin: true,
                    tickvals: [...topics_pos_names.slice().reverse(), ...topics_neg_names.slice().reverse()],
                    ticktext: [...topics_pos_ids.slice().reverse(), ...topics_neg_ids.slice().reverse()].map(t => '<b>' + t + '</b>'),
                    ticks: 'outside',
                    ticklen: 5,
                    tickcolor: 'rgba(0,0,0,0)',
                    tickfont: {{ size: 10.5, color: "#FFFFFF" }}
                }},
                legend: {{ orientation: 'h', y: 1.15, x: 0.02, itemclick: false, itemdoubleclick: false, font: {{ size: 12, color: "#FFFFFF" }} }},
                hovermode: 'closest',
                font: softFont,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent'
            }}, {{ responsive: true, displayModeBar: false }});

            function closeP4Comments() {{
                document.getElementById('p4-comments-view').style.display = 'none';
                document.getElementById('p4-chart-wrap').style.display = 'flex';
                setTimeout(() => Plotly.Plots.resize('chart_p4'), 40);
            }}
            window.closeP4Comments = closeP4Comments;

            document.getElementById('chart_p4').on('plotly_click', function(ev) {{
                if (!ev || !ev.points || ev.points.length === 0) return;
                const yKey = ev.points[0].y;
                const items = topic_sent_comments[yKey] || [];
                if (items.length === 0) return;
                const view = document.getElementById('p4-comments-view');
                view.style.gap = '8px';
                view.innerHTML = items.map(c => `
                    <div class="q1-comment-card" onclick="closeP4Comments()" style="padding:10px 14px; justify-content:flex-start; overflow-y:auto;" title="Clic para cerrar y volver a los tópicos">
                        <div class="doc-header" style="margin-bottom:5px; font-size:12px; flex-shrink:0;">
                            <span style="color:${{c.color}}; font-weight:800;">${{c.titulo}}</span>
                            <span class="doc-source-tag">${{c.fuente}}</span>
                        </div>
                        <div class="doc-text" style="font-size:19.2px; line-height:1.45;">"${{c.texto}}"</div>
                    </div>
                `).join('');
                document.getElementById('p4-chart-wrap').style.display = 'none';
                view.style.display = 'flex';
            }});



            // GRÁFICO P5: UGG vs DMO (22,113 REGISTROS REALES EXACTOS - COLOR POR TÓPICO, FORMAS Y BORDE BLANCO DMO)
            const uggPts5 = sample_pts.filter(p => !p.is_dmo);
            const dmoPts5 = sample_pts.filter(p => p.is_dmo);

            const topicColors = {{
                0: '#1E3A5F',
                1: '#B84A39',
                2: '#2A7B76',
                3: '#3D7A5A',
                4: '#6C4F77',
                5: '#C87D28'
            }};
            const uggColors = uggPts5.map(p => topicColors[p.topico] || '#1E3A5F');
            const dmoColors = dmoPts5.map(p => topicColors[p.topico] || '#1E3A5F');

            Plotly.newPlot('chart_p5', [
                {{
                    x: uggPts5.map(p => p.x),
                    y: uggPts5.map(p => p.y),
                    mode: 'markers',
                    name: 'UGC (Círculo liso)',
                    type: 'scattergl',
                    marker: {{
                        symbol: 'circle',
                        color: uggColors,
                        size: 3.5,
                        opacity: 0.40
                    }},
                    text: uggPts5.map(p => `<b>[${{p.fuente.toUpperCase()}}]</b> T${{p.topico}} (${{p.sentimiento}})<br><i>Palabras clave: ${{topic6_keywords[p.topico] || ''}}</i><br>${{p.texto}}`),
                    hoverinfo: 'text'
                }},
                {{
                    x: dmoPts5.map(p => p.x),
                    y: dmoPts5.map(p => p.y),
                    mode: 'markers',
                    name: 'DMO (Rombo + Borde blanco)',
                    type: 'scattergl',
                    marker: {{
                        symbol: 'diamond',
                        color: dmoColors,
                        size: 8.5,
                        opacity: 0.95,
                        line: {{ color: '#FFFFFF', width: 1.3 }}
                    }},
                    text: dmoPts5.map(p => `<b>[SECTUR · DMO]</b> T${{p.topico}} (${{p.sentimiento}})<br><i>Palabras clave: ${{topic6_keywords[p.topico] || ''}}</i><br>${{p.texto}}`),
                    hoverinfo: 'text'
                }}
            ], {{
                title: {{ text: '<b>DMO vs UGC</b>', font: {{ size: 15, color: '#FFFFFF' }}, x: 0.04, y: 0.96 }},
                margin: {{ l: 30, r: 15, t: 30, b: 25 }},
                dragmode: 'pan',
                xaxis: {{ title: '', showgrid: true, gridcolor: '#1E293B', zeroline: false, tickfont: {{ size: 8, color: '#94A3B8' }} }},
                yaxis: {{ title: '', showgrid: true, gridcolor: '#1E293B', zeroline: false, tickfont: {{ size: 8, color: '#94A3B8' }} }},
                legend: {{ orientation: 'h', y: 1.14, x: 0.40, font: {{ size: 9.5, color: '#FFFFFF' }} }},
                font: softFont,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent'
            }}, {{ responsive: true, displayModeBar: false, scrollZoom: true }});



            // ORDEN DE DIBUJADO: SIEMPRE UGC PRIMERO Y SECTUR AL FINAL (ENCIMA)
            const drawOrder = {json.dumps(draw_order)};

            // GRÁFICO P6: ANÁLISIS DE OUTLIERS SEMÁNTICOS POR FUENTE (SOLO OUTLIERS VÁLIDOS)
            const outSrcColors = {{
                'tripadvisor': '{OI_SKY_BLUE}',
                'sectur': '{OI_VERMILION}',
                'instagram': '{OI_BLUISH_GREEN}',
                'reddit': '{OI_PURPLE}'
            }};
            const traces_p6 = drawOrder.map(src => {{
                const pts = outlier_audit_pts.filter(p => p.fuente === src);
                return {{
                    x: pts.map(p => p.x),
                    y: pts.map(p => p.y),
                    mode: 'markers',
                    type: 'scattergl',
                    name: src.toUpperCase() + ' (' + pts.length + ')',
                    marker: {{
                        color: outSrcColors[src] || '{OI_SKY_BLUE}',
                        size: src === 'sectur' ? 8 : 5.5,
                        opacity: src === 'sectur' ? 0.95 : 0.65,
                        symbol: src === 'sectur' ? 'diamond' : 'circle',
                        line: src === 'sectur' ? {{ color: '#FFFFFF', width: 1.0 }} : {{ width: 0 }}
                    }},
                    text: pts.map(p => `<b>[${{src.toUpperCase()}}]</b> Overlap: ${{p.x}}% | Idioma: ${{p.idioma}} (${{p.y}})<br>${{p.texto}}`),
                    hoverinfo: 'text'
                }};
            }});
            Plotly.newPlot('chart_p6', traces_p6, {{
                margin: {{ l: 42, r: 15, t: 28, b: 36 }},
                dragmode: 'pan',
                xaxis: {{ title: 'Intersección con vocabulario válido (%)', visible: outlier_audit_pts.length > 0, titlefont: {{ size: 9.5, color: '#FFFFFF' }}, ticksuffix: '%', tickfont: {{ size: 8.5, color: '#FFFFFF' }}, ...gridStyle }},
                yaxis: {{ title: 'Confianza idioma', visible: outlier_audit_pts.length > 0, titlefont: {{ size: 9.5, color: '#FFFFFF' }}, tickfont: {{ size: 8.5, color: '#FFFFFF' }}, ...gridStyle }},
                annotations: outlier_audit_pts.length === 0 ? _emptyAnn('Sin outliers detectados (0)') : [],
                legend: {{ orientation: 'h', y: 1.13, x: 0.05, font: {{ size: 9, color: '#FFFFFF' }} }},
                font: softFont,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent'
            }}, {{ responsive: true, displayModeBar: false, scrollZoom: true }});


            // CONFIGURACIÓN DE GLIFOS PREATENTIVOS EN PALETA OKABE-ITO (4 COLORES Y 4 FORMAS DISTINTAS)
            const glyphConfig = {{
                'sectur':      {{ symbol: 'diamond', color: '{OI_VERMILION}',   name: 'SECTUR (Rombo Bermellón)', size: 8, opacity: 0.95, lineColor: '#FFFFFF', lineWidth: 1.1 }},
                'tripadvisor': {{ symbol: 'circle',  color: '{OI_SKY_BLUE}',    name: 'TripAdvisor (Círculo Azul Cielo)', size: 4, opacity: 0.40, lineColor: '#0B0F19', lineWidth: 0 }},
                'instagram':   {{ symbol: 'square',  color: '{OI_BLUISH_GREEN}',name: 'Instagram (Cuadrado Verde)', size: 5.5, opacity: 0.75, lineColor: '#0B0F19', lineWidth: 0.4 }},
                'reddit':      {{ symbol: 'triangle-up', color: '{OI_PURPLE}',  name: 'Reddit (Triángulo Púrpura)', size: 6.5, opacity: 0.80, lineColor: '#0B0F19', lineWidth: 0.4 }}
            }};

            // GRÁFICO P3: SCATTER PLOT 1 - ESPACIO SEMÁNTICO 2D (TODOS LOS DATOS, SECTUR ENCIMA)
            const traces_p7 = [];
            drawOrder.forEach(src => {{
                const cfg = glyphConfig[src] || {{ symbol: 'circle', color: '{OI_SKY_BLUE}', name: src.toUpperCase() + ' (UGC)', size: 5.5, opacity: 0.65, lineColor: '#0B0F19', lineWidth: 0.4 }};
                const pts = sample_pts.filter(p => p.fuente === src);
                traces_p7.push({{
                    x: pts.map(p => p.x),
                    y: pts.map(p => p.y),
                    mode: 'markers',
                    type: 'scattergl',
                    name: cfg.name,
                    marker: {{
                        symbol: cfg.symbol,
                        color: cfg.color,
                        size: cfg.size,
                        opacity: cfg.opacity,
                        line: {{ color: cfg.lineColor, width: cfg.lineWidth }}
                    }},
                    text: pts.map(p => `<b>[${{src.toUpperCase()}}]</b> T${{p.topico}} (${{p.sentimiento}})<br>${{p.texto}}`),
                    hoverinfo: 'text'
                }});
            }});

            Plotly.newPlot('chart_p3', traces_p7, {{
                margin: {{ l: 30, r: 15, t: 10, b: 35 }},
                dragmode: 'pan',
                xaxis: {{ title: "Dimensión Semántica 1 (UMAP)", titlefont: {{ size: 9, color: "#FFFFFF" }}, zeroline: false, tickfont: {{ size: 8.5, color: "#FFFFFF" }}, ...gridStyle }},
                yaxis: {{ title: "Dimensión Semántica 2 (UMAP)", titlefont: {{ size: 9, color: "#FFFFFF" }}, zeroline: false, tickfont: {{ size: 8.5, color: "#FFFFFF" }}, ...gridStyle }},
                legend: {{ orientation: 'h', y: -0.22, x: 0, font: {{ size: 8.5, color: "#FFFFFF" }} }},
                font: softFont,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent'
            }}, {{ responsive: true, displayModeBar: false, scrollZoom: true }});



            // GRÁFICO P8: SCATTER PLOT 2 - CONCEPTO 'PRECIO, VALOR, COSTO' (TODOS LOS DATOS, SECTUR ENCIMA)
            const traces_p8 = [];
            drawOrder.forEach(src => {{
                const cfg = glyphConfig[src] || {{ symbol: 'circle', color: '{OI_SKY_BLUE}', name: src.toUpperCase() + ' (UGC)', size: 5.5, opacity: 0.65, lineColor: '#0B0F19', lineWidth: 0.4 }};
                const pts = sample_pts.filter(p => p.fuente === src);
                traces_p8.push({{
                    x: pts.map(p => p.x),
                    y: pts.map(p => p.sim_precio),
                    mode: 'markers',
                    type: 'scattergl',
                    name: cfg.name,
                    marker: {{
                        symbol: cfg.symbol,
                        color: cfg.color,
                        size: cfg.size,
                        opacity: cfg.opacity,
                        line: {{ color: cfg.lineColor, width: cfg.lineWidth }}
                    }},
                    text: pts.map(p => `<b>[${{src.toUpperCase()}}]</b> Similitud Precio: ${{p.sim_precio}}<br>¿Menciona Precio?: ${{p.menciona_precio ? 'SÍ' : 'NO'}}<br>${{p.texto}}`),
                    hoverinfo: 'text'
                }});
            }});

            Plotly.newPlot('chart_p8', traces_p8, {{
                margin: {{ l: 40, r: 15, t: 10, b: 35 }},
                dragmode: 'pan',
                xaxis: {{ title: "Dimensión Semántica 1 (UMAP)", titlefont: {{ size: 9, color: "#FFFFFF" }}, zeroline: false, tickfont: {{ size: 8.5, color: "#FFFFFF" }}, ...gridStyle }},
                yaxis: {{ title: "Similitud Coseno con 'Precio, Valor, Costo'", titlefont: {{ size: 9, color: "#FFFFFF" }}, zeroline: false, tickfont: {{ size: 8.5, color: "#FFFFFF" }}, ...gridStyle }},
                legend: {{ orientation: 'h', y: -0.22, x: 0, font: {{ size: 8.5, color: "#FFFFFF" }} }},
                font: softFont,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent'
            }}, {{ responsive: true, displayModeBar: false, scrollZoom: true }});

            function closeScatterComment(chartId) {{
                document.getElementById(chartId + '-comment-view').style.display = 'none';
                document.getElementById(chartId).style.display = 'block';
                setTimeout(() => Plotly.Plots.resize(chartId), 40);
            }}
            window.closeScatterComment = closeScatterComment;

            function attachScatterClick(chartId, resolver) {{
                const el = document.getElementById(chartId);
                if (!el) return;
                let downX = 0, downY = 0, pointClicked = false;
                el.addEventListener('mousedown', function(e) {{
                    if (e.button === 0) {{
                        downX = e.clientX;
                        downY = e.clientY;
                        pointClicked = false;
                    }}
                }});
                el.addEventListener('click', function(e) {{
                    if (e.button !== 0) return;
                    if (Math.hypot(e.clientX - downX, e.clientY - downY) > 4) return;
                    if (e.target && e.target.closest && e.target.closest('.legend')) return;
                    setTimeout(() => {{
                        if (!pointClicked) {{
                            Plotly.relayout(chartId, {{ 'xaxis.autorange': true, 'yaxis.autorange': true }});
                        }}
                        pointClicked = false;
                    }}, 30);
                }});
                el.on('plotly_click', function(ev) {{
                    if (!ev || !ev.points || ev.points.length === 0) return;
                    pointClicked = true;
                    const info = resolver(ev.points[0]);
                    if (!info) return;
                    const view = document.getElementById(chartId + '-comment-view');
                    view.innerHTML = `
                        <div class="q1-comment-card" onclick="closeScatterComment('${{chartId}}')" style="padding:14px 18px; justify-content:flex-start; overflow-y:auto;" title="Clic para cerrar y volver a la gráfica">
                            <div class="doc-header" style="margin-bottom:8px; font-size:13px; flex-shrink:0;">
                                <span style="color:${{info.color}}; font-weight:800;">${{info.title}}</span>
                                <span class="doc-source-tag">${{info.badge}}</span>
                            </div>
                            <div class="doc-text" style="font-size:19.8px; line-height:1.48;">"${{info.textoFull}}"</div>
                        </div>
                    `;
                    el.style.display = 'none';
                    view.style.display = 'flex';
                }});
            }}

            attachScatterClick('chart_p5', pt => {{
                const arr = pt.curveNumber === 0 ? uggPts5 : dmoPts5;
                const d = arr[pt.pointIndex];
                if (!d) return null;
                return {{
                    title: d.is_dmo ? 'SECTUR (DMO)' : `${{d.fuente.toUpperCase()}} (UGC)`,
                    badge: `Tópico T${{d.topico}} (${{topic6_keywords[d.topico] || ''}}) · ${{d.sentimiento}}`,
                    color: d.is_dmo ? '{OI_VERMILION}' : '{OI_SKY_BLUE}',
                    textoFull: d.texto_full
                }};
            }});

            attachScatterClick('chart_p6', pt => {{
                const src = drawOrder[pt.curveNumber];
                const d = outlier_audit_pts.filter(p => p.fuente === src)[pt.pointIndex];
                if (!d) return null;
                return {{
                    title: `${{src.toUpperCase()}} (Outlier Semántico)`,
                    badge: `Overlap: ${{d.x}}% · Idioma: ${{d.idioma}} (${{d.y}})`,
                    color: outSrcColors[src] || '{OI_SKY_BLUE}',
                    textoFull: d.texto_full
                }};
            }});

            attachScatterClick('chart_p3', pt => {{
                const src = drawOrder[pt.curveNumber];
                const d = sample_pts.filter(p => p.fuente === src)[pt.pointIndex];
                if (!d) return null;
                return {{
                    title: `${{src.toUpperCase()}}`,
                    badge: `Tópico T${{d.topico}} (${{topic6_keywords[d.topico] || ''}}) · ${{d.sentimiento}}`,
                    color: glyphConfig[src].color,
                    textoFull: d.texto_full
                }};
            }});

            attachScatterClick('chart_p8', pt => {{
                const src = drawOrder[pt.curveNumber];
                const d = sample_pts.filter(p => p.fuente === src)[pt.pointIndex];
                if (!d) return null;
                return {{
                    title: `${{src.toUpperCase()}} · Similitud Precio: ${{d.sim_precio}}`,
                    badge: `¿Menciona Precio?: ${{d.menciona_precio ? 'SÍ' : 'NO'}} · ${{d.sentimiento}}`,
                    color: glyphConfig[src].color,
                    textoFull: d.texto_full
                }};
            }});


        }});

        window.addEventListener('resize', () => {{
            Plotly.Plots.resize('chart_p1_2_src');
            Plotly.Plots.resize('chart_p1_3');
            Plotly.Plots.resize('chart_p2_uni');
            Plotly.Plots.resize('chart_p2_bi');
            Plotly.Plots.resize('chart_p2_tri');
            Plotly.Plots.resize('chart_p3');
            Plotly.Plots.resize('chart_p4');
            Plotly.Plots.resize('chart_p5');
            Plotly.Plots.resize('chart_p6');
            if (document.getElementById('chart_p7_1')) Plotly.Plots.resize('chart_p7_1');
            Plotly.Plots.resize('chart_p8');
        }});
    </script>
</body>
</html>
"""

    out_html = Path(output_path) if output_path else (Path(__file__).resolve().parent / "resultados" / "tablero_2pantallas_8puntos.html")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_content, encoding="utf-8")
    print(f"[OK] Tablero interactivo (PALETA: {pal_name}) generado en:\n  -> {out_html}")
    return str(out_html)


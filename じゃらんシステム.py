import streamlit as st
import pandas as pd
from janome.tokenizer import Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances_argmin_min
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
import re
import numpy as np
from collections import Counter
from itertools import combinations
import datetime
import networkx as nx
import io

# --- Streamlit UI 設定 ---
st.set_page_config(layout="wide", page_title="統合分析ツール Ver.51")

# --- グローバル設定とヘルパー関数 ---

EMOTION_DICT = {
    '好き': ['好き', '最高', '素晴らしい', '愛用', 'お気に入り', 'リピート', '大満足', '素敵'],
    '喜び': ['嬉しい', '楽しい', '満足', '美味しい', 'うまい', '快適', '気持ちい', '癒される', '面白い'],
    '悲しみ': ['悲しい', '残念', 'がっかり', '寂しい', '切ない'],
    '怒り': ['ひどい', '最悪', '怒り', '不満', 'ありえない', '許せない', 'ムカつく'],
    '恐れ': ['不安', '怖い', '心配']
}

@st.cache_resource
def get_tokenizer(): return Tokenizer()
@st.cache_resource
def get_model(): return SentenceTransformer('all-MiniLM-L6-v2')

def convert_df_to_excel(dfs, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output) as writer:
        for df, sheet in zip(dfs, sheet_names):
            df.to_excel(writer, sheet_name=sheet, index=False)
    return output.getvalue()

@st.cache_data
def load_trend_data(trend_files):
    if not trend_files: return None
    all_trends_list = []
    for file in trend_files:
        try:
            file.seek(0)
            content = file.read().decode('utf-8-sig')
            lines = [line for line in content.splitlines() if line.strip() and line.strip() != ',,']
            if not lines: continue
            date_line_parts = lines[0].split(',')
            if len(date_line_parts) < 2: continue
            date_range_string = date_line_parts[1].strip()
            
            range_parts = date_range_string.split('～')
            start_str = range_parts[0].strip()
            end_str = range_parts[1].strip() if len(range_parts) > 1 else start_str

            def parse_jp_date(date_str):
                clean_str = re.sub(r'\(.*\)', '', date_str).strip()
                try:
                    return pd.to_datetime(clean_str, format='%Y年%m月%d日')
                except ValueError:
                    return None

            trend_start_date = parse_jp_date(start_str)
            trend_end_date = parse_jp_date(end_str)

            if trend_start_date is None: continue
            if trend_end_date is None: trend_end_date = trend_start_date

            try:
                top_start_index = [i for i, s in enumerate(lines) if s.startswith('TOP')][0] + 1
                rising_start_index = [i for i, s in enumerate(lines) if s.startswith('RISING')][0] + 1
            except IndexError: continue

            def process_section(section_lines, trend_type):
                csv_string = "\n".join(section_lines)
                if not csv_string: return None
                df = pd.read_csv(io.StringIO(csv_string), header=None, usecols=[0, 1, 2], names=['Rank', '用語', '検索インタレスト'])
                df['Rank_num'] = pd.to_numeric(df['Rank'].str.extract(r'(\d+)')[0], errors='coerce')
                df = df.dropna(subset=['Rank_num'])
                df['Rank_num'] = df['Rank_num'].astype(int)
                if not df.empty:
                    df['type'] = trend_type
                    df['date'] = trend_start_date
                    df['end_date'] = trend_end_date
                    df['期間'] = date_range_string
                    return df.drop(columns=['Rank_num'])
                return None

            df_top = process_section(lines[top_start_index:rising_start_index-1], 'TOP')
            if df_top is not None: all_trends_list.append(df_top)
            df_rising = process_section(lines[rising_start_index:], 'RISING')
            if df_rising is not None: all_trends_list.append(df_rising)
        except Exception: continue

    if not all_trends_list: return None
    return pd.concat(all_trends_list, ignore_index=True)

def analyze_emotions(text):
    if not isinstance(text, str): return {e: 1.0 for e in EMOTION_DICT.keys()}
    emotion_scores = {}
    for emotion, keywords in EMOTION_DICT.items():
        count = sum(word in text for word in keywords)
        score = float(min(count * 2, 5))
        emotion_scores[emotion] = score if score > 0 else 1.0
    return emotion_scores

def wakati(text, tokenizer):
    if not isinstance(text, str): return []
    # Janomeのエラー対策：解析できないテキストはスキップ
    try:
        return [token.base_form for token in tokenizer.tokenize(text) if token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞']]
    except Exception:
        return []

def find_most_representative_comments(embeddings, labels, comments, cluster_centers=None):
    representative_comments = {}
    representative_indices = {}
    unique_labels = sorted([l for l in np.unique(labels) if l != -1])
    embeddings_dense = embeddings.toarray() if hasattr(embeddings, "toarray") else embeddings
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) == 0: continue
        cluster_embeddings = embeddings_dense[indices]
        center = cluster_centers[label] 
        closest_idx_in_cluster, _ = pairwise_distances_argmin_min(center.reshape(1, -1), cluster_embeddings)
        original_idx = indices[closest_idx_in_cluster[0]]
        representative_comments[label] = comments[original_idx]
        representative_indices[label] = original_idx
    return representative_comments, representative_indices

def calculate_optimal_elbow(k_range, inertias):
    if len(k_range) < 2: return None
    p1 = np.array([k_range[0], inertias[0]])
    p2 = np.array([k_range[-1], inertias[-1]])
    max_dist = 0
    optimal_k = k_range[0]
    optimal_inertia = inertias[0]
    for i, k in enumerate(k_range):
        p0 = np.array([k, inertias[i]])
        dist = np.abs(np.cross(p2-p1, p1-p0)) / np.linalg.norm(p2-p1)
        if dist > max_dist:
            max_dist = dist
            optimal_k = k
            optimal_inertia = inertias[i]
    return optimal_k, optimal_inertia

def get_fps_centroids(X, k):
    """
    最遠点サンプリング(Farthest Point Sampling)を用いてk個の初期セントロイドを決定する
    """
    n_samples = X.shape[0]
    if n_samples <= k: return X.toarray() if hasattr(X, "toarray") else X
    
    if hasattr(X, "toarray"): X_dense = X.toarray()
    else: X_dense = X
    
    centers = []
    rng = np.random.default_rng(42)
    first_idx = rng.integers(n_samples)
    centers.append(X_dense[first_idx])
    
    min_dists = np.linalg.norm(X_dense - X_dense[first_idx], axis=1)
    
    for _ in range(1, k):
        next_idx = np.argmax(min_dists)
        centers.append(X_dense[next_idx])
        new_dists = np.linalg.norm(X_dense - X_dense[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        
    return np.array(centers)

def filter_dataframe(df_initial, config):
    df = df_initial.copy()
    comment_col, date_col = config.get('comment_col'), config.get('date_col')
    if not all([comment_col, date_col]): return pd.DataFrame(columns=df.columns)

    if 'parsed_date_obj' not in df.columns:
        df['parsed_date_obj'] = pd.to_datetime(df[date_col].str.extract(r'(\d{4}[年/]\d{1,2})')[0], format='%Y年%m', errors='coerce')
    df.dropna(subset=['parsed_date_obj', comment_col], inplace=True)

    filter_type = config.get('filter_type')
    if filter_type == '期間で指定' and len(config.get('date_range', [])) == 2:
        start, end = config['date_range']
        df = df[df['parsed_date_obj'].dt.date.between(start, end)]
    elif filter_type == '年月指定' and config.get('ym_list'):
        df = df[df['parsed_date_obj'].dt.strftime('%Y年%m月').isin(config['ym_list'])]
    elif filter_type == '特定期間ごと' and config.get('multi_period'):
        combined = []
        for period in config['multi_period']:
            if len(period['range']) == 2:
                start_dt, end_dt = period['range'][0], period['range'][1]
                period_df = df[df['parsed_date_obj'].dt.date.between(start_dt, end_dt)].copy()
                combined.append(period_df)
        df = pd.concat(combined).drop_duplicates() if combined else pd.DataFrame(columns=df.columns)

    return df

def create_co_occurrence_network(docs, tokenizer, k_for_ranking=3, top_n_words=30, title_prefix=""):
    word_counts = Counter()
    all_words_list = []
    for doc in docs:
        if doc is None or pd.isna(doc): continue
        doc_str = str(doc).strip()
        if not doc_str: continue
        # Janomeのエラー対策
        try:
            words = [token.surface for token in tokenizer.tokenize(doc_str) if token.part_of_speech.split(',')[0] in ['名詞', '形容詞'] and len(token.surface) > 1]
            if words:
                word_counts.update(words)
                all_words_list.append(words)
        except Exception:
            continue

    # Pairs for Network
    co_occurrence_2 = Counter()
    for words in all_words_list:
        for w1, w2 in combinations(sorted(list(set(words))), 2):
            co_occurrence_2[(w1, w2)] += 1
    
    # K-Combinations for Ranking
    co_occurrence_k = Counter()
    if len(docs) > 0 and k_for_ranking > 1:
        for words in all_words_list:
            unique_words_in_doc = list(set(words))
            if len(unique_words_in_doc) >= k_for_ranking:
                for combo in combinations(sorted(unique_words_in_doc), k_for_ranking):
                    co_occurrence_k[combo] += 1
    
    # Basic Dataframes
    df_word_rank = pd.DataFrame(word_counts.most_common(), columns=['単語', '出現回数'])
    df_word_rank['順位'] = df_word_rank.index + 1
    df_cooc_k = pd.DataFrame(co_occurrence_k.most_common(), columns=['combo', 'count'])

    # Prepare Top Words for Network
    top_words_data = word_counts.most_common(top_n_words)
    network_nodes = [word for word, count in top_words_data]
    network_nodes_set = set(network_nodes) 

    # --- Excel Export Data Preparation ---
    excel_rows = []
    limit_rows = k_for_ranking

    for i, (word, count) in enumerate(top_words_data):
        rank = i + 1
        
        partners = []
        for (w1, w2), weight in co_occurrence_2.items():
            partner = None
            if w1 == word: partner = w2
            elif w2 == word: partner = w1
            
            if partner:
                partners.append((partner, weight))
        
        partners.sort(key=lambda x: x[1], reverse=True)
        
        for r_idx in range(limit_rows):
            c_rank = r_idx + 1
            
            if r_idx < len(partners):
                partner_word, c_count = partners[r_idx]
            else:
                partner_word = "なし"
                c_count = 0
            
            excel_rows.append([word, rank, c_count, partner_word, count, c_rank])
    
    df_network_excel = pd.DataFrame(excel_rows, columns=['単語', '全体順位', '共起回数', '共起単語', '出現回数', '共起順位'])
    
    # Graph Construction
    top_edges = co_occurrence_2.most_common(200)
    G = nx.Graph()
    for node in network_nodes: G.add_node(node)
    for (w1, w2), weight in top_edges:
        if w1 in network_nodes and w2 in network_nodes: G.add_edge(w1, w2, weight=weight)
    
    isolated_nodes = [n for n, d in dict(G.degree()).items() if d == 0]
    G.remove_nodes_from(isolated_nodes)
    
    if not G.nodes():
        fig = go.Figure()
        fig.update_layout(title=f'{title_prefix}共起ネットワーク (繋がりなし)', xaxis={"visible": False}, yaxis={"visible": False})
        return fig, df_word_rank, df_cooc_k, df_network_excel

    top_k_coocs = co_occurrence_k.most_common(3)
    cluster_top_coocs_str_list = [f"<b>{i+1}位：{count}回：</b>{'，'.join(combo)}" for i, (combo, count) in enumerate(top_k_coocs)]
    cluster_top_coocs_str = "<br>".join(cluster_top_coocs_str_list) if cluster_top_coocs_str_list else ""

    pos = nx.spring_layout(G, k=0.8, seed=42)
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None]); edge_trace['y'] += tuple([y0, y1, None])
    
    node_trace = go.Scatter(x=[], y=[], textfont=dict(size=10), mode='text', text=[], hoverinfo='text', hovertext=[], marker=dict(size=[], color='skyblue', line_width=2))
    word_counts_dict = dict(word_counts)
    rank_map = {word: rank + 1 for rank, (word, count) in enumerate(top_words_data)}
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x]); node_trace['y'] += tuple([y])
        rank = rank_map.get(node, '')
        count = word_counts_dict.get(node, 0)
        node_trace['text'] += tuple([f"{rank}. {node}<br>({count}回)"])
        hover_text = f'<b>{node} ({count}回)</b>'
        if cluster_top_coocs_str: 
            hover_text += f'<br>--- {k_for_ranking}単語との共起ランキング ---<br>{cluster_top_coocs_str}'
        node_trace['hovertext'] += tuple([hover_text])
        node_trace['marker']['size'] += tuple([0])
        
    layout = go.Layout(
        title=f'{title_prefix}共起ネットワーク',
        showlegend=False, hovermode='closest', 
        margin=dict(b=0,l=0,r=0,t=40), 
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return go.Figure(data=[edge_trace, node_trace], layout=layout), df_word_rank, df_cooc_k, df_network_excel

def get_common_words(docs, tokenizer, top_n=3):
    if not docs: return "---"
    all_words = []
    for text in docs:
        if pd.isna(text) or not str(text).strip(): continue
        # Janomeのエラー対策：IndexError等をキャッチしてスキップ
        try:
            tokens = [tok.base_form for tok in tokenizer.tokenize(str(text)) if tok.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞'] and len(tok.surface) > 1]
            all_words.extend(tokens)
        except Exception:
            continue
    if not all_words: return "---"
    return ', '.join([w for w, f in Counter(all_words).most_common(top_n)])

def analyze_adversative(df, comment_col, adv_config):
    if not adv_config.get('enabled'): return None
    pre_keywords = adv_config.get('pre_keywords', [])
    pre_logic = adv_config.get('pre_logic', 'OR')
    post_keywords = adv_config.get('post_keywords', [])
    post_logic = adv_config.get('post_logic', 'OR')
    split_pattern = adv_config.get('split_pattern', r'(?:しかし)') 
    
    pre_docs = []
    post_docs = []
    original_docs = []
    full_source_docs = [] 
    
    for doc in df[comment_col]:
        if not isinstance(doc, str): continue
        sentences = re.split(r'[。．\.]', doc)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue
            parts = re.split(split_pattern, sentence, maxsplit=1)
            if len(parts) < 2: continue
            pre_text, post_text = parts[0].strip(), parts[1].strip()
            keep_pre = True
            if pre_keywords:
                if pre_logic == 'AND':
                    if not all(k in pre_text for k in pre_keywords): keep_pre = False
                else:
                    if not any(k in pre_text for k in pre_keywords): keep_pre = False
            keep_post = True
            if post_keywords:
                if post_logic == 'AND':
                    if not all(k in post_text for k in post_keywords): keep_post = False
                else:
                    if not any(k in post_text for k in post_keywords): keep_post = False
            if keep_pre and keep_post:
                pre_docs.append(pre_text)
                post_docs.append(post_text)
                original_docs.append(sentence)
                full_source_docs.append(doc) 
    return {'pre_docs': pre_docs, 'post_docs': post_docs, 'original_docs': original_docs, 'full_source_docs': full_source_docs}


# --- Streamlit UI 本体 ---
st.title("統合分析ツール Ver.51")

with st.sidebar:
    st.header("ファイル設定")
    if 'trend_uploader_key' not in st.session_state:
        st.session_state.trend_uploader_key = 0

    def reset_trend_uploader():
        st.session_state.trend_uploader_key += 1

    kanko_file = st.file_uploader("1. 観光口コミファイル", type=["csv", "xlsx"])
    kisho_file = st.file_uploader("2. 気象ファイル", type=["csv", "xlsx"])
    
    trend_files = st.file_uploader(
        "3. 観光トレンドファイル", 
        type=["csv"], 
        accept_multiple_files=True, 
        key=f"trend_uploader_{st.session_state.trend_uploader_key}"
    )
    st.button("トレンドファイル一括削除", on_click=reset_trend_uploader)

df_kanko, df_kisho, df_trend = None, None, None
if kanko_file: df_kanko = pd.read_excel(kanko_file, dtype=str) if kanko_file.name.endswith('xlsx') else pd.read_csv(kanko_file, dtype=str)
if kisho_file: df_kisho = pd.read_excel(kisho_file, dtype=str) if kisho_file.name.endswith('xlsx') else pd.read_csv(kisho_file, dtype=str)
if trend_files: df_trend = load_trend_data(trend_files)

if 'show_results' not in st.session_state: st.session_state.show_results = False

if df_kanko is None:
    st.info("サイドバーから観光口コミファイルをアップロードしてください。")
else:
    required_cols = {"コメント", "評価星", "旅行時期"}
    if not required_cols.issubset(df_kanko.columns):
        st.error(f"必須列不足: {', '.join(required_cols - set(df_kanko.columns))}")
    else:
        comment_col, rating_col, date_col = "コメント", "評価星", "旅行時期"
        st.markdown("---")
        st.header("口コミ分析設定")
        
        analysis_method = st.radio(
            "分析手法", 
            ('SentenceTransformer', 'TF-IDF'), 
            horizontal=True, 
            key='analysis_method',
            captions=[
                "文意で仲間分け。高精度だが時間がかかる。",
                "単語で仲間分け。キーワード重視の高速分析。",
            ]
        )
        
        # --- 分析設定 ---
        with st.expander("分析パラメータ設定", expanded=True):
            tab_cluster_count, tab_seed, tab_adv, tab_elbow, tab_cosine = st.tabs(["分析グループ数", "軸コメント指定(シード)", "逆説調査", "最適数検討", "最低コサイン類似度"])

            if 'use_seed_comment' not in st.session_state: st.session_state.use_seed_comment = False
            
            with tab_seed:
                st.info("特定のコメントを基準（軸）にしてグルーピングを行います。")
                use_seed = st.checkbox("特定のコメントを軸にする", key="use_seed_comment")
                seed_comment_data = None
                
                if use_seed:
                    n_clusters = 1 
                    st.success("※「軸コメント指定」が有効なため、分析グループ数は **1** に固定されます。")
                    
                    st.write("検索したいキーワードを入力 (すべて含む)")
                    sc1, sc2, sc3 = st.columns(3)
                    sk1 = sc1.text_input("検索語1", key="seed_k1")
                    sk2 = sc2.text_input("検索語2", key="seed_k2")
                    sk3 = sc3.text_input("検索語3", key="seed_k3")
                    
                    search_terms = [k for k in [sk1, sk2, sk3] if k]
                    
                    if search_terms:
                        mask = pd.Series([True] * len(df_kanko))
                        for term in search_terms:
                            mask &= df_kanko[comment_col].str.contains(term, na=False)
                        
                        matches = df_kanko[mask][comment_col].unique()
                        
                        if len(matches) > 0:
                            seed_comment_data = st.selectbox("軸にするコメントを選択", matches, key="seed_selector")
                            st.caption(f"選択中: 「{seed_comment_data[:30]}...」")
                        else:
                            st.warning("条件をすべて満たすコメントが見つかりません。")
                else:
                    pass

            with tab_cluster_count:
                if use_seed:
                    st.info(f"軸コメント指定中のため、グループ数は 1 です。")
                else:
                    n_clusters = st.slider("分析グループ数（クラスター数）", 1, 10, 3, key='n_clusters')
            
            with tab_adv:
                st.markdown("#### 逆説（ギャップ）調査")
                use_adv = st.checkbox("逆説調査機能を利用する", key="use_adv")
                adv_config = {'enabled': use_adv, 'pre_keywords': [], 'post_keywords': [], 'pre_logic': 'OR', 'post_logic': 'OR'}
                
                if use_adv:
                    st.markdown("###### 前件(〇〇)の条件")
                    c_pre1, c_pre2 = st.columns([1, 3])
                    pre_log = c_pre1.radio("前件条件", ["いずれか(OR)", "すべて(AND)"], key="adv_pre_logic")
                    c_pk1, c_pk2, c_pk3 = st.columns(3)
                    pre_kws = [k for k in [c_pk1.text_input("前件KW1"), c_pk2.text_input("前件KW2"), c_pk3.text_input("前件KW3")] if k]
                    adv_config['pre_logic'] = 'AND' if "AND" in pre_log else 'OR'
                    adv_config['pre_keywords'] = pre_kws
                    st.markdown("###### 後件(▽▽)の条件")
                    c_post1, c_post2 = st.columns([1, 3])
                    post_log = c_post1.radio("後件条件", ["いずれか(OR)", "すべて(AND)"], key="adv_post_logic")
                    c_pok1, c_pok2, c_pok3 = st.columns(3)
                    post_kws = [k for k in [c_pok1.text_input("後件KW1"), c_pok2.text_input("後件KW2"), c_pok3.text_input("後件KW3")] if k]
                    adv_config['post_logic'] = 'AND' if "AND" in post_log else 'OR'
                    adv_config['post_keywords'] = post_kws

                    st.markdown("---")
                    st.markdown("###### 前件・後件の区切り単語設定")
                    st.caption("※以下のフォームに入力された単語のいずれかで文章が分割されます。")
                    
                    default_words = ["しかし", "けれども", "だが", "なのだが", "ですが", "でも"]
                    custom_split_words = []
                    
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        for i in range(2):
                            val = default_words[i] if i < len(default_words) else ""
                            w = st.text_input(f"区切り語{i+1}", value=val, key=f"sw_{i}")
                            if w: custom_split_words.append(w)
                    with sc2:
                        for i in range(2, 4):
                            val = default_words[i] if i < len(default_words) else ""
                            w = st.text_input(f"区切り語{i+1}", value=val, key=f"sw_{i}")
                            if w: custom_split_words.append(w)
                    with sc3:
                        for i in range(4, 6):
                            val = default_words[i] if i < len(default_words) else ""
                            w = st.text_input(f"区切り語{i+1}", value=val, key=f"sw_{i}")
                            if w: custom_split_words.append(w)
                    
                    if st.checkbox("さらに区切り語を追加する"):
                        sc4, sc5, sc6 = st.columns(3)
                        w_xtra1 = sc4.text_input("追加語1", key="sw_ex1")
                        w_xtra2 = sc5.text_input("追加語2", key="sw_ex2")
                        w_xtra3 = sc6.text_input("追加語3", key="sw_ex3")
                        if w_xtra1: custom_split_words.append(w_xtra1)
                        if w_xtra2: custom_split_words.append(w_xtra2)
                        if w_xtra3: custom_split_words.append(w_xtra3)

                    if custom_split_words:
                        adv_config['split_pattern'] = r'(?:' + '|'.join(map(re.escape, custom_split_words)) + r')'
                    else:
                        adv_config['split_pattern'] = r'(?!x)x'

            with tab_elbow:
                if st.button("📈 エルボーグラフを表示"): st.session_state.show_elbow_graph = True 
                if st.session_state.get('show_elbow_graph', False): 
                    df_for_elbow = st.session_state.get('filtered_df', df_kanko)
                    if len(df_for_elbow) > 10:
                        unique_docs = df_for_elbow[comment_col].drop_duplicates().tolist()
                        with st.spinner("計算中..."):
                            if 'TF-IDF' in analysis_method: embeddings = TfidfVectorizer(tokenizer=lambda x: wakati(x, get_tokenizer())).fit_transform(unique_docs)
                            else: embeddings = get_model().encode(unique_docs, show_progress_bar=True)
                            k_range = list(range(2, 11))
                            inertias = [KMeans(n_clusters=k, random_state=42, n_init='auto').fit(embeddings).inertia_ for k in k_range]
                            optimal_k, _ = calculate_optimal_elbow(k_range, inertias)
                            fig = px.line(x=k_range, y=inertias, title='エルボー法', markers=True, labels={'x':'クラスター数 (k)', 'y':'凝集度 (Inertia)'})
                            if optimal_k: fig.add_vline(x=optimal_k, line_dash="dash", line_color="red")
                            st.plotly_chart(fig, use_container_width=True)
                    else: st.warning("データ不足")
            
            with tab_cosine:
                if 'min_similarity' not in st.session_state: st.session_state.min_similarity = 0.0
                st.session_state.min_similarity = st.slider("最低コサイン類似度", 0.0, 1.0, st.session_state.min_similarity, 0.01, key="min_sim_slider")

        st.markdown("---")
        st.subheader("分析対象の絞り込み")
        
        filter_config = {'comment_col': comment_col, 'date_col': date_col}
        
        if use_adv:
            st.info("※「逆説調査」モードのため、利用可能な絞り込みは「期間」のみです。")
            tab_period = st.tabs(["期間"])[0]
            tab_kw = tab_emotion = None
        else:
            tab_kw, tab_period, tab_emotion = st.tabs(["キーワード", "期間", "感情"])

        filter_config['cluster_keywords'] = []
        filter_config['cluster_emotions'] = []

        if tab_kw:
            with tab_kw:
                st.write("各クラスターのキーワード絞り込み設定")
                for i in range(n_clusters):
                    with st.expander(f"クラスター {i+1} のキーワード", expanded=False):
                        kw_log = st.radio(f"KW条件 (Cluster {i+1})", ["いずれか(OR)", "すべて(AND)"], index=0, horizontal=True, key=f"ckw_log_{i}")
                        ck1, ck2, ck3 = st.columns(3)
                        kw_list = [t for t in [ck1.text_input(f"KW{i+1}-1", key=f"ckw1_{i}"), ck2.text_input(f"KW{i+1}-2", key=f"ckw2_{i}"), ck3.text_input(f"KW{i+1}-3", key=f"ckw3_{i}")] if t]
                        filter_config['cluster_keywords'].append({'logic': kw_log, 'keywords': kw_list})

        with tab_period:
            p_ops = ['指定なし', '期間で指定', '年月指定', '特定期間ごと']
            filter_config['filter_type'] = st.radio("期間設定", p_ops, key="p_filter")
            if 'parsed_date_obj' not in df_kanko.columns:
                df_kanko['parsed_date_obj'] = pd.to_datetime(df_kanko[date_col].str.extract(r'(\d{4}[年/]\d{1,2})')[0], format='%Y年%m', errors='coerce')
            valid_dates = df_kanko['parsed_date_obj'].dropna()
            d_min, d_max = (valid_dates.min().date(), valid_dates.max().date()) if not valid_dates.empty else (datetime.date.today(), datetime.date.today())

            if filter_config['filter_type'] == '期間で指定': filter_config['date_range'] = st.date_input("期間", [d_min, d_max])
            elif filter_config['filter_type'] == '年月指定':
                available_ym = sorted(df_kanko['parsed_date_obj'].dt.strftime('%Y年%m月').unique())
                filter_config['ym_list'] = st.multiselect("年月", available_ym)
            elif filter_config['filter_type'] == '特定期間ごと':
                st.info(f"クラスター数（{n_clusters}）に合わせて設定")
                multi_period_filters = []
                for i in range(n_clusters):
                    with st.container(border=True):
                        st.write(f"期間 {i+1} (クラスター{i+1})")
                        dr = st.date_input(f"日付 {i+1}", [d_min, d_max], key=f"md_{i}")
                        multi_period_filters.append({'range': dr})
                filter_config['multi_period'] = multi_period_filters
        
        if tab_emotion:
            with tab_emotion:
                filter_config['use_emotion'] = True
                st.write("各クラスターの感情値設定")
                for i in range(n_clusters):
                    with st.expander(f"クラスター {i+1} の感情", expanded=False):
                        e_filts = {}
                        cols = st.columns(5)
                        for j, (k, c) in enumerate(zip(EMOTION_DICT.keys(), cols)):
                            with c: e_filts[k] = st.slider(f"{k} (C{i+1})", 1, 5, (1, 5), step=1, key=f"cemo_{i}_{j}")
                        filter_config['cluster_emotions'].append(e_filts)

        
        filtered_df = filter_dataframe(df_kanko, filter_config)
        st.session_state.filtered_df = filtered_df
        filter_config['min_similarity'] = st.session_state.min_similarity
        filter_config['adversative'] = adv_config
        st.session_state.filter_config = filter_config

        # --- 対象数表示（クラスターごとのカウント） ---
        pre_calc_adv_count = 0
        cluster_pre_counts = {}

        if use_adv:
            temp_adv_res = analyze_adversative(filtered_df, comment_col, adv_config)
            pre_calc_adv_count = len(temp_adv_res['original_docs']) if temp_adv_res else 0
            st.info(f"対象文数 (条件合致): **{pre_calc_adv_count}** 件")
        else:
            st.markdown("##### 各クラスターの対象コメント数（絞り込み条件合致数の目安）")
            temp_emo_df = None
            has_emo_setting = any(any((v[0]>1 or v[1]<5) for v in e.values()) for e in filter_config['cluster_emotions'])
            if has_emo_setting:
                scores_list = [analyze_emotions(text) for text in filtered_df[comment_col]]
                temp_emo_df = pd.DataFrame(scores_list, index=filtered_df.index)

            cols = st.columns(min(n_clusters, 4))
            for i in range(n_clusters):
                temp_df_c = filtered_df.copy()
                
                # 1. Period (Specific Mode)
                if filter_config['filter_type'] == '特定期間ごと' and i < len(filter_config['multi_period']):
                     if len(filter_config['multi_period'][i]['range']) == 2:
                        s, e = filter_config['multi_period'][i]['range']
                        temp_df_c = temp_df_c[temp_df_c['parsed_date_obj'].dt.date.between(s, e)]

                # 2. KW
                if i < len(filter_config['cluster_keywords']):
                    ks = filter_config['cluster_keywords'][i]
                    if ks['keywords']:
                        pat = '|'.join([re.escape(k) for k in ks['keywords']])
                        if "AND" in ks['logic']:
                            for k in ks['keywords']: temp_df_c = temp_df_c[temp_df_c[comment_col].str.contains(re.escape(k), na=False)]
                        else:
                            temp_df_c = temp_df_c[temp_df_c[comment_col].str.contains(pat, na=False)]

                # 3. Emotion
                if i < len(filter_config['cluster_emotions']) and temp_emo_df is not None:
                     es = filter_config['cluster_emotions'][i]
                     if any((v[0]>1 or v[1]<5) for v in es.values()):
                        c_valid = temp_emo_df.index.intersection(temp_df_c.index)
                        for em, (mn, mx) in es.items():
                            if mn > 1 or mx < 5:
                                c_valid = c_valid.intersection(temp_emo_df[temp_emo_df[em].between(mn, mx)].index)
                        temp_df_c = temp_df_c.loc[c_valid]
                
                count_val = len(temp_df_c)
                cluster_pre_counts[i] = count_val
                
                with cols[i % 4]:
                     st.metric(f"クラスター{i+1}", f"{count_val}件")

        # --- 分析ボタンエリア ---
        st.markdown("---")
        st.caption("設定が完了したら分析を実行してください")

        # ボタン押下状態を管理する変数を初期化
        execute_analysis = False
        method_selected = None # 'kmeans', 'fps', 'seed'

        # 軸コメント指定(シード)の有無でボタンの表示を切り替え
        if use_seed:
            st.info("※「軸コメント指定」モードのため、ボタンは1つに統合されています。")
            # ボタンを1つにする
            if st.button("分析実行（軸コメント基準）", type="primary", use_container_width=True):
                execute_analysis = True
                method_selected = 'seed'
        else:
            # 通常時は2つのボタンを表示
            st.caption("全体としてどんな傾向の不満・称賛があるか知りたい場合はクリック")
            btn_kmeans = st.button("分析実行：K-means++", type="primary", use_container_width=True)
            
            st.caption("どんなに珍しくても全パターンの意見を網羅したい場合はクリック")
            btn_fps = st.button("分析実行：最遠点サンプリング", use_container_width=True)
            
            if btn_kmeans:
                execute_analysis = True
                method_selected = 'kmeans'
            elif btn_fps:
                execute_analysis = True
                method_selected = 'fps'

        # --- 分析実行ロジック ---
        if execute_analysis:
            st.session_state.show_elbow_graph = False
            st.session_state.cluster_pre_counts = cluster_pre_counts
            df_proc = st.session_state.filtered_df.copy()
            
            # 【重要】軸コメント指定時に、絞り込み条件外であっても強制的に分析データに追加する処理
            if method_selected == 'seed' and seed_comment_data:
                # 分析データ(df_proc)に軸コメントが含まれていない場合
                if seed_comment_data not in df_proc[comment_col].values:
                    # 元データから該当行を取得して強制的に結合
                    seed_row = df_kanko[df_kanko[comment_col] == seed_comment_data].head(1)
                    if not seed_row.empty:
                        df_proc = pd.concat([seed_row, df_proc], ignore_index=True)

            if use_adv:
                # 逆説分析モード
                if pre_calc_adv_count < 1: st.error("条件に合致する文章がありません。")
                else:
                    with st.spinner('逆説構造を抽出中...'):
                        adv_res = analyze_adversative(df_proc, comment_col, adv_config)
                        st.session_state.analysis_results = {'adv_res': adv_res, 'cols': (comment_col, rating_col, date_col), 'mode': 'adversative'}
                        st.session_state.show_results = True
                        st.success("抽出完了")
            else:
                # クラスター分析モード
                if len(df_proc) < 1: st.error("データ不足")
                else:
                    with st.spinner('クラスター分析中...'):
                        # 重複削除とベクトル化
                        uniq_df = df_proc.drop_duplicates(subset=[comment_col]).copy().reset_index(drop=True)
                        docs = uniq_df[comment_col].tolist()
                        
                        if 'TF-IDF' in analysis_method: 
                            vec = TfidfVectorizer(tokenizer=lambda x: wakati(x, get_tokenizer()), min_df=5, max_df=0.5)
                            embs = vec.fit_transform(docs)
                        else: 
                            embs = get_model().encode(docs, show_progress_bar=True)
                        
                        k = n_clusters
                        if len(uniq_df) < k: st.error("データ数がクラスター数未満です"); st.stop()
                        
                        # 初期値設定ロジック
                        init_params = 'k-means++'
                        n_init_val = 10
                        
                        # シード指定がある場合
                        if method_selected == 'seed' and seed_comment_data:
                            if 'TF-IDF' in analysis_method:
                                seed_vec = vec.transform([seed_comment_data]).toarray()
                            else:
                                seed_vec = get_model().encode([seed_comment_data]).reshape(1, -1)
                            
                            init_params = seed_vec # 初期値をシードに固定
                            n_init_val = 1
                        
                        # FPS指定がある場合
                        elif method_selected == 'fps':
                            init_params = get_fps_centroids(embs, k)
                            n_init_val = 1
                        
                        # モデル構築と学習
                        model = KMeans(n_clusters=k, init=init_params, n_init=n_init_val, random_state=42)
                        labels = model.fit_predict(embs)
                        uniq_df['cluster'] = labels
                        
                        # 結果のマージ
                        df_res = pd.merge(df_proc, uniq_df[[comment_col, 'cluster']], on=comment_col, how='left')
                        
                        # --- 代表コメント決定ロジック ---
                        # 通常時の算出（重心に近いもの）
                        rep_coms, rep_inds = find_most_representative_comments(embs, labels, docs, model.cluster_centers_)
                        
                        # 【重要】シード指定時は、代表コメントを強制的に「選んだ軸コメント」に上書きする
                        if method_selected == 'seed' and seed_comment_data:
                            try:
                                # 現在の分析対象データ内におけるシードコメントのインデックスを探す
                                seed_index_in_docs = docs.index(seed_comment_data)
                                # クラスター0（シード時は1つだけなので0固定）の代表をシードコメントに強制設定
                                rep_coms[0] = seed_comment_data
                                rep_inds[0] = seed_index_in_docs
                            except ValueError:
                                # 救済ロジックを入れたので基本的にはここには来ないはずだが念のため
                                st.warning("注意: 軸コメントのマッチングに失敗しました。")

                        # 結果の保存
                        st.session_state.analysis_results = {
                            'df': df_res, 'rep_comments': rep_coms, 'rep_indices': rep_inds,
                            'embeddings': embs, 'unique_comments_df': uniq_df,
                            'total_count': len(df_proc), 'cols': (comment_col, rating_col, date_col),
                            'mode': 'cluster'
                        }
                        st.session_state.show_results = True
                        
                        # 完了メッセージ
                        if method_selected == 'seed':
                            st.success("完了 (軸コメント基準)")
                        elif method_selected == 'fps':
                            st.success("完了 (最遠点サンプリング)")
                        else:
                            st.success("完了")

        if st.session_state.show_results:
            res = st.session_state.analysis_results
            mode = res.get('mode', 'cluster')
            
            st.markdown("---")
            st.markdown("#### 共通設定")
            c1, c2, c3 = st.columns(3)
            top_n_cw = c1.slider("頻出単語数", 2, 5, 3) 
            top_n_net = c2.slider("共起Net:表示単語数(10-50位)", 10, 50, 30) 
            k_rank = c3.slider("共起Net:組合せ", 2, 5, 3)
            tok = get_tokenizer()

            if mode == 'adversative':
                adv_data = res.get('adv_res')
                if adv_data:
                    st.markdown("---"); st.header("▼ 逆説（ギャップ）分析結果")
                    pre_docs, post_docs, orig_docs, full_docs = adv_data['pre_docs'], adv_data['post_docs'], adv_data['original_docs'], adv_data['full_source_docs']
                    adv_conf = st.session_state.filter_config.get('adversative', {})
                    pk, pok = adv_conf.get('pre_keywords', []), adv_conf.get('post_keywords', [])
                    pre_cond = f"KW:{pk}" if pk else "条件なし(全て)"
                    post_cond = f"KW:{pok}" if pok else "条件なし(全て)"
                    st.info(f"抽出条件: 「{pre_cond}」...➡(逆接)➡...「{post_cond}」")
                    
                    tab_adv_viz, tab_adv_list = st.tabs(["可視化・比較", "抽出文一覧"])
                    with tab_adv_viz:
                        col_adv_a, col_adv_b = st.columns(2)
                        with col_adv_a:
                            st.subheader("① 前件 (〜だけど)")
                            fig_a, wr_a, _, wex_a = create_co_occurrence_network(pre_docs, tok, k_rank, top_n_net, "【前件】")
                            st.plotly_chart(fig_a, use_container_width=True)
                            buffer_a = io.StringIO()
                            fig_a.write_html(buffer_a, include_plotlyjs='cdn')
                            st.download_button("HTMLで解析結果を保存", buffer_a.getvalue().encode(), "adv_pre_net.html", "text/html")
                            st.write("**頻出単語 (前件)**")
                            st.dataframe(wr_a.head(10), hide_index=True)
                            
                            excel_data_a = convert_df_to_excel([wex_a], ['前件ネットワーク分析'])
                            st.download_button("📥 前件Net解析結果をExcelでDL", excel_data_a, "adv_pre_net.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                        with col_adv_b:
                            st.subheader("② 後件 (しかし〜)")
                            fig_b, wr_b, _, wex_b = create_co_occurrence_network(post_docs, tok, k_rank, top_n_net, "【後件】")
                            st.plotly_chart(fig_b, use_container_width=True)
                            buffer_b = io.StringIO()
                            fig_b.write_html(buffer_b, include_plotlyjs='cdn')
                            st.download_button("HTMLで解析結果を保存", buffer_b.getvalue().encode(), "adv_post_net.html", "text/html")
                            st.write("**頻出単語 (後件)**")
                            st.dataframe(wr_b.head(10), hide_index=True)
                            
                            excel_data_b = convert_df_to_excel([wex_b], ['後件ネットワーク分析'])
                            st.download_button("📥 後件Net解析結果をExcelでDL", excel_data_b, "adv_post_net.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                    with tab_adv_list:
                        adv_df_disp = pd.DataFrame({'前件': pre_docs, '後件': post_docs, '抽出された一文': orig_docs, '引用全文': full_docs})
                        st.dataframe(adv_df_disp)

            elif mode == 'cluster':
                st.markdown("---"); st.header("クラスター分析結果")
                
                # --- 代表コメント間類似度行列 ---
                st.subheader("クラスター代表コメント間の類似度")
                rep_indices = res['rep_indices']
                embeddings = res['embeddings']
                cluster_ids = sorted([i for i in res['df']['cluster'].unique() if i != -1])
                
                # インデックスが正しく取得できているか確認しつつベクトル抽出
                target_vectors = []
                valid_c_ids = []
                for cid in cluster_ids:
                    if cid in rep_indices:
                        idx = rep_indices[cid]
                        if idx < len(embeddings): # 念のため範囲チェック
                            target_vectors.append(embeddings[idx])
                            valid_c_ids.append(cid)
                
                if len(target_vectors) > 1:
                    sim_matrix = cosine_similarity(target_vectors)
                    cols_name = [f"クラスター{i+1}" for i in valid_c_ids]
                    df_sim = pd.DataFrame(sim_matrix, index=cols_name, columns=cols_name)
                    st.dataframe(df_sim.style.background_gradient(cmap="Blues", vmin=0, vmax=1).format("{:.2f}"))
                else:
                    st.caption("※クラスターが1つのみ、または有効なデータがないため類似度は算出されません。")
                
                st.markdown("---")

                df_view = res['df'].copy()
                c_col, r_col, d_col = res['cols']
                scores_list = [analyze_emotions(text) for text in df_view[c_col]]
                scores = pd.DataFrame(scores_list)
                df_view = pd.concat([df_view.reset_index(drop=True), scores], axis=1)

                f_conf = st.session_state.filter_config
                multi_period_settings = f_conf.get('multi_period') if f_conf.get('filter_type') == '特定期間ごと' else None
                cluster_kw_settings = f_conf.get('cluster_keywords', [])
                cluster_emo_settings = f_conf.get('cluster_emotions', [])

                grand_total = len(df_kanko)

                for i in cluster_ids:
                    cluster_label = f"クラスター {i+1}"
                    cdf = df_view[df_view['cluster'] == i].copy()
                    
                    if multi_period_settings and i < len(multi_period_settings):
                        if len(multi_period_settings[i]['range']) == 2:
                            current_c_start, current_c_end = multi_period_settings[i]['range']
                            cdf = cdf[cdf['parsed_date_obj'].dt.date.between(current_c_start, current_c_end)]
                            if cdf.empty: continue

                    cluster_initial_count = st.session_state.get('cluster_pre_counts', {}).get(i, len(cdf))

                    is_filtered = False
                    
                    # KW Filter
                    if i < len(cluster_kw_settings):
                        ks = cluster_kw_settings[i]
                        if ks['keywords']:
                            pat = '|'.join([re.escape(k) for k in ks['keywords']])
                            if "AND" in ks['logic']:
                                for k in ks['keywords']: cdf = cdf[cdf[c_col].str.contains(re.escape(k), na=False)]
                            else:
                                cdf = cdf[cdf[c_col].str.contains(pat, na=False)]
                            is_filtered = True

                    # Emotion Filter
                    if i < len(cluster_emo_settings) and f_conf.get('use_emotion'):
                        es = cluster_emo_settings[i]
                        valid_indices = cdf.index
                        for emotion, (min_val, max_val) in es.items():
                            if emotion in cdf.columns:
                                if min_val > 1.0 or max_val < 5.0:
                                    current_valid = cdf[cdf[emotion].between(min_val, max_val)].index
                                    valid_indices = valid_indices.intersection(current_valid)
                        cdf = cdf.loc[valid_indices]
                        is_filtered = True
                    
                    # 類似度フィルタ (表示前に適用)
                    ridx = res['rep_indices'].get(i)
                    if ridx is not None and not cdf.empty:
                        remb = res['embeddings'][ridx].reshape(1, -1)
                        udf = res['unique_comments_df']
                        t_inds = udf[udf['cluster'] == i].index
                        t_embs = res['embeddings'][t_inds]
                        sims = cosine_similarity(remb, t_embs).flatten()
                        sim_df = udf.iloc[t_inds].copy()
                        sim_df['sim'] = sims
                        cdf = pd.merge(cdf, sim_df[[c_col, 'sim']], on=c_col, how='left')
                        cdf.rename(columns={'sim': 'コサイン類似度'}, inplace=True)
                        cdf['コサイン類似度'] = cdf['コサイン類似度'].fillna(0)
                        
                        min_s_global = f_conf.get('min_similarity', 0.0)
                        if 'コサイン類似度' in cdf.columns: 
                            cdf = cdf[cdf['コサイン類似度'] >= min_s_global]
                        
                        cdf = cdf.sort_values('コサイン類似度', ascending=False)
                    
                    # 【修正】軸コメントが「絞り込み条件（期間など）に合致しない」場合のみ、表示リストから除外する
                    # （合致する場合は、そのまま表示・カウントに含める）
                    if use_seed and seed_comment_data:
                         # filtered_dfは「期間・KW等のフィルタ適用後」のデータ
                         # ここに含まれていない＝強制追加されたデータなので隠す
                         if seed_comment_data not in st.session_state.filtered_df[c_col].values:
                             cdf = cdf[cdf[c_col] != seed_comment_data]

                    cnt = len(cdf)
                    
                    # 【修正】軸コメントの強制追加により、母数(フィルタ合致数)を超えた場合の表示補正
                    if cnt > cluster_initial_count:
                        cluster_initial_count = cnt

                    pct_cluster = (cnt / cluster_initial_count * 100) if cluster_initial_count > 0 else 0
                    pct_total = (cnt / grand_total * 100) if grand_total > 0 else 0
                    
                    rep = res['rep_comments'].get(i, "なし")
                    suffix = " 🔍(フィルタ適用済)" if is_filtered else ""

                    with st.expander(f"**{cluster_label}**"):
                        st.info(f"**対象コメント数:** {cnt}件 {suffix}  \n"
                                f"対クラスター初期: {cnt}/{cluster_initial_count}件 ({pct_cluster:.1f}%)  \n"
                                f"対全体データ: {cnt}/{grand_total}件 ({pct_total:.1f}%)  \n"
                                f"**代表コメント:** 「{rep}」")
                        
                        st.metric(f"共通頻出単語 TOP{top_n_cw}", get_common_words(cdf[c_col].tolist(), tok, top_n_cw))
                        
                        st.subheader("口コミ一覧")
                        cols = [c for c in [c_col, 'コサイン類似度', r_col, d_col] + list(EMOTION_DICT.keys()) if c in cdf.columns]
                        st.dataframe(cdf[cols])

                        ft = st.session_state.filter_config.get('filter_type')
                        if ft in ['指定なし', '年月指定']:
                            st.markdown("###### 旅行時期別件数集計")
                            if not cdf.empty and 'parsed_date_obj' in cdf.columns:
                                valid_dates = cdf['parsed_date_obj'].dropna()
                                if not valid_dates.empty:
                                    monthly_counts = valid_dates.dt.to_period('M').value_counts().sort_index()
                                    df_mc = monthly_counts.reset_index()
                                    df_mc.columns = ['年月', '件数']
                                    df_mc['年月'] = df_mc['年月'].astype(str)
                                    
                                    fig_mc = px.bar(df_mc, x='年月', y='件数', title=f"【{cluster_label}】 旅行時期別件数")
                                    st.plotly_chart(fig_mc, use_container_width=True, key=f"mc_chart_{i}")
                                    
                                    buffer_mc = io.StringIO()
                                    fig_mc.write_html(buffer_mc, include_plotlyjs='cdn')
                                    st.download_button(
                                        label="HTMLで解析結果を保存",
                                        data=buffer_mc.getvalue().encode(),
                                        file_name=f"cluster_{i+1}_period_counts.html",
                                        mime="text/html",
                                        key=f"dl_mc_{i}"
                                    )

                        st.markdown("---"); st.subheader("共起ネットワーク")
                        
                        nd_mode, nd_m, nd_r, nr_sel, ns_min = None, [], [], [], min_s_global
                        if st.checkbox("▼ ネットワーク作成対象の絞り込み設定を表示", key=f"toggle_net_{i}"):
                            with st.container(border=True):
                                nc1, nc2, nc3 = st.columns(3)
                                nd_mode = nc1.radio("時期", ["絞り込み無", "年月指定", "期間指定"], index=None, key=f"nd_{i}")
                                if nd_mode == "年月指定":
                                    available_ym = sorted(cdf['parsed_date_obj'].dt.strftime('%Y年%m月').unique()) if not cdf.empty else []
                                    nd_m = nc1.multiselect("年月", available_ym, key=f"nm_{i}")
                                elif nd_mode == "期間指定": nd_r = nc1.date_input("期間", [], key=f"nr_{i}")
                                all_r = sorted(cdf[r_col].unique().astype(str)) if r_col in cdf.columns else []
                                nr_sel = nc2.multiselect("評価星", all_r, default=all_r, key=f"nrs_{i}")
                                ns_min = nc3.slider("コサイン類似度(最低値)", min_value=min_s_global, max_value=1.0, value=min_s_global, step=0.01, key=f"ns_{i}")
                        
                        out_of_range_net = False
                        if nd_mode == "期間指定" and len(nd_r) == 2:
                            c_dates = cdf['parsed_date_obj'].dropna()
                            if not c_dates.empty:
                                if nd_r[0] < c_dates.min().date() or nd_r[1] > c_dates.max().date(): out_of_range_net = True
                            else: out_of_range_net = True
                        
                        if out_of_range_net: st.error("分析対象の絞り込みの期間内で指定して下さい")
                        else:
                            dnet = cdf.copy()
                            if 'parsed_date_obj' not in dnet.columns: dnet['parsed_date_obj'] = pd.to_datetime(dnet[d_col], errors='coerce')
                            if nd_mode == "年月指定" and nd_m: dnet = dnet[dnet['parsed_date_obj'].dt.strftime('%Y年%m月').isin(nd_m)]
                            elif nd_mode == "期間指定" and len(nd_r) == 2: dnet = dnet[dnet['parsed_date_obj'].dt.date.between(nd_r[0], nd_r[1])]
                            if r_col in dnet.columns and nr_sel: dnet = dnet[dnet[r_col].astype(str).isin(nr_sel)]
                            if 'コサイン類似度' in dnet.columns: dnet = dnet[dnet['コサイン類似度'] >= ns_min]
                            
                            if not dnet.empty:
                                fig_n, wr_n, wc_n, wex_n = create_co_occurrence_network(dnet[c_col].tolist(), tok, k_rank, top_n_net)
                                st.plotly_chart(fig_n, use_container_width=True, key=f"p_net_{i}")
                                buffer_n = io.StringIO()
                                fig_n.write_html(buffer_n, include_plotlyjs='cdn')
                                st.download_button(label="HTMLで解析結果を保存", data=buffer_n.getvalue().encode(), file_name=f"cluster_{i+1}_network_analysis.html", mime="text/html", key=f"dl_net_{i}")
                            else:
                                if nd_mode: st.warning("該当データなし")

                        st.markdown("---"); st.subheader("気象推移")
                        if df_kisho is not None:
                            valid_cols = [c for c in df_kisho.columns[2:] if pd.to_numeric(df_kisho[c], errors='coerce').notna().any()]
                            
                            wd_mode, wd_m, wd_r, disp_m = "絞り込み無", [], [], valid_cols[0] if valid_cols else None
                            w_logic_local, w_conds_local = "AND", []
                            k_date_col_name = df_kisho.columns[0]

                            if st.checkbox("▼ グラフ表示データの絞り込み設定を表示", key=f"toggle_weather_{i}"):
                                with st.container(border=True):
                                    wd_mode = st.radio("時期", ["絞り込み無", "年月指定", "期間指定"], index=0, horizontal=True, key=f"wd_{i}")
                                    if wd_mode == "年月指定":
                                        temp_dates = pd.to_datetime(df_kisho[k_date_col_name], errors='coerce').dropna()
                                        if not temp_dates.empty:
                                            temp_df = pd.DataFrame({'date': temp_dates})
                                            temp_df['ym'] = temp_df['date'].dt.year.astype(str) + '/' + temp_df['date'].dt.month.astype(str)
                                            available_k_ym = temp_df.sort_values('date')['ym'].unique().tolist()
                                        else:
                                            available_k_ym = []
                                        wd_m = st.multiselect("年月", available_k_ym, key=f"wm_{i}_g")
                                    elif wd_mode == "期間指定": 
                                        wd_r = st.date_input("期間", [], key=f"wr_{i}_g")
                                    
                                    st.markdown("###### 気象条件設定")
                                    w_logic_local_label = st.radio("条件", ["すべて", "いずれか"], horizontal=True, key=f"wl_{i}_loc")
                                    w_logic_local = 'AND' if w_logic_local_label == "すべて" else 'OR'
                                    
                                    for j in range(3):
                                        c1, c2, c3 = st.columns([2, 1, 1])
                                        m = c1.selectbox(f"指標 {j+1}", ["未選択"]+valid_cols, key=f"wm_{i}_{j}_loc")
                                        o = c2.selectbox(f"条件 {j+1}", ["未選択", '>', '>=', '<', '<=', '=='], key=f"wo_{i}_{j}_loc")
                                        v = c3.number_input(f"値 {j+1}", value=None, step=1.0, key=f"wv_{i}_{j}_loc")
                                        if m!="未選択" and o!="未選択" and v is not None: w_conds_local.append({'metric': m, 'op': o, 'val': v})

                                    if valid_cols:
                                        st.write("##### 表示指標")
                                        disp_m = st.radio("グラフにするデータを選択", valid_cols, horizontal=True, key=f"wdisp_{i}")
                            
                            c_dates = cdf['parsed_date_obj'].dropna()
                            dkp = df_kisho.copy()
                            dkp[k_date_col_name] = pd.to_datetime(dkp[k_date_col_name], errors='coerce')
                            dkp.dropna(subset=[k_date_col_name], inplace=True)
                            
                            if wd_mode == "年月指定" and wd_m:
                                ym_series = dkp[k_date_col_name].dt.year.astype(str) + '/' + dkp[k_date_col_name].dt.month.astype(str)
                                dkp = dkp[ym_series.isin(wd_m)]
                            elif wd_mode == "期間指定" and len(wd_r) == 2:
                                dkp = dkp[dkp[k_date_col_name].dt.date.between(wd_r[0], wd_r[1])]
                            else: 
                                if not c_dates.empty:
                                    c_min, c_max = c_dates.min(), c_dates.max()
                                    dkp = dkp[(dkp[k_date_col_name] >= c_min) & (dkp[k_date_col_name] <= c_max)]
                                else:
                                    dkp = pd.DataFrame(columns=dkp.columns) 

                            if w_conds_local:
                                vd_sets = []
                                for cond in w_conds_local:
                                    metric, op, val = cond['metric'], cond['op'], cond['val']
                                    if metric in dkp.columns:
                                        dkp[metric] = pd.to_numeric(dkp[metric], errors='coerce')
                                        if op == '>': dates = dkp[dkp[metric] > val][k_date_col_name]
                                        elif op == '>=': dates = dkp[dkp[metric] >= val][k_date_col_name]
                                        elif op == '<': dates = dkp[dkp[metric] < val][k_date_col_name]
                                        elif op == '<=': dates = dkp[dkp[metric] <= val][k_date_col_name]
                                        else: dates = dkp[dkp[metric] == val][k_date_col_name]
                                        vd_sets.append(set(dates))
                                if vd_sets:
                                    final_d = set.intersection(*vd_sets) if w_logic_local == 'AND' else set.union(*vd_sets)
                                    dkp = dkp[dkp[k_date_col_name].isin(final_d)]

                            if not dkp.empty and disp_m:
                                fig_w = px.scatter(dkp, x=k_date_col_name, y=disp_m, title=f"{disp_m}")
                                st.plotly_chart(fig_w, use_container_width=True, key=f"p_weather_{i}")
                                
                                buffer_w = io.StringIO()
                                fig_w.write_html(buffer_w, include_plotlyjs='cdn')
                                st.download_button(label="HTMLで解析結果を保存", data=buffer_w.getvalue().encode(), file_name=f"cluster_{i+1}_weather_data.html", mime="text/html", key=f"dl_weather_html_{i}")
                                
                                dkp_excel = dkp.copy()
                                if k_date_col_name in dkp_excel.columns:
                                     dkp_excel[k_date_col_name] = dkp_excel[k_date_col_name].dt.date
                                
                                excel_data_w = convert_df_to_excel([dkp_excel], ['気象推移データ'])
                                st.download_button(label="📥 気象データをExcelでDL", data=excel_data_w, file_name=f"cluster_{i+1}_weather_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_weather_excel_{i}")

                            else:
                                if wd_mode != "絞り込み無" or w_conds_local: st.warning("該当データなし")
                                else: st.warning("旅行時期データがないため気象データを表示できません")

                        if df_trend is not None:
                            st.markdown("---"); st.subheader("関連トレンド")
                            cd = cdf['parsed_date_obj'].dropna()
                            if not cd.empty:
                                c_min, c_max = cd.min(), cd.max()
                                
                                tr_mode, tr_m, tr_r = "絞り込み無", [], []
                                tr_top_range = (1, 100)
                                tr_rise_range = (1, 100)

                                if st.checkbox("▼ トレンド表示データの絞り込み設定を表示", key=f"toggle_trend_{i}"):
                                    with st.container(border=True):
                                        tr_mode = st.radio("時期", ["絞り込み無", "年月指定", "期間指定"], index=0, horizontal=True, key=f"tr_mode_{i}")
                                        if tr_mode == "年月指定":
                                            available_tr_ym = sorted(df_trend['date'].dt.strftime('%Y年%m月').unique())
                                            tr_m = st.multiselect("年月", available_tr_ym, key=f"tr_m_{i}")
                                        elif tr_mode == "期間指定":
                                            tr_r = st.date_input("期間", [], key=f"tr_r_{i}")

                                        st.markdown("###### 表示順位フィルター")
                                        tr_c1, tr_c2 = st.columns(2)
                                        tr_top_range = tr_c1.slider("TOP順位", 1, 100, (1, 50), key=f"tr_top_r_{i}")
                                        tr_rise_range = tr_c2.slider("RISING順位", 1, 100, (1, 50), key=f"tr_rise_r_{i}")

                                if tr_mode == "年月指定" and tr_m:
                                    rt = df_trend[df_trend['date'].dt.strftime('%Y年%m月').isin(tr_m)]
                                elif tr_mode == "期間指定" and len(tr_r) == 2:
                                    s, e = pd.to_datetime(tr_r[0]), pd.to_datetime(tr_r[1])
                                    rt = df_trend[(df_trend['date'] <= e) & (df_trend['end_date'] >= s)]
                                else:
                                    rt = df_trend[(df_trend['date'] <= c_max) & (df_trend['end_date'] >= c_min)]
                                
                                if not rt.empty:
                                    unique_weeks = rt.drop_duplicates(subset=['date']).sort_values('date')
                                    t_titles = [r['期間'] for _, r in unique_weeks.iterrows()]
                                    if t_titles:
                                        sub_tabs = st.tabs(t_titles[:20])
                                        for t_idx, t_tab in enumerate(sub_tabs):
                                            with t_tab:
                                                w_date = unique_weeks['date'].iloc[t_idx]
                                                w_df = rt[rt['date'] == w_date]
                                                col_tr1, col_tr2 = st.columns(2)
                                                with col_tr1:
                                                    st.write("##### TOP")
                                                    d_t = w_df[w_df['type']=='TOP'].copy()
                                                    d_t['rn'] = pd.to_numeric(d_t['Rank'].astype(str).str.extract(r'(\d+)')[0], errors='coerce').fillna(0).astype(int)
                                                    d_t = d_t[(d_t['rn'] >= tr_top_range[0]) & (d_t['rn'] <= tr_top_range[1])]
                                                    st.dataframe(d_t[['Rank', '用語', '検索インタレスト']], hide_index=True, use_container_width=True)
                                                with col_tr2:
                                                    st.write("##### RISING")
                                                    d_r = w_df[w_df['type']=='RISING'].copy()
                                                    d_r['rn'] = pd.to_numeric(d_r['Rank'].astype(str).str.extract(r'(\d+)')[0], errors='coerce').fillna(0).astype(int)
                                                    d_r = d_r[(d_r['rn'] >= tr_rise_range[0]) & (d_r['rn'] <= tr_rise_range[1])]
                                                    st.dataframe(d_r[['Rank', '用語', '検索インタレスト']], hide_index=True, use_container_width=True)
                                    
                                    st.markdown("---")
                                    dl_col1, dl_col2 = st.columns(2)
                                    with dl_col1:
                                        st.write("###### TOP (期間内全件)")
                                        df_dl_top = rt[rt['type'] == 'TOP'].copy()
                                        if not df_dl_top.empty:
                                            # Filter by Rank (A column)
                                            df_dl_top['rn'] = pd.to_numeric(df_dl_top['Rank'].astype(str).str.extract(r'(\d+)')[0], errors='coerce').fillna(0).astype(int)
                                            df_dl_top = df_dl_top[(df_dl_top['rn'] >= tr_top_range[0]) & (df_dl_top['rn'] <= tr_top_range[1])]
                                            df_dl_top = df_dl_top.drop(columns=['rn'])
                                            
                                            # Format Date columns (E and F)
                                            if 'date' in df_dl_top.columns: df_dl_top['date'] = df_dl_top['date'].dt.date
                                            if 'end_date' in df_dl_top.columns: df_dl_top['end_date'] = df_dl_top['end_date'].dt.date
                                            
                                            excel_top = convert_df_to_excel([df_dl_top], ['TOP_Trends'])
                                            st.download_button(label="📥 TOP全ランキングをExcelで保存", data=excel_top, file_name="trend_top_full_period.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_trend_top_all_{i}")
                                    with dl_col2:
                                        st.write("###### RISING (期間内全件)")
                                        df_dl_rising = rt[rt['type'] == 'RISING'].copy()
                                        if not df_dl_rising.empty:
                                            # Filter by Rank (A column)
                                            df_dl_rising['rn'] = pd.to_numeric(df_dl_rising['Rank'].astype(str).str.extract(r'(\d+)')[0], errors='coerce').fillna(0).astype(int)
                                            df_dl_rising = df_dl_rising[(df_dl_rising['rn'] >= tr_rise_range[0]) & (df_dl_rising['rn'] <= tr_rise_range[1])]
                                            df_dl_rising = df_dl_rising.drop(columns=['rn'])

                                            # Format Date columns (E and F)
                                            if 'date' in df_dl_rising.columns: df_dl_rising['date'] = df_dl_rising['date'].dt.date
                                            if 'end_date' in df_dl_rising.columns: df_dl_rising['end_date'] = df_dl_rising['end_date'].dt.date

                                            excel_rising = convert_df_to_excel([df_dl_rising], ['RISING_Trends'])
                                            st.download_button(label="📥 RISING全ランキングをExcelで保存", data=excel_rising, file_name="trend_rising_full_period.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_trend_rising_all_{i}")

                                else: st.info("該当するトレンドデータはありません")
                            else: st.info("日付データなし")
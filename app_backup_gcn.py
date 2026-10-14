
import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import GCNConv
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

BERT_PATH = "./models/bert-base-uncased"
GNN_PATH = "./models/gnn_model.pt"
GRAPH_PATH = "./data/processed/news_graph.pt"
DATA_PATH = "./data/processed/clean_news.csv"

st.set_page_config(
    page_title="Verity",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# PREMIUM UI
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

* {
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 90% 5%, rgba(99,102,241,0.07), transparent 25%),
        radial-gradient(circle at 5% 30%, rgba(139,92,246,0.05), transparent 25%),
        #f8f8fb;
    color: #17171c;
}

header, footer, #MainMenu {
    visibility: hidden;
}

.block-container {
    max-width: 1120px;
    padding-top: 28px;
    padding-bottom: 70px;
}

/* ---------------------------------------------------------
   TOP BAR
--------------------------------------------------------- */

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 42px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-mark {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: #17171c;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    font-weight: 700;
}

.brand-name {
    font-size: 19px;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.brand-name span {
    color: #6d4aff;
}

.status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 13px;
    border-radius: 100px;
    background: white;
    border: 1px solid #e7e5ee;
    font-size: 11px;
    color: #6f6d78;
    font-weight: 600;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 0 4px rgba(34,197,94,0.10);
}

/* ---------------------------------------------------------
   HERO
--------------------------------------------------------- */

.hero-wrap {
    text-align: center;
    margin-bottom: 38px;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    color: #6d4aff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

.eyebrow-line {
    width: 22px;
    height: 1px;
    background: #6d4aff;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 54px;
    line-height: 1.08;
    letter-spacing: -1.8px;
    margin: 0;
    color: #17171c;
}

.hero-title span {
    color: #6d4aff;
}

.hero-description {
    max-width: 620px;
    margin: 17px auto 0 auto;
    color: #777580;
    font-size: 15px;
    line-height: 1.7;
}

/* ---------------------------------------------------------
   ANALYZER
--------------------------------------------------------- */

.analyzer {
    background: white;
    border: 1px solid #e5e3eb;
    border-radius: 24px;
    padding: 9px;
    box-shadow:
        0 20px 60px rgba(30, 20, 70, 0.07),
        0 2px 8px rgba(0,0,0,0.02);
}

.analyzer-inner {
    border: 1px solid #eeeef3;
    border-radius: 17px;
    overflow: hidden;
    background: #fcfcfd;
}

.input-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 19px;
    border-bottom: 1px solid #eeeeF2;
}

.input-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #777580;
}

.input-hint {
    font-size: 10px;
    color: #a09da8;
}

.stTextArea {
    margin-bottom: 0 !important;
}

textarea {
    background: #fcfcfd !important;
    color: #202027 !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    padding: 18px 20px !important;
}

textarea:focus {
    border: none !important;
    box-shadow: none !important;
}

/* ---------------------------------------------------------
   BUTTONS
--------------------------------------------------------- */

.action-row {
    padding: 8px 10px 10px 10px;
}

.stButton > button {
    min-height: 45px !important;
    border-radius: 11px !important;
    border: 1px solid #e2e0e9 !important;
    background: white !important;
    color: #34323a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    transition: all .18s ease !important;
}

.stButton > button:hover {
    border-color: #6d4aff !important;
    color: #6d4aff !important;
    transform: translateY(-1px);
}

div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button {
    background: #17171c !important;
    color: white !important;
    border-color: #17171c !important;
    box-shadow: 0 8px 20px rgba(23,23,28,0.18);
}

div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button:hover {
    background: #6d4aff !important;
    border-color: #6d4aff !important;
    color: white !important;
}

/* ---------------------------------------------------------
   RESULT
--------------------------------------------------------- */

.result-section {
    margin-top: 42px;
}

.result-card {
    background: #17171c;
    color: white;
    border-radius: 24px;
    padding: 34px;
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: "";
    position: absolute;
    width: 330px;
    height: 330px;
    right: -100px;
    top: -160px;
    border-radius: 50%;
    background: rgba(109,74,255,0.18);
}

.result-card::after {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    left: -150px;
    bottom: -150px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.06);
}

.result-label {
    color: #aaa7b3;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-size: 10px;
    font-weight: 700;
}

.result-main {
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    z-index: 2;
}

.result-word {
    font-family: 'Playfair Display', serif;
    font-size: 62px;
    line-height: 1;
    margin-top: 9px;
}

.result-word.real {
    color: #75e6a1;
}

.result-word.fake {
    color: #ff8585;
}

.result-text {
    color: #aaa7b3;
    font-size: 13px;
    line-height: 1.6;
    max-width: 510px;
    margin-top: 11px;
}

.confidence {
    width: 118px;
    height: 118px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: conic-gradient(
        #8b6cff var(--confidence),
        #302f36 0
    );
    flex-shrink: 0;
}

.confidence-inner {
    width: 94px;
    height: 94px;
    border-radius: 50%;
    background: #17171c;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.confidence-number {
    font-size: 22px;
    font-weight: 700;
}

.confidence-label {
    font-size: 8px;
    letter-spacing: 1px;
    color: #9996a1;
    margin-top: 2px;
}

/* ---------------------------------------------------------
   PROBABILITY
--------------------------------------------------------- */

.probability {
    position: relative;
    z-index: 2;
    margin-top: 31px;
    padding-top: 24px;
    border-top: 1px solid rgba(255,255,255,0.08);
}

.prob-head {
    display: flex;
    justify-content: space-between;
    color: #aaa7b3;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .8px;
    margin-bottom: 8px;
}

.prob-bar {
    height: 7px;
    background: #302f36;
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 16px;
}

.prob-fill-real {
    height: 100%;
    background: #75e6a1;
    border-radius: 20px;
}

.prob-fill-fake {
    height: 100%;
    background: #ff8585;
    border-radius: 20px;
}

/* ---------------------------------------------------------
   SECTION
--------------------------------------------------------- */

.section {
    margin-top: 58px;
}

.section-header {
    margin-bottom: 22px;
}

.section-kicker {
    color: #6d4aff;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 27px;
    letter-spacing: -.5px;
    color: #17171c;
}

.section-description {
    color: #85818c;
    font-size: 13px;
    margin-top: 5px;
}

/* ---------------------------------------------------------
   PIPELINE
--------------------------------------------------------- */

.pipeline-card {
    background: white;
    border: 1px solid #e6e4ec;
    border-radius: 17px;
    padding: 23px;
    height: 100%;
    transition: transform .18s ease, box-shadow .18s ease;
}

.pipeline-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(30,20,70,0.07);
}

.pipeline-number {
    font-size: 10px;
    font-weight: 800;
    color: #6d4aff;
    letter-spacing: 1px;
}

.pipeline-icon {
    margin: 18px 0 13px 0;
    width: 39px;
    height: 39px;
    border-radius: 11px;
    background: #f1efff;
    color: #6d4aff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
}

.pipeline-title {
    font-size: 14px;
    font-weight: 700;
    color: #25242a;
}

.pipeline-text {
    margin-top: 7px;
    color: #85818c;
    font-size: 12px;
    line-height: 1.65;
}

/* ---------------------------------------------------------
   RELATED ARTICLES
--------------------------------------------------------- */

.article {
    background: white;
    border: 1px solid #e6e4ec;
    border-radius: 14px;
    padding: 17px 19px;
    margin-bottom: 9px;
    display: flex;
    align-items: center;
    gap: 15px;
}

.article-rank {
    width: 29px;
    height: 29px;
    border-radius: 9px;
    background: #f2f0f8;
    color: #6d4aff;
    font-size: 10px;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.article-content {
    flex: 1;
    color: #44414b;
    font-size: 12.5px;
    line-height: 1.5;
}

.article-score {
    font-size: 11px;
    font-weight: 700;
    color: #6d4aff;
    white-space: nowrap;
}

/* ---------------------------------------------------------
   ARCHITECTURE
--------------------------------------------------------- */

.architecture {
    background: #17171c;
    border-radius: 20px;
    padding: 30px;
    color: white;
    position: relative;
    overflow: hidden;
}

.architecture::before {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -80px;
    top: -100px;
    border-radius: 50%;
    background: rgba(109,74,255,.13);
}

.arch-title {
    font-family: 'Playfair Display', serif;
    font-size: 25px;
    position: relative;
    z-index: 1;
}

.arch-description {
    color: #aaa7b3;
    font-size: 12.5px;
    line-height: 1.7;
    max-width: 690px;
    margin-top: 9px;
    position: relative;
    z-index: 1;
}

.arch-flow {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-top: 25px;
    position: relative;
    z-index: 1;
    flex-wrap: wrap;
}

.arch-node {
    padding: 10px 15px;
    border: 1px solid rgba(255,255,255,.12);
    background: rgba(255,255,255,.05);
    border-radius: 9px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .5px;
}

.arch-arrow {
    color: #6d4aff;
    font-weight: 700;
}

/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.footer {
    margin-top: 70px;
    padding-top: 25px;
    border-top: 1px solid #e4e2ea;
    text-align: center;
    color: #9995a0;
    font-size: 10.5px;
    letter-spacing: .2px;
}

.footer strong {
    color: #5d5965;
}

/* ---------------------------------------------------------
   STREAMLIT CLEANUP
--------------------------------------------------------- */

div[data-testid="stVerticalBlock"] > div:has(.result-section) {
    gap: 0 !important;
}

.stProgress > div > div > div > div {
    border-radius: 20px !important;
}

.stProgress {
    display: none;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(768, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.conv3(x, edge_index)


@st.cache_resource
def load_models():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)

    bert = AutoModel.from_pretrained(
        BERT_PATH
    ).to(device).eval()

    gnn = GCN()

    gnn.load_state_dict(
        torch.load(
            GNN_PATH,
            map_location=device
        )
    )

    gnn.to(device).eval()

    graph = torch.load(
        GRAPH_PATH,
        weights_only=False
    )

    df = pd.read_csv(DATA_PATH)

    return tokenizer, bert, gnn, graph, df, device


# ============================================================
# TOP BAR
# ============================================================

st.markdown("""
<div class="topbar">

    <div class="brand">
        <div class="brand-mark">◆</div>
        <div class="brand-name">Verity<span>.</span></div>
    </div>

    <div class="status">
        <div class="status-dot"></div>
        BERT + GCN ENGINE ONLINE
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero-wrap">

    <div class="eyebrow">
        <div class="eyebrow-line"></div>
        AI-powered news verification
        <div class="eyebrow-line"></div>
    </div>

    <div class="hero-title">
        Before you believe it,<br>
        <span>verify it.</span>
    </div>

    <div class="hero-description">
        Verity analyzes the language and semantic relationships within
        a news article using BERT and a Graph Convolutional Network.
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT
# ============================================================

st.markdown('<div class="analyzer">', unsafe_allow_html=True)

st.markdown("""
<div class="analyzer-inner">

<div class="input-header">
    <div class="input-label">Article to analyze</div>
    <div class="input-hint">HEADLINE OR FULL ARTICLE</div>
</div>

""", unsafe_allow_html=True)

news = st.text_area(
    "News",
    height=190,
    placeholder="Paste a news headline or article here...",
    label_visibility="collapsed",
    key="news_input"
)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="action-row">', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1.2, 1])

with c1:
    sample_tech = st.button(
        "Try a sample",
        use_container_width=True,
        key="sample_tech"
    )

with c2:
    analyze = st.button(
        "◆  Analyze article",
        use_container_width=True,
        key="analyze"
    )

with c3:
    clear = st.button(
        "Clear",
        use_container_width=True,
        key="clear"
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


if sample_tech:

    st.session_state.sample_text = (
        "Researchers have announced a new technology "
        "that could significantly improve the efficiency "
        "of renewable energy systems."
    )

    st.rerun()


if clear:

    st.session_state.sample_text = ""

    st.rerun()


if "sample_text" in st.session_state and st.session_state.sample_text:

    news = st.session_state.sample_text

    st.markdown(
        """
        <div style="
            margin-top:10px;
            color:#85818c;
            font-size:11px;
            text-align:center;
        ">
        Sample article loaded
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    if not news.strip():

        st.warning("Please paste a news article first.")

    else:

        with st.spinner("Analyzing linguistic and graph signals..."):

            tokenizer, bert, gnn, graph, df, device = load_models()

            existing_embeddings = graph["x"]
            existing_edges = graph["edge_index"]

            inputs = tokenizer(
                news,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            with torch.no_grad():

                bert_output = bert(**inputs)

                new_embedding = (
                    bert_output
                    .last_hidden_state[:, 0, :]
                )

            new_embedding_cpu = new_embedding.cpu()

            similarities = cosine_similarity(
                new_embedding_cpu.numpy(),
                existing_embeddings.numpy()
            )[0]

            top_indices = (
                similarities
                .argsort()[-5:][::-1]
            )

            new_node_index = existing_embeddings.shape[0]

            all_features = torch.cat(
                [
                    existing_embeddings,
                    new_embedding_cpu
                ],
                dim=0
            )

            new_edges = []

            for i in top_indices:

                new_edges.append(
                    [new_node_index, int(i)]
                )

                new_edges.append(
                    [int(i), new_node_index]
                )

            new_edges = torch.tensor(
                new_edges,
                dtype=torch.long
            ).t()

            all_edges = torch.cat(
                [
                    existing_edges,
                    new_edges
                ],
                dim=1
            )

            all_features = all_features.to(device)
            all_edges = all_edges.to(device)

            with torch.no_grad():

                output = gnn(
                    all_features,
                    all_edges
                )

                probabilities = F.softmax(
                    output[new_node_index],
                    dim=0
                )

                prediction = (
                    probabilities
                    .argmax()
                    .item()
                )

                confidence = (
                    probabilities[prediction]
                    .item() * 100
                )

            real_prob = probabilities[0].item() * 100
            fake_prob = probabilities[1].item() * 100


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:

            result = "FAKE"
            result_class = "fake"

            description = (
                "The model flagged this article as potentially "
                "misleading based on its learned content and "
                "graph-based signals."
            )

        else:

            result = "REAL"
            result_class = "real"

            description = (
                "The model classified this article as likely "
                "authentic based on its learned content and "
                "graph-based signals."
            )


        st.markdown(
            '<div class="result-section">',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-main">

                    <div>

                        <div class="result-label">
                            Analysis complete
                        </div>

                        <div class="result-word {result_class}">
                            {result}
                        </div>

                        <div class="result-text">
                            {description}
                        </div>

                    </div>

                    <div
                        class="confidence"
                        style="--confidence:{confidence:.1f}%"
                    >
                        <div class="confidence-inner">

                            <div class="confidence-number">
                                {confidence:.0f}%
                            </div>

                            <div class="confidence-label">
                                CONFIDENCE
                            </div>

                        </div>
                    </div>

                </div>

                <div class="probability">

                    <div class="prob-head">
                        <span>REAL</span>
                        <span>{real_prob:.1f}%</span>
                    </div>

                    <div class="prob-bar">
                        <div
                            class="prob-fill-real"
                            style="width:{real_prob:.1f}%"
                        ></div>
                    </div>

                    <div class="prob-head">
                        <span>FAKE</span>
                        <span>{fake_prob:.1f}%</span>
                    </div>

                    <div class="prob-bar">
                        <div
                            class="prob-fill-fake"
                            style="width:{fake_prob:.1f}%"
                        ></div>
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)


        # ----------------------------------------------------
        # PIPELINE
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section">

                <div class="section-header">

                    <div class="section-kicker">
                        Under the hood
                    </div>

                    <div class="section-title">
                        How Verity analyzed it
                    </div>

                    <div class="section-description">
                        Three computational stages contribute to the final classification.
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        p1, p2, p3 = st.columns(3)

        with p1:

            st.markdown(
                """
                <div class="pipeline-card">

                    <div class="pipeline-number">
                        01 / LANGUAGE
                    </div>

                    <div class="pipeline-icon">
                        B
                    </div>

                    <div class="pipeline-title">
                        BERT representation
                    </div>

                    <div class="pipeline-text">
                        The article is transformed into a
                        768-dimensional contextual representation
                        capturing its linguistic information.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with p2:

            st.markdown(
                """
                <div class="pipeline-card">

                    <div class="pipeline-number">
                        02 / GRAPH
                    </div>

                    <div class="pipeline-icon">
                        G
                    </div>

                    <div class="pipeline-title">
                        Semantic relationships
                    </div>

                    <div class="pipeline-text">
                        The new article is connected to its five
                        most semantically similar articles using
                        cosine similarity.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with p3:

            st.markdown(
                """
                <div class="pipeline-card">

                    <div class="pipeline-number">
                        03 / CLASSIFICATION
                    </div>

                    <div class="pipeline-icon">
                        C
                    </div>

                    <div class="pipeline-title">
                        Graph convolution
                    </div>

                    <div class="pipeline-text">
                        The GCN propagates information through
                        neighboring nodes before producing the
                        final real/fake classification.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # RELATED ARTICLES
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section">

                <div class="section-header">

                    <div class="section-kicker">
                        Semantic neighborhood
                    </div>

                    <div class="section-title">
                        Connected articles
                    </div>

                    <div class="section-description">
                        Articles with the highest semantic similarity to your input.
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        for rank, idx in enumerate(top_indices, 1):

            content = str(
                df.iloc[idx]["content"]
            )

            similarity = similarities[idx] * 100

            snippet = content[:220]

            if len(content) > 220:
                snippet += "..."

            st.markdown(
                f"""
                <div class="article">

                    <div class="article-rank">
                        {rank:02d}
                    </div>

                    <div class="article-content">
                        {snippet}
                    </div>

                    <div class="article-score">
                        {similarity:.1f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# ARCHITECTURE
# ============================================================

st.markdown(
    """
    <div class="section">

        <div class="section-header">

            <div class="section-kicker">
                System architecture
            </div>

            <div class="section-title">
                From language to graph intelligence
            </div>

        </div>

        <div class="architecture">

            <div class="arch-title">
                Verity detection pipeline
            </div>

            <div class="arch-description">
                Each article becomes a node in a semantic news graph.
                BERT provides contextual representations, cosine similarity
                establishes relationships between articles, and the GCN
                uses those relationships for classification.
            </div>

            <div class="arch-flow">

                <div class="arch-node">
                    NEWS ARTICLE
                </div>

                <div class="arch-arrow">
                    →
                </div>

                <div class="arch-node">
                    BERT
                </div>

                <div class="arch-arrow">
                    →
                </div>

                <div class="arch-node">
                    768-D EMBEDDING
                </div>

                <div class="arch-arrow">
                    →
                </div>

                <div class="arch-node">
                    SEMANTIC GRAPH
                </div>

                <div class="arch-arrow">
                    →
                </div>

                <div class="arch-node">
                    GCN
                </div>

                <div class="arch-arrow">
                    →
                </div>

                <div class="arch-node">
                    REAL / FAKE
                </div>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <strong>VERITY</strong> · Fake News Detection Prototype

        <br><br>

        BERT · Semantic Similarity · Graph Convolutional Network

    </div>
    """,
    unsafe_allow_html=True
)

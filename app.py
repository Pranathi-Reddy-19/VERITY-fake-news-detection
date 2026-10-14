# import re
# import html
# import streamlit as st
# import plotly.graph_objects as go
# from src.verity_pipeline import classify_article, generate_explanation

# st.set_page_config(page_title="VERITY | Factual Verification", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")


# def md(s):
#     # strip indentation + blank lines so markdown never turns html into code blocks
#     st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


# def section(t, s):
#     md(f'<div class="section-title">{t}</div><div class="section-subtitle">{s}</div>')


# md("""
# <style>
# .stApp{background:radial-gradient(circle at 10% 0%,rgba(124,58,237,.18),transparent 28%),radial-gradient(circle at 90% 10%,rgba(6,182,212,.13),transparent 25%),radial-gradient(circle at 50% 100%,rgba(236,72,153,.08),transparent 30%),#070b16;color:#f8fafc}
# .block-container{max-width:1250px;padding-top:1.5rem;padding-bottom:3rem}
# #MainMenu,footer,header{visibility:hidden}
# .header-box{padding:18px 24px;border-radius:18px;background:rgba(15,23,42,.68);border:1px solid rgba(148,163,184,.14);backdrop-filter:blur(14px)}
# .brand-title{font-size:28px;font-weight:850;letter-spacing:4px;background:linear-gradient(90deg,#a78bfa,#67e8f9,#f472b6);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
# .brand-subtitle{margin-top:3px;color:#64748b;font-size:11px;letter-spacing:1.2px;text-transform:uppercase}
# .status-wrapper{text-align:right;padding-top:13px}
# .status{display:inline-flex;align-items:center;gap:8px;padding:8px 13px;border-radius:999px;background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.2);color:#86efac;font-size:11px;font-weight:700;letter-spacing:.6px}
# .status-dot{width:7px;height:7px;border-radius:50%;background:#22c55e;box-shadow:0 0 6px #22c55e,0 0 14px rgba(34,197,94,.7)}
# .hero{text-align:center;padding:58px 20px 38px}
# .badge{display:inline-block;padding:7px 14px;border-radius:999px;background:rgba(124,58,237,.12);border:1px solid rgba(167,139,250,.24);color:#c4b5fd;font-size:10px;font-weight:800;letter-spacing:1.5px}
# .hero-title{margin:18px 0 12px;font-size:clamp(42px,6vw,70px);line-height:1.04;font-weight:850;background:linear-gradient(90deg,#fff,#c4b5fd,#67e8f9);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
# .hero-text{max-width:700px;margin:auto;color:#94a3b8;font-size:15px;line-height:1.7}
# .input-card{padding:25px;border-radius:20px;background:rgba(15,23,42,.72);border:1px solid rgba(148,163,184,.14);box-shadow:0 20px 60px rgba(0,0,0,.18);margin-bottom:12px}
# .input-title{color:#f8fafc;font-size:20px;font-weight:800}
# .input-subtitle{margin-top:4px;color:#64748b;font-size:13px}
# textarea{background:rgba(2,6,23,.78)!important;color:#f8fafc!important;border:1px solid rgba(139,92,246,.28)!important;border-radius:13px!important;font-size:14px!important}
# textarea:focus{border-color:#8b5cf6!important;box-shadow:0 0 0 1px #8b5cf6!important}
# .stButton>button{height:50px;border:none!important;border-radius:13px!important;background:linear-gradient(90deg,#7c3aed,#6366f1,#06b6d4)!important;color:#fff!important;font-size:14px!important;font-weight:800!important;letter-spacing:.5px;box-shadow:0 10px 30px rgba(99,102,241,.24);transition:.2s ease}
# .stButton>button:hover{transform:translateY(-2px);box-shadow:0 15px 40px rgba(6,182,212,.28)}
# .section-title{margin-top:40px;margin-bottom:4px;color:#f8fafc;font-size:25px;font-weight:800}
# .section-subtitle{margin-bottom:18px;color:#64748b;font-size:13px}
# .result-card{min-height:145px;padding:22px;border-radius:18px;background:linear-gradient(145deg,rgba(30,41,59,.8),rgba(15,23,42,.68));border:1px solid rgba(148,163,184,.13);box-shadow:0 15px 40px rgba(0,0,0,.16)}
# .card-label{color:#64748b;font-size:10px;font-weight:800;letter-spacing:1.3px;text-transform:uppercase}
# .card-value{margin-top:12px;color:#f8fafc;font-size:30px;font-weight:850}
# .card-description{margin-top:5px;color:#64748b;font-size:11px}
# .real{color:#4ade80;text-shadow:0 0 18px rgba(74,222,128,.25)}
# .fake{color:#fb7185;text-shadow:0 0 18px rgba(251,113,133,.25)}
# .unknown{color:#facc15}
# .pipeline-wrapper{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;padding:20px;border-radius:18px;background:rgba(15,23,42,.6);border:1px solid rgba(148,163,184,.12)}
# .pipeline-node{text-align:center;padding:13px 6px;border-radius:12px;background:rgba(124,58,237,.08);border:1px solid rgba(139,92,246,.2)}
# .pipeline-name{color:#e2e8f0;font-size:13px;font-weight:800}
# .pipeline-desc{margin-top:4px;color:#64748b;font-size:10px}
# .evidence-card{padding:18px;margin-bottom:10px;border-radius:14px;background:rgba(15,23,42,.62);border:1px solid rgba(148,163,184,.11)}
# .evidence-row{display:flex;justify-content:space-between;gap:20px}
# .evidence-title{color:#e2e8f0;font-size:13px;font-weight:700}
# .evidence-text{margin-top:5px;color:#94a3b8;font-size:12px;line-height:1.55}
# .evidence-score{color:#67e8f9;font-size:12px;font-weight:800}
# .reasoning-card{padding:23px;border-radius:18px;background:rgba(15,23,42,.68);border:1px solid rgba(148,163,184,.12);color:#cbd5e1;font-size:14px;line-height:1.75}
# .signal-card,.feature-card{padding:20px;border-radius:17px;background:rgba(15,23,42,.56);border:1px solid rgba(148,163,184,.12);transition:.2s ease}
# .signal-card{min-height:160px}
# .feature-card{min-height:165px}
# .signal-card:hover,.feature-card:hover{transform:translateY(-3px);border-color:rgba(139,92,246,.3)}
# .signal-icon,.feature-icon{margin-bottom:10px;color:#a78bfa;font-size:23px}
# .signal-title,.feature-title{color:#f1f5f9;font-size:14px;font-weight:800}
# .signal-text,.feature-text{margin-top:7px;color:#94a3b8;font-size:12px;line-height:1.6}
# .footer{margin-top:55px;padding-top:22px;border-top:1px solid rgba(148,163,184,.08);text-align:center;color:#475569;font-size:10px;letter-spacing:.7px}
# </style>
# """)

# # header
# l, r = st.columns([3, 1])
# with l:
#     md("""
#     <div class="header-box">
#         <div class="brand-title">◈ VERITY</div>
#         <div class="brand-subtitle">AI-Powered Factual Verification System</div>
#     </div>
#     """)
# with r:
#     md("""
#     <div class="status-wrapper"><div class="status"><span class="status-dot"></span>SYSTEM OPERATIONAL</div></div>
#     """)

# # hero
# md("""
# <div class="hero">
#     <div class="badge">AI-POWERED CLAIM ANALYSIS</div>
#     <div class="hero-title">Factual Verification,<br>Reimagined.</div>
#     <div class="hero-text">Analyze factual claims using semantic retrieval, graph reasoning, live web evidence, and LLM-powered explanation through one verification pipeline.</div>
# </div>
# """)

# # input
# md("""
# <div class="input-card">
#     <div class="input-title">Analyze a Claim</div>
#     <div class="input-subtitle">Enter a news statement or factual claim to begin verification.</div>
# </div>
# """)
# claim = st.text_area("Claim", placeholder="Example: NASA discovered life on Mars.", height=120, label_visibility="collapsed")
# verify = st.button("VERIFY CLAIM  →", use_container_width=True)

# if verify:
#     if not claim.strip():
#         st.warning("Please enter a claim before starting verification.")
#         st.stop()
#     try:
#         with st.spinner("Running VERITY verification pipeline..."):
#             result = classify_article(claim)
#             explanation = generate_explanation(claim, result)
#     except Exception as e:
#         st.error("Verification pipeline failed.")
#         with st.expander("Technical details"):
#             st.exception(e)
#         st.stop()

#     if not isinstance(result, dict): result = {}
#     if not isinstance(explanation, dict): explanation = {"reasoning": str(explanation)}

#     pred = str(result.get("prediction", "UNKNOWN")).upper()
#     gat = str(result.get("gat_prediction", "N/A")).upper()
#     try: conf = float(result.get("confidence", 0))
#     except Exception: conf = 0.0
#     if 0 < conf <= 1: conf *= 100  # pipeline may return a 0-1 fraction
#     conf = max(0.0, min(100.0, conf))
#     pc = "real" if pred in ("REAL", "TRUE") else "fake" if pred in ("FAKE", "FALSE") else "unknown"

#     # result cards
#     section("Verification Result", "Final assessment generated by the VERITY pipeline.")
#     c1, c2, c3 = st.columns(3)
#     cards = [
#         (c1, "Final Verdict", html.escape(pred), pc, "Overall verification result"),
#         (c2, "Confidence", f"{conf:.1f}%", "", "Model confidence"),
#         (c3, "GAT Prediction", html.escape(gat), "", "Graph-based prediction"),
#     ]
#     for c, lab, val, cls, desc in cards:
#         with c:
#             md(f"""
#             <div class="result-card">
#                 <div class="card-label">{lab}</div>
#                 <div class="card-value {cls}">{val}</div>
#                 <div class="card-description">{desc}</div>
#             </div>
#             """)

#     # pipeline (single html block so the wrapper actually wraps the nodes)
#     section("Verification Pipeline", "Multiple AI stages contribute to the final assessment.")
#     pipe = [("BERT", "Semantic Understanding"), ("RAG", "Evidence Retrieval"), ("GAT", "Graph Reasoning"), ("WEB", "Live Evidence"), ("LLM", "Explanation"), ("VERDICT", "Final Assessment")]
#     nodes = "".join(f'<div class="pipeline-node"><div class="pipeline-name">{n}</div><div class="pipeline-desc">{d}</div></div>' for n, d in pipe)
#     md(f'<div class="pipeline-wrapper">{nodes}</div>')

#     # analytics
#     section("Verification Analytics", "Confidence and retrieved-evidence signals from this analysis.")
#     ret = result.get("retrieved_articles") or result.get("retrieved") or result.get("evidence") or []
#     sims, ev = [], []
#     if isinstance(ret, list):
#         for i, it in enumerate(ret[:5]):
#             if not isinstance(it, dict): continue
#             s = it.get("similarity") or it.get("score") or it.get("similarity_score")
#             t = it.get("title") or it.get("headline") or it.get("text") or f"Retrieved Article {i + 1}"
#             if s is None: continue
#             try: s = float(s)
#             except Exception: continue
#             sims.append(s)
#             ev.append({"title": str(t)[:120], "score": s})

#     al, ar = st.columns([1, 1.25])
#     with al:
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number", value=conf,
#             number={"suffix": "%", "font": {"size": 32}},
#             title={"text": "Verification Confidence"},
#             gauge={"axis": {"range": [0, 100]}, "bar": {"thickness": 0.25, "color": "#a78bfa"},
#                    "steps": [{"range": [0, 50], "color": "#3f1d2e"}, {"range": [50, 75], "color": "#4a3b1b"}, {"range": [75, 100], "color": "#173b2c"}]},
#         ))
#         fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
#         st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
#     with ar:
#         if sims:
#             fig = go.Figure(go.Bar(x=[f"Article {i + 1}" for i in range(len(sims))], y=sims, text=[f"{v:.3f}" for v in sims], textposition="auto", marker_color="#8b5cf6"))
#             fig.update_layout(title="Top Retrieved Evidence", yaxis={"title": "Similarity", "range": [0, max(1.0, max(sims) + 0.1)]}, height=300,
#                               margin=dict(l=40, r=20, t=60, b=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "white"}, showlegend=False)
#             st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
#         else:
#             st.info("Similarity scores are not available for this result.")

#     # evidence
#     section("Retrieved Evidence", "Highest-ranked evidence retrieved for the submitted claim.")
#     if ev:
#         for i, it in enumerate(ev):
#             md(f"""
#             <div class="evidence-card">
#                 <div class="evidence-row">
#                     <div class="evidence-title">{i + 1}. {html.escape(it['title'])}</div>
#                     <div class="evidence-score">{it['score']:.3f}</div>
#                 </div>
#                 <div class="evidence-text">Semantic similarity score</div>
#             </div>
#             """)
#     else:
#         st.info("No structured retrieved evidence was returned by the pipeline.")

#     # reasoning
#     section("AI Reasoning", "Human-readable explanation generated from the verification results.")
#     rs = explanation.get("reasoning") or explanation.get("explanation") or explanation.get("verdict") or ""
#     rs = re.sub(r"```(?:html|markdown|md)?", "", str(rs), flags=re.I)
#     rs = re.sub(r"<[^>]*>", "", rs).replace("**", "").strip()
#     if not rs: rs = f"VERITY classified this claim as {pred} with {conf:.1f}% confidence."
#     rs = html.escape(rs).replace("\n", "<br>")
#     md(f'<div class="reasoning-card">{rs}</div>')

#     # signals
#     section("Verification Signals", "Components contributing evidence to the final assessment.")
#     eq = html.escape(str(explanation.get("evidence_quality", "Available")))
#     ws = html.escape(str(explanation.get("web_signal", "Available")))
#     sigs = [
#         ("◈", "Semantic Retrieval", "BERT-based semantic representation was used to understand the claim and retrieve related information."),
#         ("◎", "Graph Reasoning", f"GAT analyzed relationships between connected news articles. Current output: {html.escape(gat)}"),
#         ("⌁", "Web Evidence", f"Live evidence: {ws}<br><br>Evidence quality: {eq}"),
#     ]
#     for c, (ic, t, tx) in zip(st.columns(3), sigs):
#         with c:
#             md(f"""
#             <div class="signal-card">
#                 <div class="signal-icon">{ic}</div>
#                 <div class="signal-title">{t}</div>
#                 <div class="signal-text">{tx}</div>
#             </div>
#             """)

# # how it works
# section("How VERITY Works", "A multi-stage architecture for explainable factual verification.")
# feats = [
#     ("◉", "Semantic Retrieval", "BERT understands the meaning and context of the submitted claim."),
#     ("◎", "Graph Reasoning", "GAT analyzes relationships between connected news articles."),
#     ("⌁", "Live Evidence", "Web search provides additional external evidence for verification."),
#     ("✦", "LLM Verification", "An LLM converts technical verification results into a human-readable explanation."),
# ]
# for c, (ic, t, d) in zip(st.columns(4), feats):
#     with c:
#         md(f"""
#         <div class="feature-card">
#             <div class="feature-icon">{ic}</div>
#             <div class="feature-title">{t}</div>
#             <div class="feature-text">{d}</div>
#         </div>
#         """)

# # footer
# md("""
# <div class="footer">VERITY · AI-POWERED FACTUAL VERIFICATION<br><br>BERT · RAG · GAT · WEB EVIDENCE · LLM</div>
# """)


import re
import html
import streamlit as st
import plotly.graph_objects as go

from src.verity_pipeline import classify_article, generate_explanation


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VERITY | Factual Verification",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# HELPERS
# ============================================================

def md(s):
    """
    Render HTML safely through Streamlit.
    Removes indentation so HTML never becomes a code block.
    """
    st.markdown(
        "\n".join(
            line.strip()
            for line in s.splitlines()
            if line.strip()
        ),
        unsafe_allow_html=True
    )


def section(title, subtitle):
    md(
        f"""
        <div class="section-title">{html.escape(title)}</div>
        <div class="section-subtitle">{html.escape(subtitle)}</div>
        """
    )


def clean_text(text):
    """
    Remove accidental markdown/HTML from LLM output.
    """
    text = str(text or "")

    text = re.sub(
        r"```(?:html|markdown|md)?",
        "",
        text,
        flags=re.I
    )

    text = re.sub(
        r"<[^>]*>",
        "",
        text
    )

    text = text.replace("**", "")
    text = text.replace("__", "")

    return text.strip()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# CUSTOM CSS
# ============================================================

md(
"""
<style>

.stApp{
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(124,58,237,.18),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(6,182,212,.13),
            transparent 25%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(236,72,153,.08),
            transparent 30%
        ),
        #070b16;
    color:#f8fafc;
}

.block-container{
    max-width:1250px;
    padding-top:1.5rem;
    padding-bottom:3rem;
}

#MainMenu,
footer,
header{
    visibility:hidden;
}


/* ================= HEADER ================= */

.header-box{
    padding:18px 24px;
    border-radius:18px;
    background:rgba(15,23,42,.68);
    border:1px solid rgba(148,163,184,.14);
    backdrop-filter:blur(14px);
}

.brand-title{
    font-size:28px;
    font-weight:850;
    letter-spacing:4px;
    background:linear-gradient(
        90deg,
        #a78bfa,
        #67e8f9,
        #f472b6
    );
    -webkit-background-clip:text;
    background-clip:text;
    -webkit-text-fill-color:transparent;
}

.brand-subtitle{
    margin-top:3px;
    color:#64748b;
    font-size:11px;
    letter-spacing:1.2px;
    text-transform:uppercase;
}

.status-wrapper{
    text-align:right;
    padding-top:13px;
}

.status{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:8px 13px;
    border-radius:999px;
    background:rgba(34,197,94,.08);
    border:1px solid rgba(34,197,94,.2);
    color:#86efac;
    font-size:11px;
    font-weight:700;
    letter-spacing:.6px;
}

.status-dot{
    width:7px;
    height:7px;
    border-radius:50%;
    background:#22c55e;
    box-shadow:
        0 0 6px #22c55e,
        0 0 14px rgba(34,197,94,.7);
}


/* ================= HERO ================= */

.hero{
    text-align:center;
    padding:58px 20px 38px;
}

.badge{
    display:inline-block;
    padding:7px 14px;
    border-radius:999px;
    background:rgba(124,58,237,.12);
    border:1px solid rgba(167,139,250,.24);
    color:#c4b5fd;
    font-size:10px;
    font-weight:800;
    letter-spacing:1.5px;
}

.hero-title{
    margin:18px 0 12px;
    font-size:clamp(42px,6vw,70px);
    line-height:1.04;
    font-weight:850;
    background:linear-gradient(
        90deg,
        #fff,
        #c4b5fd,
        #67e8f9
    );
    -webkit-background-clip:text;
    background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero-text{
    max-width:700px;
    margin:auto;
    color:#94a3b8;
    font-size:15px;
    line-height:1.7;
}


/* ================= INPUT ================= */

.input-card{
    padding:25px;
    border-radius:20px;
    background:rgba(15,23,42,.72);
    border:1px solid rgba(148,163,184,.14);
    box-shadow:0 20px 60px rgba(0,0,0,.18);
    margin-bottom:12px;
}

.input-title{
    color:#f8fafc;
    font-size:20px;
    font-weight:800;
}

.input-subtitle{
    margin-top:4px;
    color:#64748b;
    font-size:13px;
}

textarea{
    background:rgba(2,6,23,.78)!important;
    color:#f8fafc!important;
    border:1px solid rgba(139,92,246,.28)!important;
    border-radius:13px!important;
    font-size:14px!important;
}

textarea:focus{
    border-color:#8b5cf6!important;
    box-shadow:0 0 0 1px #8b5cf6!important;
}

.stButton>button{
    height:50px;
    border:none!important;
    border-radius:13px!important;
    background:linear-gradient(
        90deg,
        #7c3aed,
        #6366f1,
        #06b6d4
    )!important;
    color:#fff!important;
    font-size:14px!important;
    font-weight:800!important;
    letter-spacing:.5px;
    box-shadow:0 10px 30px rgba(99,102,241,.24);
    transition:.2s ease;
}

.stButton>button:hover{
    transform:translateY(-2px);
    box-shadow:0 15px 40px rgba(6,182,212,.28);
}


/* ================= SECTIONS ================= */

.section-title{
    margin-top:40px;
    margin-bottom:4px;
    color:#f8fafc;
    font-size:25px;
    font-weight:800;
}

.section-subtitle{
    margin-bottom:18px;
    color:#64748b;
    font-size:13px;
}


/* ================= RESULT CARDS ================= */

.result-card{
    min-height:145px;
    padding:22px;
    border-radius:18px;
    background:
        linear-gradient(
            145deg,
            rgba(30,41,59,.8),
            rgba(15,23,42,.68)
        );
    border:1px solid rgba(148,163,184,.13);
    box-shadow:0 15px 40px rgba(0,0,0,.16);
}

.card-label{
    color:#64748b;
    font-size:10px;
    font-weight:800;
    letter-spacing:1.3px;
    text-transform:uppercase;
}

.card-value{
    margin-top:12px;
    color:#f8fafc;
    font-size:30px;
    font-weight:850;
}

.card-description{
    margin-top:5px;
    color:#64748b;
    font-size:11px;
}

.real{
    color:#4ade80;
    text-shadow:0 0 18px rgba(74,222,128,.25);
}

.fake{
    color:#fb7185;
    text-shadow:0 0 18px rgba(251,113,133,.25);
}

.unknown{
    color:#facc15;
}


/* ================= PIPELINE ================= */

.pipeline-wrapper{
    display:grid;
    grid-template-columns:repeat(6,1fr);
    gap:10px;
    padding:20px;
    border-radius:18px;
    background:rgba(15,23,42,.6);
    border:1px solid rgba(148,163,184,.12);
}

.pipeline-node{
    text-align:center;
    padding:13px 6px;
    border-radius:12px;
    background:rgba(124,58,237,.08);
    border:1px solid rgba(139,92,246,.2);
}

.pipeline-name{
    color:#e2e8f0;
    font-size:13px;
    font-weight:800;
}

.pipeline-desc{
    margin-top:4px;
    color:#64748b;
    font-size:10px;
}


/* ================= EVIDENCE ================= */

.evidence-card{
    padding:18px;
    margin-bottom:10px;
    border-radius:14px;
    background:rgba(15,23,42,.62);
    border:1px solid rgba(148,163,184,.11);
}

.evidence-row{
    display:flex;
    justify-content:space-between;
    gap:20px;
}

.evidence-title{
    color:#e2e8f0;
    font-size:13px;
    font-weight:700;
}

.evidence-text{
    margin-top:7px;
    color:#94a3b8;
    font-size:12px;
    line-height:1.6;
}

.evidence-score{
    color:#67e8f9;
    font-size:12px;
    font-weight:800;
    white-space:nowrap;
}

.evidence-meta{
    margin-top:8px;
    color:#64748b;
    font-size:10px;
}

.evidence-url{
    margin-top:8px;
    color:#818cf8;
    font-size:10px;
    word-break:break-all;
}


/* ================= EVIDENCE GROUP ================= */

.evidence-group{
    margin-top:15px;
    margin-bottom:12px;
    padding:12px 15px;
    border-radius:12px;
    background:rgba(124,58,237,.07);
    border:1px solid rgba(139,92,246,.14);
}

.evidence-group-title{
    color:#c4b5fd;
    font-size:12px;
    font-weight:800;
    letter-spacing:.8px;
    text-transform:uppercase;
}


/* ================= REASONING ================= */

.reasoning-card{
    padding:23px;
    border-radius:18px;
    background:rgba(15,23,42,.68);
    border:1px solid rgba(148,163,184,.12);
    color:#cbd5e1;
    font-size:14px;
    line-height:1.75;
}


/* ================= SIGNALS ================= */

.signal-card,
.feature-card{
    padding:20px;
    border-radius:17px;
    background:rgba(15,23,42,.56);
    border:1px solid rgba(148,163,184,.12);
    transition:.2s ease;
}

.signal-card{
    min-height:160px;
}

.feature-card{
    min-height:165px;
}

.signal-card:hover,
.feature-card:hover{
    transform:translateY(-3px);
    border-color:rgba(139,92,246,.3);
}

.signal-icon,
.feature-icon{
    margin-bottom:10px;
    color:#a78bfa;
    font-size:23px;
}

.signal-title,
.feature-title{
    color:#f1f5f9;
    font-size:14px;
    font-weight:800;
}

.signal-text,
.feature-text{
    margin-top:7px;
    color:#94a3b8;
    font-size:12px;
    line-height:1.6;
}


/* ================= FOOTER ================= */

.footer{
    margin-top:55px;
    padding-top:22px;
    border-top:1px solid rgba(148,163,184,.08);
    text-align:center;
    color:#475569;
    font-size:10px;
    letter-spacing:.7px;
}


/* ================= MOBILE ================= */

@media(max-width:900px){

    .pipeline-wrapper{
        grid-template-columns:repeat(3,1fr);
    }

}

@media(max-width:600px){

    .pipeline-wrapper{
        grid-template-columns:repeat(2,1fr);
    }

    .hero-title{
        font-size:42px;
    }

}

</style>
"""
)


# ============================================================
# HEADER
# ============================================================

left, right = st.columns([3, 1])

with left:
    md(
        """
        <div class="header-box">
            <div class="brand-title">◈ VERITY</div>
            <div class="brand-subtitle">
                AI-Powered Factual Verification System
            </div>
        </div>
        """
    )

with right:
    md(
        """
        <div class="status-wrapper">
            <div class="status">
                <span class="status-dot"></span>
                SYSTEM OPERATIONAL
            </div>
        </div>
        """
    )


# ============================================================
# HERO
# ============================================================

md(
    """
    <div class="hero">

        <div class="badge">
            AI-POWERED CLAIM ANALYSIS
        </div>

        <div class="hero-title">
            Factual Verification,<br>
            Reimagined.
        </div>

        <div class="hero-text">
            Analyze factual claims using semantic retrieval,
            graph reasoning, live web evidence, and
            LLM-powered verification through one pipeline.
        </div>

    </div>
    """
)


# ============================================================
# INPUT
# ============================================================

md(
    """
    <div class="input-card">

        <div class="input-title">
            Analyze a Claim
        </div>

        <div class="input-subtitle">
            Enter a news statement or factual claim to begin verification.
        </div>

    </div>
    """
)

claim = st.text_area(
    "Claim",
    placeholder="Example: NASA discovered life on Mars.",
    height=120,
    label_visibility="collapsed"
)

verify = st.button(
    "VERIFY CLAIM  →",
    use_container_width=True
)


# ============================================================
# VERIFICATION
# ============================================================

if verify:

    if not claim.strip():
        st.warning(
            "Please enter a claim before starting verification."
        )
        st.stop()

    try:

        with st.spinner(
            "Running VERITY verification pipeline..."
        ):

            result = classify_article(claim)

            explanation = generate_explanation(
                claim,
                result
            )

    except Exception as e:

        st.error(
            "Verification pipeline failed."
        )

        with st.expander(
            "Technical details"
        ):
            st.exception(e)

        st.stop()


    # ========================================================
    # NORMALIZE RESULT
    # ========================================================

    if not isinstance(result, dict):
        result = {}

    if not isinstance(explanation, dict):
        explanation = {
            "reasoning": str(explanation)
        }


    pred = str(
        result.get(
            "prediction",
            "UNKNOWN"
        )
    ).upper()

    gat = str(
        result.get(
            "gat_prediction",
            "N/A"
        )
    ).upper()


    conf = safe_float(
        result.get(
            "confidence",
            0
        )
    )

    if 0 < conf <= 1:
        conf *= 100

    conf = max(
        0.0,
        min(
            100.0,
            conf
        )
    )


    evidence_status = str(
        result.get(
            "evidence_status",
            "UNKNOWN"
        )
    ).upper()


    llm_verdict = str(
        result.get(
            "llm_verdict",
            "UNCERTAIN"
        )
    ).upper()


    # ========================================================
    # RESULT COLOR
    # ========================================================

    if pred in ("REAL", "TRUE", "SUPPORTS"):
        prediction_class = "real"

    elif pred in ("FAKE", "FALSE", "CONTRADICTS"):
        prediction_class = "fake"

    else:
        prediction_class = "unknown"


    # ========================================================
    # VERIFICATION RESULT
    # ========================================================

    section(
        "Verification Result",
        "Final assessment generated by the VERITY pipeline."
    )


    c1, c2, c3 = st.columns(3)


    cards = [

        (
            c1,
            "Final Verdict",
            html.escape(pred),
            prediction_class,
            "Overall verification result"
        ),

        (
            c2,
            "Confidence",
            f"{conf:.1f}%",
            "",
            "LLM verification confidence"
        ),

        (
            c3,
            "GAT Prediction",
            html.escape(gat),
            "",
            "Graph-based prediction"
        ),

    ]


    for column, label, value, css_class, description in cards:

        with column:

            md(
                f"""
                <div class="result-card">

                    <div class="card-label">
                        {label}
                    </div>

                    <div class="card-value {css_class}">
                        {value}
                    </div>

                    <div class="card-description">
                        {description}
                    </div>

                </div>
                """
            )


    # ========================================================
    # PIPELINE
    # ========================================================

    section(
        "Verification Pipeline",
        "Multiple AI stages contribute to the final assessment."
    )


    pipeline = [

        (
            "BERT",
            "Semantic Understanding"
        ),

        (
            "RAG",
            "Evidence Retrieval"
        ),

        (
            "GAT",
            "Graph Reasoning"
        ),

        (
            "WEB",
            "Live Evidence"
        ),

        (
            "LLM",
            "Evidence Verification"
        ),

        (
            "VERDICT",
            "Final Assessment"
        )

    ]


    nodes = "".join(

        f"""
        <div class="pipeline-node">

            <div class="pipeline-name">
                {html.escape(name)}
            </div>

            <div class="pipeline-desc">
                {html.escape(description)}
            </div>

        </div>
        """

        for name, description in pipeline

    )


    md(
        f"""
        <div class="pipeline-wrapper">
            {nodes}
        </div>
        """
    )


    # ========================================================
    # GET STRUCTURED BACKEND DATA
    # ========================================================

    # IMPORTANT:
    # These are the exact keys produced by the updated
    # verity_pipeline.py.

    similarity_scores = result.get(
        "similarity_scores",
        []
    )

    rag_evidence = result.get(
        "rag_evidence",
        []
    )

    web_evidence = result.get(
        "structured_web_evidence",
        []
    )

    llm_sources = result.get(
        "llm_sources",
        []
    )


    if not isinstance(similarity_scores, list):
        similarity_scores = []

    if not isinstance(rag_evidence, list):
        rag_evidence = []

    if not isinstance(web_evidence, list):
        web_evidence = []

    if not isinstance(llm_sources, list):
        llm_sources = []


    # ========================================================
    # ANALYTICS
    # ========================================================

    section(
        "Verification Analytics",
        "Semantic retrieval strength and verification confidence."
    )


    al, ar = st.columns([1, 1.25])


    # --------------------------------------------------------
    # CONFIDENCE GAUGE
    # --------------------------------------------------------

    with al:

        fig = go.Figure(
            go.Indicator(

                mode="gauge+number",

                value=conf,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 32
                    }
                },

                title={
                    "text": "Verification Confidence"
                },

                gauge={

                    "axis": {
                        "range": [
                            0,
                            100
                        ]
                    },

                    "bar": {
                        "thickness": 0.25,
                        "color": "#a78bfa"
                    },

                    "steps": [

                        {
                            "range": [
                                0,
                                50
                            ],
                            "color": "#3f1d2e"
                        },

                        {
                            "range": [
                                50,
                                75
                            ],
                            "color": "#4a3b1b"
                        },

                        {
                            "range": [
                                75,
                                100
                            ],
                            "color": "#173b2c"
                        }

                    ]

                }

            )
        )


        fig.update_layout(

            height=300,

            margin=dict(
                l=20,
                r=20,
                t=60,
                b=10
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            font={
                "color": "white"
            }

        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    # --------------------------------------------------------
    # SIMILARITY CHART
    # --------------------------------------------------------

    with ar:

        valid_scores = []

        for score in similarity_scores[:5]:

            value = safe_float(
                score,
                None
            )

            if value is not None:
                valid_scores.append(value)


        if valid_scores:

            labels = [
                f"Article {i + 1}"
                for i in range(
                    len(valid_scores)
                )
            ]


            fig = go.Figure(

                go.Bar(

                    x=labels,

                    y=valid_scores,

                    text=[
                        f"{value:.3f}"
                        for value in valid_scores
                    ],

                    textposition="auto",

                    marker_color="#8b5cf6"

                )

            )


            upper = max(
                1.0,
                max(valid_scores) + 0.1
            )


            fig.update_layout(

                title="Top Retrieved Evidence",

                yaxis={
                    "title": "Similarity",
                    "range": [
                        0,
                        upper
                    ]
                },

                height=300,

                margin=dict(
                    l=40,
                    r=20,
                    t=60,
                    b=40
                ),

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                font={
                    "color": "white"
                },

                showlegend=False

            )


            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )

        else:

            st.info(
                "No semantic similarity scores were returned."
            )


    # ========================================================
    # RAG EVIDENCE
    # ========================================================

    section(
        "Retrieved Evidence",
        "Top documents retrieved by the semantic RAG stage."
    )


    if rag_evidence:

        md(
            """
            <div class="evidence-group">
                <div class="evidence-group-title">
                    RAG / Semantic Retrieval
                </div>
            </div>
            """
        )


        for i, item in enumerate(
            rag_evidence[:5],
            start=1
        ):

            if not isinstance(item, dict):
                continue


            title = clean_text(
                item.get(
                    "title",
                    f"Retrieved Article {i}"
                )
            )


            snippet = clean_text(
                item.get(
                    "snippet",
                    item.get(
                        "text",
                        ""
                    )
                )
            )


            similarity = safe_float(
                item.get(
                    "similarity",
                    item.get(
                        "score",
                        0
                    )
                )
            )


            label = clean_text(
                item.get(
                    "label_name",
                    item.get(
                        "label",
                        "N/A"
                    )
                )
            )


            url = clean_text(
                item.get(
                    "url",
                    ""
                )
            )


            md(
                f"""
                <div class="evidence-card">

                    <div class="evidence-row">

                        <div class="evidence-title">
                            {i}. {html.escape(title)}
                        </div>

                        <div class="evidence-score">
                            {similarity:.3f}
                        </div>

                    </div>

                    <div class="evidence-text">
                        {html.escape(snippet[:600])}
                    </div>

                    <div class="evidence-meta">
                        RAG similarity: {similarity:.3f}
                        &nbsp; · &nbsp;
                        Dataset label: {html.escape(label)}
                    </div>

                    {
                        f'<div class="evidence-url">{html.escape(url)}</div>'
                        if url else ""
                    }

                </div>
                """
            )

    else:

        st.info(
            "No RAG evidence was returned for this claim."
        )


    # ========================================================
    # WEB EVIDENCE
    # ========================================================

    section(
        "Live Web Evidence",
        "External evidence retrieved during verification."
    )


    if web_evidence:

        md(
            """
            <div class="evidence-group">
                <div class="evidence-group-title">
                    Live Web Search
                </div>
            </div>
            """
        )


        for i, item in enumerate(
            web_evidence[:8],
            start=1
        ):

            if not isinstance(item, dict):
                continue


            title = clean_text(
                item.get(
                    "title",
                    f"Web Source {i}"
                )
            )


            snippet = clean_text(
                item.get(
                    "snippet",
                    ""
                )
            )


            url = clean_text(
                item.get(
                    "url",
                    ""
                )
            )


            relation = clean_text(
                item.get(
                    "relation",
                    "UNCERTAIN"
                )
            ).upper()


            quality = safe_float(
                item.get(
                    "quality",
                    0
                )
            )


            overlap = safe_float(
                item.get(
                    "overlap",
                    0
                )
            )


            relation_display = relation


            if relation == "SUPPORTS":
                relation_display = "SUPPORTS CLAIM"

            elif relation == "CONTRADICTS":
                relation_display = "CONTRADICTS CLAIM"

            elif relation == "UNCERTAIN":
                relation_display = "UNCERTAIN"


            md(
                f"""
                <div class="evidence-card">

                    <div class="evidence-row">

                        <div class="evidence-title">
                            {i}. {html.escape(title)}
                        </div>

                        <div class="evidence-score">
                            {html.escape(relation_display)}
                        </div>

                    </div>

                    <div class="evidence-text">
                        {html.escape(snippet[:700])}
                    </div>

                    <div class="evidence-meta">
                        Source quality: {quality:.2f}
                        &nbsp; · &nbsp;
                        Text overlap: {overlap:.2f}
                    </div>

                    {
                        f'<div class="evidence-url">{html.escape(url)}</div>'
                        if url else ""
                    }

                </div>
                """
            )

    else:

        st.info(
            "No live web evidence was returned."
        )


    # ========================================================
    # LLM SELECTED SOURCES
    # ========================================================

    section(
        "Evidence Used for Verification",
        "Sources explicitly selected by the LLM as directly relevant to the claim."
    )


    if llm_sources:

        source_text = ", ".join(
            str(x)
            for x in llm_sources
        )

        md(
            f"""
            <div class="reasoning-card">
                LLM-selected source indices:
                <strong>{html.escape(source_text)}</strong>
            </div>
            """
        )

    else:

        st.info(
            "The LLM did not identify specific source indices."
        )


    # ========================================================
    # AI REASONING
    # ========================================================

    section(
        "AI Reasoning",
        "Human-readable explanation generated from the verification evidence."
    )


    # First preference:
    # actual LLM reason returned by backend.

    reasoning = (

        result.get(
            "llm_reason",
            ""
        )

        or

        explanation.get(
            "reasoning",
            ""
        )

        or

        explanation.get(
            "explanation",
            ""
        )

        or

        explanation.get(
            "verdict",
            ""
        )

    )


    reasoning = clean_text(
        reasoning
    )


    if not reasoning:

        reasoning = (
            f"VERITY classified this claim as "
            f"{pred} with {conf:.1f}% confidence."
        )


    md(
        f"""
        <div class="reasoning-card">
            {html.escape(reasoning).replace(chr(10), "<br>")}
        </div>
        """
    )


    # ========================================================
    # VERIFICATION SIGNALS
    # ========================================================

    section(
        "Verification Signals",
        "Individual signals produced by the verification pipeline."
    )


    # --------------------------------------------------------
    # WEB SIGNAL
    # --------------------------------------------------------

    if web_evidence:

        web_signal = "AVAILABLE"

    else:

        web_signal = "NOT AVAILABLE"


    # --------------------------------------------------------
    # EVIDENCE STATUS
    # --------------------------------------------------------

    if evidence_status:

        evidence_quality = evidence_status

    else:

        evidence_quality = "UNKNOWN"


    # --------------------------------------------------------
    # RAG SIGNAL
    # --------------------------------------------------------

    if similarity_scores:

        semantic_signal = (
            f"{len(similarity_scores)} "
            f"semantic matches retrieved"
        )

    else:

        semantic_signal = (
            "No semantic similarity scores returned"
        )


    signals = [

        (
            "◈",
            "Semantic Retrieval",
            (
                "BERT-based semantic representation was used "
                "to understand the claim and retrieve related "
                f"information.<br><br>{html.escape(semantic_signal)}."
            )
        ),

        (
            "◎",
            "Graph Reasoning",
            (
                "GAT analyzed relationships between connected "
                f"news articles.<br><br>"
                f"Current output: {html.escape(gat)}"
            )
        ),

        (
            "⌁",
            "Web Evidence",
            (
                f"Live evidence: "
                f"{html.escape(web_signal)}"
                f"<br><br>"
                f"Verification status: "
                f"{html.escape(evidence_quality)}"
            )
        )

    ]


    columns = st.columns(3)


    for column, signal in zip(
        columns,
        signals
    ):

        icon, title, text = signal

        with column:

            md(
                f"""
                <div class="signal-card">

                    <div class="signal-icon">
                        {icon}
                    </div>

                    <div class="signal-title">
                        {title}
                    </div>

                    <div class="signal-text">
                        {text}
                    </div>

                </div>
                """
            )


    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    with st.expander(
        "Pipeline Details"
    ):

        st.write(
            "LLM Verdict:",
            llm_verdict
        )

        st.write(
            "Evidence Status:",
            evidence_status
        )

        st.write(
            "RAG Evidence Count:",
            len(rag_evidence)
        )

        st.write(
            "Web Evidence Count:",
            len(web_evidence)
        )

        st.write(
            "Similarity Scores:",
            similarity_scores
        )

        st.write(
            "LLM Source Indices:",
            llm_sources
        )


# ============================================================
# HOW VERITY WORKS
# ============================================================

section(
    "How VERITY Works",
    "A multi-stage architecture for explainable factual verification."
)


features = [

    (
        "◉",
        "Semantic Retrieval",
        "BERT understands the meaning and context of the submitted claim."
    ),

    (
        "◎",
        "Graph Reasoning",
        "GAT analyzes relationships between connected news articles."
    ),

    (
        "⌁",
        "Live Evidence",
        "Web search provides additional external evidence for verification."
    ),

    (
        "✦",
        "LLM Verification",
        "An LLM evaluates the retrieved evidence and produces an explanation."
    )

]


for column, feature in zip(
    st.columns(4),
    features
):

    icon, title, description = feature

    with column:

        md(
            f"""
            <div class="feature-card">

                <div class="feature-icon">
                    {icon}
                </div>

                <div class="feature-title">
                    {html.escape(title)}
                </div>

                <div class="feature-text">
                    {html.escape(description)}
                </div>

            </div>
            """
        )


# ============================================================
# FOOTER
# ============================================================

md(
    """
    <div class="footer">

        VERITY · AI-POWERED FACTUAL VERIFICATION

        <br><br>

        BERT · RAG · GAT · WEB EVIDENCE · LLM

    </div>
    """
)
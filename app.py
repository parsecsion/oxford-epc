"""THE OXFORD ENERGY SURVEY — interactive dashboard (Streamlit).

A civic-editorial front-end for the COM6003 analysis. Rather than a generic
dark SaaS dashboard, this is styled as a typeset municipal survey document:
Oxford Blue ink on cream paper, the EPC certificate's A-G chevron band as the
masthead motif, Fraunces / Spectral / Space Mono typography, paper grain and a
warm radial wash for depth. It hosts the chart types that suit an interactive
surface (KPI readouts, a gauge, interactive distribution / part-to-whole /
relationship charts) plus a live band predictor that runs the frozen champion
for every property type.

Run:  streamlit run app.py
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parent
RATING_ORDER = list("ABCDEFG")

# ---- palette ----------------------------------------------------------------
INK = "#002147"        # Oxford Blue — the dominant ink
INK_SOFT = "#46566b"   # muted ink for secondary text
PAPER = "#F3EFE4"      # warm cream paper
PLATE = "#FBFAF4"      # slightly lighter "plate" for chart cards
HAIR = "#D9D1BE"       # hairline rule
GOLD = "#B0883C"       # signature gilt accent
# refined, earthy EPC spectrum (sits on cream far better than traffic-light)
BAND_COLORS = {"A": "#1F7A4D", "B": "#5D9B57", "C": "#9DBA59", "D": "#E4C04A",
               "E": "#DD9A3C", "F": "#CF6B33", "G": "#B23A2E"}

st.set_page_config(page_title="The Oxford Energy Survey", page_icon="▰",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------
# Design system — fonts, paper, grain, components
# ----------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=Spectral:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Space+Mono:wght@400;700&display=swap');

:root {{
  --ink:{INK}; --ink-soft:{INK_SOFT}; --paper:{PAPER}; --plate:{PLATE};
  --hair:{HAIR}; --gold:{GOLD};
}}

/* ---- paper + atmosphere ---- */
.stApp {{
  background:
    radial-gradient(1100px 600px at 78% -8%, rgba(176,136,60,0.10), transparent 60%),
    radial-gradient(900px 700px at -6% 110%, rgba(0,33,71,0.06), transparent 55%),
    var(--paper);
  color: var(--ink);
}}
.block-container {{ padding-top: 1.6rem; max-width: 1320px; }}
/* Hide ONLY the deploy button + main menu — NOT the whole stToolbar, because the
   sidebar-reopen control (stExpandSidebarButton) is a child of stToolbar and a
   display:none ancestor makes it unrenderable (the bug behind "can't reopen"). */
#MainMenu, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], footer {{ display:none !important; }}
header[data-testid="stHeader"] {{ background:transparent !important; box-shadow:none !important; }}
/* reopen-sidebar chevron: pin top-left, on-brand, unmissable */
[data-testid="stExpandSidebarButton"] {{ position:fixed !important; top:.7rem; left:.7rem; z-index:1001;
  background:var(--ink) !important; color:#fff !important; border-radius:8px !important; border:none !important;
  width:2.5rem !important; height:2.5rem !important; box-shadow:0 3px 10px rgba(0,33,71,.35) !important;
  display:inline-flex !important; align-items:center; justify-content:center; }}
[data-testid="stExpandSidebarButton"] svg {{ fill:#fff !important; color:#fff !important; }}
[data-testid="stSidebarCollapseButton"] button svg {{ fill:#e9e2cf !important; color:#e9e2cf !important; }}
/* native per-chart fullscreen button — on-brand and visible */
button[aria-label="Fullscreen"] svg, button[aria-label="Close fullscreen"] svg {{
  fill:var(--ink) !important; color:var(--ink) !important; }}

/* ---- typography ---- */
html, body, [class*="css"], .stMarkdown, p, li, label, .stMarkdown p {{
  font-family:'Spectral', Georgia, serif; color:var(--ink); font-size:16px;
}}
h1,h2,h3,h4 {{ font-family:'Fraunces', Georgia, serif !important; color:var(--ink); letter-spacing:-.01em; }}

/* ---- masthead ---- */
.masthead {{ margin:0 0 .4rem 0; animation:fadeUp .6s cubic-bezier(.2,.7,.2,1) both; }}
.mast-kicker {{ font-family:'Space Mono',monospace; font-size:12px; letter-spacing:.32em;
  text-transform:uppercase; color:var(--gold); margin-bottom:.5rem; }}
.mast-title {{ font-family:'Fraunces',serif; font-weight:900; font-size:clamp(2.6rem,6vw,4.6rem);
  line-height:.95; margin:0; letter-spacing:-.02em; }}
.mast-title em {{ font-style:italic; font-weight:400; color:var(--gold); }}
.mast-sub {{ font-size:1.05rem; color:var(--ink-soft); margin-top:.7rem; max-width:60ch;
  border-left:2px solid var(--gold); padding-left:.85rem; font-style:italic; }}

/* ---- EPC chevron band (the signature motif) ---- */
.epc-bands {{ display:flex; align-items:center; gap:5px; margin:1.4rem 0 .2rem; flex-wrap:wrap;
  animation:fadeUp .6s .12s cubic-bezier(.2,.7,.2,1) both; }}
.epc-chev {{ height:30px; display:flex; align-items:center; padding:0 20px 0 12px; color:#fff;
  font-family:'Space Mono',monospace; font-weight:700; font-size:13px;
  clip-path:polygon(0 0, calc(100% - 13px) 0, 100% 50%, calc(100% - 13px) 100%, 0 100%);
  box-shadow:0 1px 2px rgba(0,0,0,.18); }}
.rule {{ border:none; border-top:1px solid var(--hair); margin:1.3rem 0 1.6rem; }}

/* ---- section headers ---- */
.sec {{ margin:1.9rem 0 .5rem; animation:fadeUp .5s cubic-bezier(.2,.7,.2,1) both; }}
.sec-num {{ font-family:'Space Mono',monospace; color:var(--gold); font-size:.82rem;
  letter-spacing:.14em; margin-right:.7rem; }}
.sec-title {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.5rem; }}
.sec-sub {{ color:var(--ink-soft); font-style:italic; margin-top:.15rem; font-size:.98rem; }}

/* ---- KPI readouts ---- */
.kpi-row {{ display:flex; gap:0; flex-wrap:wrap; margin:.4rem 0 .3rem;
  border-top:2px solid var(--ink); }}
.kpi {{ flex:1; min-width:150px; padding:1.05rem 1.2rem 1.1rem; border-right:1px solid var(--hair);
  animation:fadeUp .55s cubic-bezier(.2,.7,.2,1) both; }}
.kpi:last-child {{ border-right:none; }}
.kpi:nth-child(1){{animation-delay:.04s}} .kpi:nth-child(2){{animation-delay:.10s}}
.kpi:nth-child(3){{animation-delay:.16s}} .kpi:nth-child(4){{animation-delay:.22s}}
.kpi:nth-child(5){{animation-delay:.28s}}
.kpi-label {{ font-family:'Space Mono',monospace; font-size:.7rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink-soft); }}
.kpi-value {{ font-family:'Fraunces',serif; font-weight:900; font-size:2.15rem; line-height:1.05;
  margin:.25rem 0 .1rem; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.kpi-note {{ font-size:.84rem; color:var(--ink-soft); font-style:italic; }}

/* ---- chart "plates" ---- (no padding: it conflicts with Plotly's width calc
   and causes the right-edge cut-off + scrollbar; the chart's own margins breathe) */
[data-testid="stFullScreenFrame"] {{ background:#FFFFFF; border:1px solid var(--hair);
  border-radius:6px; box-shadow:0 10px 26px -20px rgba(0,33,71,.5); overflow:hidden; }}
div[data-testid="stPlotlyChart"] {{ background:transparent; overflow:hidden; }}
div[data-testid="stPlotlyChart"] > div, .js-plotly-plot, .plot-container {{ width:100% !important; }}

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {{ background:#10243d; border-right:1px solid #0a1a2e; }}
section[data-testid="stSidebar"] * {{ color:#e9e2cf !important; }}
section[data-testid="stSidebar"] .stRadio label {{ font-family:'Spectral',serif; font-size:1.02rem; }}
.side-mark {{ font-family:'Fraunces',serif; font-weight:900; font-size:1.35rem; line-height:1.05;
  color:#fff !important; margin:.2rem 0 .1rem; }}
.side-kick {{ font-family:'Space Mono',monospace; font-size:.66rem; letter-spacing:.2em;
  text-transform:uppercase; color:var(--gold) !important; }}

/* ---- inputs / buttons ---- (force high contrast in BOTH states for primary
   and secondary, else the gold theme bg + cream text is invisible until hover) */
.stButton>button, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {{
  font-family:'Space Mono',monospace !important; letter-spacing:.06em; text-transform:uppercase;
  font-size:.8rem; background:var(--ink) !important; color:#fff !important; border:none !important;
  border-radius:2px; padding:.6rem 1.2rem; transition:transform .15s ease, background .15s ease; }}
.stButton>button *, [data-testid="stBaseButton-primary"] *, [data-testid="stBaseButton-secondary"] * {{
  color:#fff !important; }}
.stButton>button:hover, [data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-secondary"]:hover {{
  background:var(--gold) !important; transform:translateY(-1px); }}
.stButton>button:hover * {{ color:#1a1308 !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap:1.4rem; }}
.stDataFrame {{ border:1px solid var(--hair); }}
/* multiselect tags — ink chip with WHITE text (was invisible: no colour set) */
span[data-baseweb="tag"] {{ background:var(--ink) !important; font-family:'Space Mono',monospace;
  font-size:.72rem !important; border-radius:2px !important; }}
span[data-baseweb="tag"] span, span[data-baseweb="tag"] svg {{ color:#fff !important; fill:#fff !important; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background:var(--gold) !important; }}
/* dropdown option text + select value: keep dark on the light popovers */
[data-baseweb="popover"] li, [data-baseweb="menu"] li,
div[data-testid="stMain"] [data-baseweb="select"] div {{ color:var(--ink) !important; }}

@keyframes fadeUp {{ from {{opacity:0; transform:translateY(15px);}} to {{opacity:1; transform:none;}} }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Plotly template tuned to the editorial palette
# ----------------------------------------------------------------------
pio.templates["oxford"] = go.layout.Template(layout=dict(
    font=dict(family="Spectral, Georgia, serif", color=INK, size=13),
    title=dict(font=dict(family="Fraunces, Georgia, serif", size=16, color=INK), x=0.01, xanchor="left"),
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    colorway=[INK, GOLD, "#5D9B57", "#CF6B33", "#6E7B8B", "#9DBA59", "#9E2B25"],
    xaxis=dict(gridcolor="rgba(0,33,71,0.10)", linecolor="rgba(0,33,71,0.40)",
               zerolinecolor="rgba(0,33,71,0.18)", tickfont=dict(family="Space Mono, monospace", size=11)),
    yaxis=dict(gridcolor="rgba(0,33,71,0.10)", linecolor="rgba(0,33,71,0.40)",
               zerolinecolor="rgba(0,33,71,0.18)", tickfont=dict(family="Space Mono, monospace", size=11)),
    legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=54, r=22, t=52, b=44),
    colorscale=dict(sequential=[[0, "#F3EFE4"], [0.5, GOLD], [1, INK]]),
))
pio.templates.default = "oxford"


# Interactivity + native fullscreen for every chart; no Plotly logo.
PLOTLY_CONFIG = {
    "displaylogo": False, "responsive": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def style_fig(fig, h=360):
    """Uniform robust layout. Categorical legends go ABOVE the plot (so they
    never collide with the x-axis title); colourbars stay on the right with
    room reserved — both handled per-chart where present."""
    fig.update_layout(
        height=h, autosize=True, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        margin=dict(l=66, r=44, t=64, b=56),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11), title_font=dict(size=11)),
    )
    fig.update_xaxes(automargin=True, title_font=dict(size=12))
    fig.update_yaxes(automargin=True, title_font=dict(size=12))
    return fig


def cbar_right(fig, title, rmargin=96):
    """Put a continuous colourbar on the right with reserved margin + title."""
    fig.update_layout(margin_r=rmargin,
                      coloraxis_colorbar=dict(orientation="v", x=1.02, xanchor="left",
                                              thickness=12, len=0.85, title=dict(text=title, side="right")))
    return fig


def show(fig, h=360):
    """style + render with the interactive/fullscreen config in one call."""
    style_fig(fig, h)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ----------------------------------------------------------------------
# Editorial helpers
# ----------------------------------------------------------------------
def section(num, title, sub=""):
    sub_html = f'<div class="sec-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="sec"><span class="sec-num">§{num}</span>'
                f'<span class="sec-title">{title}</span>{sub_html}</div>', unsafe_allow_html=True)


def kpis(items):
    """items: list of (label, value, note)."""
    cards = "".join(
        f'<div class="kpi"><div class="kpi-label">{l}</div>'
        f'<div class="kpi-value">{v}</div><div class="kpi-note">{n}</div></div>'
        for l, v, n in items)
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)


def chevrons():
    cells = "".join(
        f'<div class="epc-chev" style="background:{BAND_COLORS[b]};width:{52 + i*11}px">{b}</div>'
        for i, b in enumerate(RATING_ORDER))
    st.markdown(f'<div class="epc-bands">{cells}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_clean() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "processed" / "oxford_epc_clean.csv", low_memory=False)
    df["SAP"] = pd.to_numeric(df["CURRENT_ENERGY_EFFICIENCY"], errors="coerce")
    df["YEAR"] = pd.to_datetime(df["INSPECTION_DATE"], errors="coerce").dt.year
    df["CO2"] = pd.to_numeric(df.get("CO2_EMISSIONS_CURRENT"), errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    p = ROOT / "reports" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    p = ROOT / "reports" / "predictions_oxford.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


df = load_clean()
art = load_json("champion_artefact.json")
rob = load_json("champion_robustness.json")
sig = load_json("champion_significance.json")
clu = load_json("clustering.json")
rec = load_json("recommender.json")
prop = load_json("property_analysis.json")
exp = load_json("champion_explanations.json")
preds = load_predictions()

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.markdown('<div class="side-kick">Local Authority E07000178</div>'
                    '<div class="side-mark">The Oxford<br>Energy Survey</div>', unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border-color:#23415f'>", unsafe_allow_html=True)
PAGES = ["Overview", "Descriptive", "Diagnostic", "Predictive model",
         "Band predictor", "Clustering", "Recommender"]
page = st.sidebar.radio("Folio", PAGES, label_visibility="collapsed")
st.sidebar.markdown("<hr style='border-color:#23415f'>", unsafe_allow_html=True)
st.sidebar.caption(f"{len(df):,} cleaned dwellings · COM6003 Data Science")


def band_counts(frame):
    return frame["CURRENT_ENERGY_RATING"].value_counts().reindex(RATING_ORDER).fillna(0)


def masthead(title_html, sub):
    st.markdown(f'<div class="masthead"><div class="mast-kicker">City of Oxford · '
                f'Domestic EPC register</div><h1 class="mast-title">{title_html}</h1>'
                f'<div class="mast-sub">{sub}</div></div>', unsafe_allow_html=True)


# ======================================================================
# Overview
# ======================================================================
if page == "Overview":
    masthead('The Oxford <em>Energy</em> Survey',
             "Predicting the statutory EPC band of a dwelling from its physical "
             "fabric alone — across 49,442 cleaned certificates for the city.")
    chevrons()
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    qwk = art.get("holdout_overall", {}).get("qwk", float("nan"))
    w15 = rob.get("overall", {}).get("within_15", float("nan"))
    pct_cplus = 100 * df["CURRENT_ENERGY_RATING"].isin(["A", "B", "C"]).mean()
    kpis([
        ("Cleaned dwellings", f"{len(df):,}", "deduplicated certificates"),
        ("Mean SAP score", f"{df['SAP'].mean():.0f}", "on the C/D boundary"),
        ("Band C or better", f"{pct_cplus:.0f}%", "of the stock"),
        ("Champion QWK", f"{qwk:.3f}", "strict temporal hold-out"),
        ("Within 15 SAP pts", f"{w15*100:.0f}%" if w15 == w15 else "—", "score reliability"),
    ])

    g, d = st.columns([1, 1.25])
    with g:
        section("01", "Model quality", "Quadratic Weighted Kappa, 0–1")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=qwk, number={"valueformat": ".3f", "font": {"size": 46}},
            gauge={"axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": INK_SOFT},
                   "bar": {"color": INK, "thickness": 0.28},
                   "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                   "steps": [{"range": [0, 0.4], "color": "rgba(178,58,46,0.16)"},
                             {"range": [0.4, 0.6], "color": "rgba(221,154,60,0.16)"},
                             {"range": [0.6, 1.0], "color": "rgba(31,122,77,0.16)"}],
                   "threshold": {"line": {"color": GOLD, "width": 4}, "thickness": 0.85, "value": qwk}}))
        style_fig(fig, 320)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    with d:
        section("02", "Band composition", "Share of dwellings by EPC band")
        bc = band_counts(df)
        fig = px.pie(values=bc.values, names=bc.index, hole=0.58,
                     color=bc.index, color_discrete_map=BAND_COLORS)
        # inside labels (plotly auto-drops those that don't fit), so the tiny
        # A/F/G slices no longer collide; the legend names every band.
        fig.update_traces(textposition="inside", textinfo="percent",
                          insidetextorientation="horizontal",
                          marker=dict(line=dict(color="#FFFFFF", width=1.5)),
                          hovertemplate="Band %{label}: %{value:,} dwellings (%{percent})<extra></extra>")
        style_fig(fig, 320); fig.update_layout(legend_title_text="EPC band")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown(f"<p style='font-size:1.08rem;border-left:2px solid {GOLD};padding-left:1rem;"
                "font-style:italic;color:var(--ink-soft)'>Oxford's stock concentrates in bands "
                "C and D; the champion recovers the band from physical fabric alone at "
                "QWK&nbsp;&asymp;&nbsp;0.77 on a post-2022 temporal hold-out.</p>",
                unsafe_allow_html=True)


# ======================================================================
# Descriptive
# ======================================================================
elif page == "Descriptive":
    masthead("Descriptive <em>analytics</em>",
             "The shape of the stock — distributions, composition and trend across the register.")
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    main_types = ["Flat", "Maisonette", "House", "Bungalow"]
    sel = st.multiselect("Property types", main_types, default=main_types)
    d = df[df["PROPERTY_TYPE"].isin(sel)] if sel else df

    a, b = st.columns(2)
    with a:
        section("01", "SAP distribution", "histogram with the statutory A–G band regions")
        fig = px.histogram(d, x="SAP", nbins=40, color_discrete_sequence=[INK],
                           labels={"SAP": "SAP score", "count": "Number of dwellings"})
        # shade + label each statutory band region (clear, self-explanatory)
        bounds = [(92, 100, "A"), (81, 92, "B"), (69, 81, "C"), (55, 69, "D"),
                  (39, 55, "E"), (21, 39, "F"), (1, 21, "G")]
        for lo, hi, lab in bounds:
            fig.add_vrect(x0=lo, x1=hi, fillcolor=BAND_COLORS[lab], opacity=0.10, line_width=0,
                          annotation_text=lab, annotation_position="top",
                          annotation_font=dict(size=11, color=INK_SOFT))
        fig.update_layout(yaxis_title="Number of dwellings", bargap=0.02, showlegend=False)
        style_fig(fig, 360)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    with b:
        section("02", "SAP by property type", "distribution shape per typology")
        fig = px.violin(d, x="PROPERTY_TYPE", y="SAP", box=True, color="PROPERTY_TYPE",
                        color_discrete_sequence=[INK, GOLD, "#5D9B57", "#CF6B33"],
                        labels={"PROPERTY_TYPE": "Property type", "SAP": "SAP score"})
        style_fig(fig, 360); fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    c, e = st.columns(2)
    with c:
        section("03", "Type × built form", "tile size = count · colour = mean SAP")
        tm = (d.groupby(["PROPERTY_TYPE", "BUILT_FORM"])
                .agg(n=("SAP", "size"), mean_sap=("SAP", "mean")).reset_index())
        fig = px.treemap(tm, path=["PROPERTY_TYPE", "BUILT_FORM"], values="n",
                         color="mean_sap", color_continuous_scale="RdYlGn",
                         range_color=[tm["mean_sap"].min(), tm["mean_sap"].max()],
                         labels={"mean_sap": "Mean SAP", "n": "Dwellings"})
        fig.update_traces(texttemplate="<b>%{label}</b><br>%{value:,} dwellings",
                          hovertemplate="%{label}<br>%{value:,} dwellings<br>Mean SAP %{color:.0f}<extra></extra>")
        style_fig(fig, 380); cbar_right(fig, "Mean SAP", rmargin=92)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    with e:
        section("04", "Floor area × SAP × CO₂", "size = CO₂ · colour = property type")
        bub = d.dropna(subset=["TOTAL_FLOOR_AREA", "CO2", "SAP"])
        bub = bub[bub["CO2"] > 0].sample(min(1500, len(bub)), random_state=42)
        fig = px.scatter(bub, x="TOTAL_FLOOR_AREA", y="SAP", size="CO2",
                         color="PROPERTY_TYPE", size_max=24, opacity=0.5,
                         color_discrete_sequence=[INK, GOLD, "#5D9B57", "#CF6B33"],
                         labels={"CO2": "CO₂ (t)", "PROPERTY_TYPE": "Property type",
                                 "TOTAL_FLOOR_AREA": "Total floor area (m²)", "SAP": "SAP score"})
        fig.update_layout(legend_title_text="Property type  ·  bubble size = CO₂ (t)")
        style_fig(fig, 380)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    section("05", "Lodgements over time", "annual certificates by band — the register growing")
    tab = pd.crosstab(d["YEAR"], d["CURRENT_ENERGY_RATING"]).reindex(columns=RATING_ORDER, fill_value=0)
    tab = tab[(tab.index >= 2008) & (tab.index <= int(df["YEAR"].max()) - 1)]
    fig = go.Figure()
    for bnd in RATING_ORDER:
        fig.add_trace(go.Scatter(x=tab.index, y=tab[bnd], stackgroup="one", name=bnd,
                                 mode="lines", line=dict(width=0.5, color=BAND_COLORS[bnd]),
                                 fillcolor=BAND_COLORS[bnd]))
    style_fig(fig, 340)
    fig.update_layout(legend_title_text="EPC band", xaxis_title="Inspection year",
                      yaxis_title="Certificates lodged")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ======================================================================
# Diagnostic
# ======================================================================
elif page == "Diagnostic":
    masthead("Diagnostic <em>analytics</em>",
             "Why the bands fall as they do — efficiency by typology and by fabric size.")
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    if prop:
        pe = prop["property_type_efficiency"]
        section("01", "Mean SAP by property type", f"most efficient typology: {pe['most_efficient_type']}")
        bt = pd.DataFrame(pe["by_type"])
        fig = px.bar(bt.sort_values("mean_sap"), x="mean_sap", y="property_type",
                     orientation="h", color="mean_sap", color_continuous_scale="RdYlGn",
                     text="modal_band", labels={"mean_sap": "Mean SAP score",
                     "property_type": "Property type", "modal_band": "Modal band"})
        # colour duplicates the x-axis here, so the colourbar is redundant — drop it.
        # the band letter at each bar end is the modal EPC band for that type.
        fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color=INK))
        style_fig(fig, 320); fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(range=[0, bt["mean_sap"].max() * 1.12])
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    section("02", "SAP vs floor area", "each point a dwelling, coloured by EPC band")
    s = df.dropna(subset=["TOTAL_FLOOR_AREA", "SAP"]).sample(min(2000, len(df)), random_state=1)
    fig = px.scatter(s, x="TOTAL_FLOOR_AREA", y="SAP", color="CURRENT_ENERGY_RATING",
                     category_orders={"CURRENT_ENERGY_RATING": RATING_ORDER},
                     color_discrete_map=BAND_COLORS, opacity=0.55,
                     labels={"TOTAL_FLOOR_AREA": "Total floor area (m²)", "SAP": "SAP score",
                             "CURRENT_ENERGY_RATING": "EPC band"})
    style_fig(fig, 380); fig.update_layout(legend_title_text="EPC band")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    tl = prop.get("timeline_table_mean_sap_by_type_era") if prop else None
    if tl:
        section("03", "Efficiency by construction era",
                "mean SAP per property type across build eras — newer fabric scores higher")
        tdf = pd.DataFrame(tl)
        types = [c for c in tdf.columns if c != "era"]
        long = (tdf.melt(id_vars="era", value_vars=types, var_name="Property type",
                         value_name="Mean SAP").dropna(subset=["Mean SAP"]))
        # one era label ("1996-06") is date-parseable, which makes Plotly switch
        # the whole x-axis to datetime and collapse every point — force category.
        long["era"] = long["era"].astype(str)
        fig = px.line(long, x="era", y="Mean SAP", color="Property type", markers=True,
                      color_discrete_sequence=[INK, GOLD, "#5D9B57", "#CF6B33", "#6E7B8B"],
                      category_orders={"era": [str(e) for e in tdf["era"].tolist()]},
                      labels={"era": "Construction era", "Mean SAP": "Mean SAP score"})
        fig.update_xaxes(type="category")
        style_fig(fig, 360); fig.update_layout(legend_title_text="Property type")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    section("04", "How physical factors relate",
            "Spearman correlation among the key numeric drivers (ρ from −1 to +1)")
    ncols = ["CURRENT_ENERGY_EFFICIENCY", "TOTAL_FLOOR_AREA", "NUMBER_HABITABLE_ROOMS",
             "MULTI_GLAZE_PROPORTION", "CO2_EMISSIONS_CURRENT", "ENERGY_CONSUMPTION_CURRENT",
             "LOW_ENERGY_LIGHTING"]
    nice = {"CURRENT_ENERGY_EFFICIENCY": "SAP", "TOTAL_FLOOR_AREA": "Floor area",
            "NUMBER_HABITABLE_ROOMS": "Rooms", "MULTI_GLAZE_PROPORTION": "Glazing",
            "CO2_EMISSIONS_CURRENT": "CO₂", "ENERGY_CONSUMPTION_CURRENT": "Energy use",
            "LOW_ENERGY_LIGHTING": "LE lighting"}
    present = [c for c in ncols if c in df.columns]
    corr = df[present].apply(pd.to_numeric, errors="coerce").corr(method="spearman").round(2)
    corr.index = [nice[c] for c in corr.index]; corr.columns = [nice[c] for c in corr.columns]
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r",
                    labels=dict(x="", y="", color="Spearman ρ"))
    style_fig(fig, 460); cbar_right(fig, "Spearman ρ", rmargin=96)
    # rotate the variable labels so the 7 categories never overlap
    fig.update_xaxes(tickangle=-45, title_text="", automargin=True, tickfont=dict(size=11))
    fig.update_yaxes(title_text="", automargin=True, tickfont=dict(size=11))
    fig.update_layout(margin_b=90)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ======================================================================
# Predictive model
# ======================================================================
elif page == "Predictive model":
    masthead("The <em>champion</em>",
             "A SAP-stratified gradient-boosted regressor, scored on a strict post-2022 hold-out.")
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    ov = rob.get("overall", {})
    kpis([
        ("QWK", f"{ov.get('qwk', float('nan')):.4f}", "primary metric"),
        ("Accuracy", f"{ov.get('accuracy', float('nan')):.3f}", "exact band"),
        ("Macro-F1", f"{ov.get('macro_f1', float('nan')):.3f}", "class-balanced"),
        ("MAE", f"{ov.get('mae_band_units', float('nan')):.3f}", "band units"),
    ])
    ci = rob.get("per_class_f1_bootstrap_ci", {})
    if ci:
        section("01", "Per-band F1", "with 95% bootstrap confidence intervals")
        rows = [{"band": b, "F1": ci[b]["mean"], "lo": ci[b]["ci_low_2.5"],
                 "hi": ci[b]["ci_high_97.5"]} for b in RATING_ORDER if b in ci]
        f = pd.DataFrame(rows)
        fig = px.bar(f, x="band", y="F1", color="band", color_discrete_map=BAND_COLORS,
                     error_y=f["hi"] - f["F1"], error_y_minus=f["F1"] - f["lo"],
                     labels={"band": "EPC band", "F1": "F1 score"})
        style_fig(fig, 360); fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    if not preds.empty:
        section("02", "Confusion matrix", "actual vs predicted band on the hold-out")
        te = preds[preds["split"] == "test"]
        cm = pd.crosstab(te["actual_band"], te["predicted_band"]).reindex(
            index=RATING_ORDER, columns=RATING_ORDER, fill_value=0)
        fig = px.imshow(cm, text_auto=True,
                        labels=dict(x="Predicted band", y="Actual band", color="Dwellings"),
                        color_continuous_scale=[[0, PAPER], [0.5, GOLD], [1, INK]])
        fig.update_xaxes(side="bottom"); style_fig(fig, 430); cbar_right(fig, "Dwellings", rmargin=92)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    pi = exp.get("permutation_importance", {})
    if pi.get("top_20"):
        section("03", "What drives the prediction", "permutation importance — drop in QWK when a "
                "feature is randomly shuffled (top 12)")
        imp = pd.DataFrame(pi["top_20"]).head(12).sort_values("importance_mean")
        fig = px.bar(imp, x="importance_mean", y="feature", orientation="h",
                     error_x="importance_std", color="importance_mean",
                     color_continuous_scale=[[0, "#9DBA59"], [1, INK]],
                     labels={"importance_mean": "Importance (QWK drop when shuffled)",
                             "feature": "Feature"})
        style_fig(fig, 440); fig.update_layout(coloraxis_showscale=False)
        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ======================================================================
# Band predictor
# ======================================================================
elif page == "Band predictor":
    masthead("Band <em>predictor</em>",
             "Describe a dwelling's fabric; the frozen champion returns a predicted SAP and "
             "band for every property type. Property type is decided by the model, not the user.")
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    @st.cache_resource(show_spinner=False)
    def load_champion():
        return pickle.load(open(ROOT / "artefacts" / "champion.pkl", "rb"))

    @st.cache_data(show_spinner=False)
    def base_frame():
        """A real, schema-valid cleaned training frame (correct dtypes for all
        features) — predictions are built by copying a real row and overriding
        only the user-facing fields, which avoids dtype clashes."""
        import sys
        sys.path.insert(0, str(ROOT))
        from src.data import (load_certificates, filter_oxford, coerce_numeric,
                              validate_consistency, cap_outliers, drop_fully_missing,
                              group_temporal_split, to_model_matrix)
        from src.features import engineer
        raw = load_certificates(ROOT / "certificates.csv")
        eng = (raw.pipe(filter_oxford).pipe(coerce_numeric, cols=[
                "CURRENT_ENERGY_EFFICIENCY", "TOTAL_FLOOR_AREA", "NUMBER_HABITABLE_ROOMS",
                "NUMBER_HEATED_ROOMS", "CO2_EMISSIONS_CURRENT", "ENERGY_CONSUMPTION_CURRENT",
                "REPORT_TYPE"]).pipe(validate_consistency).pipe(cap_outliers)
               .pipe(drop_fully_missing, threshold=0.999)
               .dropna(subset=["CURRENT_ENERGY_RATING"])
               .loc[lambda d: d["CURRENT_ENERGY_RATING"].isin(RATING_ORDER)].reset_index(drop=True))
        eng = engineer(eng)
        tr, _ = group_temporal_split(eng)
        X, _, _ = to_model_matrix(eng, missingness_ref=tr)
        cols = [s["name"] for s in art["feature_schema"]]
        Xc = X.loc[tr, [c for c in cols if c in X.columns]].reset_index(drop=True)
        for c in Xc.columns:
            if Xc[c].dtype == object:
                Xc[c] = Xc[c].astype(object).where(Xc[c].notna(), "__MISSING__")
        return Xc, cols, eng

    try:
        champ = load_champion()
        Xc, cols, eng = base_frame()
        base_idx = (Xc["CONSTRUCTION_AGE_NUM"] - Xc["CONSTRUCTION_AGE_NUM"].median()).abs().idxmin()
        section("01", "Describe the dwelling", "fabric inputs")
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Construction year", 1850, 2024, 1950)
        walls = c2.selectbox("Wall construction",
                             ["solid brick", "cavity (uninsulated)", "cavity (insulated)", "stone"])
        fuel = c3.selectbox("Main fuel", sorted(eng["MAIN_FUEL"].dropna().unique().tolist())[:8])
        area = c1.slider("Floor area (m²)", 20, 300, 90)
        glaze = c2.slider("Multi-glaze proportion", 0.0, 1.0, 1.0, 0.1)
        if st.button("Run the champion for all property types", type="primary"):
            from src.models import sap_score_to_band
            results = []
            for ptype in ["House", "Flat", "Bungalow", "Maisonette"]:
                Xrow = Xc.iloc[[base_idx]].copy().reset_index(drop=True)
                overrides = {"CONSTRUCTION_AGE_NUM": age, "TOTAL_FLOOR_AREA": area,
                             "MULTI_GLAZE_PROPORTION": glaze, "PROPERTY_TYPE": ptype,
                             "MAIN_FUEL": fuel, "FLAG_SOLID_BRICK_BARE": int("solid" in walls),
                             "FLAG_CAVITY_INSULATED": int("insulated" in walls),
                             "LOG_FLOOR_AREA": float(np.log1p(area))}
                for k, v in overrides.items():
                    if k in Xrow.columns:
                        Xrow.loc[0, k] = v
                rt = pd.Series([100], dtype="Int64")
                score = float(champ.predict_score(Xrow, rt)[0])
                results.append({"Property type": ptype, "Predicted SAP": round(score, 1),
                                "Predicted band": sap_score_to_band(np.array([score]))[0]})
            res = pd.DataFrame(results)
            section("02", "Predicted bands", "one run of the frozen champion per typology")
            kpis([(r["Property type"], r["Predicted band"], f"SAP {r['Predicted SAP']}") for r in results])
            fig = px.bar(res, x="Property type", y="Predicted SAP", color="Predicted band",
                         color_discrete_map=BAND_COLORS, text="Predicted band")
            # outside labels need headroom or they clip at the top of the card
            fig.update_traces(textposition="outside", cliponaxis=False)
            style_fig(fig, 360); fig.update_layout(showlegend=False)
            fig.update_yaxes(range=[0, max(res["Predicted SAP"]) * 1.18])
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    except Exception as ex:
        st.error(f"Predictor unavailable ({type(ex).__name__}: {ex}). "
                 "Ensure artefacts/champion.pkl exists (run scripts/freeze_champion.py).")


# ======================================================================
# Clustering
# ======================================================================
elif page == "Clustering":
    masthead("Unsupervised <em>segmentation</em>",
             "K-Prototypes over mixed fabric features — distinct dwelling archetypes in the stock.")
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    if clu:
        idx = clu.get("indices_at_k", {})
        kpis([
            ("Clusters (k)", f"{clu.get('k_selected', '—')}", "selected by sweep"),
            ("Silhouette", f"{idx.get('silhouette', float('nan')):.3f}"
                if idx.get("silhouette") is not None else "—", "cohesion/separation"),
            ("Davies–Bouldin", f"{idx.get('davies_bouldin', float('nan')):.3f}"
                if idx.get("davies_bouldin") is not None else "—", "lower is better"),
        ])
        section("01", "Segment efficiency", "mean SAP per archetype")
        seg = pd.DataFrame(clu["segments"])
        seg2 = seg.sort_values("mean_sap").copy()
        seg2["seg_label"] = seg2["modal_property_type"] + " · " + seg2["modal_band"].astype(str)
        fig = px.bar(seg2, x="mean_sap", y="seg_label",
                     orientation="h", color="mean_sap", color_continuous_scale="RdYlGn",
                     text="modal_band",
                     labels={"mean_sap": "Mean SAP score", "seg_label": "Segment (modal type · band)"},
                     hover_data={"pct": True, "mean_construction_year": True, "modal_wall_type": True,
                                 "seg_label": False})
        fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color=INK))
        style_fig(fig, 340); fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(range=[0, seg2["mean_sap"].max() * 1.12])
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.dataframe(seg, use_container_width=True)


# ======================================================================
# Recommender
# ======================================================================
elif page == "Recommender":
    masthead("Retrofit <em>recommender</em>",
             "Which measures to suggest — content-based kNN and Apriori association rules over "
             "the recommendation register.")
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    if rec:
        knn = rec.get("content_based_knn", {}); base = rec.get("popularity_baseline", {})
        kpis([
            ("kNN hit-rate", f"{knn.get('hit_rate', float('nan')):.3f}"
                if knn.get("hit_rate") is not None else "—", "at least one hit"),
            ("Precision@5", f"{knn.get('precision_at_5', float('nan')):.3f}"
                if knn.get("precision_at_5") is not None else "—", "top-5 relevance"),
            ("Lift vs popularity", f"+{base.get('knn_lift_precision_at_5', 0)}", "beats the baseline"),
        ])
        section("01", "Most-recommended measures", "count, shaded by median cost")
        cm = pd.DataFrame(rec["cost_effectiveness_top_measures"])
        fig = px.bar(cm.sort_values("count"), x="count", y="measure", orientation="h",
                     color="median_cost_gbp", color_continuous_scale=[[0, "#5D9B57"], [1, INK]],
                     labels={"count": "Times recommended", "measure": "Retrofit measure",
                             "median_cost_gbp": "Median cost (£)"})
        style_fig(fig, 460); cbar_right(fig, "Median cost (£)", rmargin=104)
        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        section("02", "Top association rules", "Apriori, ranked by lift")
        st.dataframe(pd.DataFrame(rec["association_rules"]["top_rules"]), use_container_width=True)

import os
import json
import base64
from dotenv import load_dotenv
load_dotenv('env')  # loads 'env' file from the project root into os.environ

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from city_coordinates import CITY_COORDINATES

from simulation import (
    calculate_risk,
    dashboard_metrics,
    generate_recommendation,
    top_risky_cities,
    predict_next_year
)
from route_planner import (
    geocode_place,
    get_route_alternatives,
    rank_routes
)

# --------------------------------------------------
# THEME STATE  — driven by URL query param set by the nav JS button
# --------------------------------------------------

_qp_theme  = st.query_params.get("theme", "light")
_is_light  = _qp_theme == "light"
chart_text = "#334155" if _is_light else "#94a3b8"
chart_bg   = "rgba(0,0,0,0)"

# Route planner search params — set by the JS form widget when user clicks Find Route.
# Coordinates are already resolved in JS (via GPS or Places Autocomplete).
_qp_olat  = st.query_params.get("o_lat")
_qp_olon  = st.query_params.get("o_lon")
_qp_dlat  = st.query_params.get("d_lat")
_qp_dlon  = st.query_params.get("d_lon")
_qp_oname = st.query_params.get("o_name", "Origin")
_qp_dname = st.query_params.get("d_name", "Destination")
_qp_sw    = st.query_params.get("sw", "50")      # safety weight 0-100
_qp_lit   = st.query_params.get("lit", "1")      # include lighting 1/0
_qp_rsearch = st.query_params.get("route_search", "0")  # "1" = triggered
_qp_user_lat = st.query_params.get("user_lat")
_qp_user_lon = st.query_params.get("user_lon")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

def _load_logo_b64():
    for fname in ("logo.png", "assets/logo.png"):
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None

_LOGO_B64 = _load_logo_b64()
_LOGO_URI = f"data:image/png;base64,{_LOGO_B64}" if _LOGO_B64 else None

st.set_page_config(
    page_title="SHEild",
    page_icon=_LOGO_URI if _LOGO_URI else "🛡️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS  — theme variables + sticky nav + transitions
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ═══════════════════════════════════════════
   DARK THEME — default  (:root)
   ═══════════════════════════════════════════ */
:root {
    --bg-start:         #241019;
    --bg-end:           #0f1f18;
    --text-primary:     #fdf2f7;
    --text-secondary:   #dcb3c6;
    --chart-text:       #dcb3c6;
    --nav-bg:           rgba(36, 16, 25, 0.97);
    --nav-border:       rgba(236, 72, 153, 0.25);
    --nav-link:         #e8a9c8;
    --nav-link-hover:   rgba(236, 72, 153, 0.14);
    --card-bg:          rgba(236, 72, 153, 0.08);
    --card-border:      rgba(236, 72, 153, 0.22);
    --sidebar-start:    #2b1420;
    --sidebar-end:      #0f1f18;
    --toggle-bg:        rgba(236, 72, 153, 0.12);
    --toggle-border:    rgba(236, 72, 153, 0.30);
    --toggle-color:     #f3aecd;
    --divider-color:    rgba(236, 72, 153, 0.22);
    --accent-pink:      #ec4899;
    --accent-pink-soft: #f9a8d4;
    --accent-green:     #10b981;
    --accent-green-soft:#6ee7b7;
}

/* ═══════════════════════════════════════════
   LIGHT THEME  (:root.light-theme)  — baby pink & green pastel
   ═══════════════════════════════════════════ */
:root.light-theme {
    --bg-start:         #fff2f8;
    --bg-end:           #f0fdf6;
    --text-primary:     #3f2530;
    --text-secondary:   #8a5b71;
    --chart-text:       #4b3140;
    --nav-bg:           rgba(255, 246, 250, 0.97);
    --nav-border:       rgba(236, 72, 153, 0.18);
    --nav-link:         #a15d7c;
    --nav-link-hover:   rgba(236, 72, 153, 0.10);
    --card-bg:          rgba(236, 72, 153, 0.05);
    --card-border:      rgba(236, 72, 153, 0.16);
    --sidebar-start:    #ffe4f0;
    --sidebar-end:      #e6faf0;
    --toggle-bg:        rgba(236, 72, 153, 0.09);
    --toggle-border:    rgba(236, 72, 153, 0.22);
    --toggle-color:     #ec4899;
    --divider-color:    rgba(236, 72, 153, 0.18);
    --accent-pink:      #ec4899;
    --accent-pink-soft: #f9a8d4;
    --accent-green:     #10b981;
    --accent-green-soft:#6ee7b7;
}

/* ═══════════════════════════════════════════
   SMOOTH TRANSITION ON EVERY ELEMENT
   ═══════════════════════════════════════════ */
*, *::before, *::after {
    transition:
        background      0.50s ease,
        background-color 0.50s ease,
        color           0.45s ease,
        border-color    0.45s ease,
        box-shadow      0.45s ease !important;
}

/* ── globals ── */
html { scroll-behavior: smooth; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── hide Streamlit's default menu/toolbar/footer, but keep the header
      container itself so the "reopen sidebar" arrow still works ── */
[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}
[data-testid="stToolbar"] { visibility: hidden !important; }
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }

/* Streamlit's native collapse/reopen arrow lives inside [data-testid="stHeader"],
   which creates its own stacking context — so no z-index on the arrow itself can
   raise it above the fixed emergency/nav bars declared outside that header. We
   hide it and instead render our own always-on-top toggle button (see the
   sheild-sidebar-toggle injection below) that proxies clicks to it. */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Our custom, always-reachable sidebar toggle button */
#sheild-sidebar-toggle {
    position: fixed;
    top: 100px;
    left: 14px;
    z-index: 1000003;
    width: 40px;
    height: 40px;
    border-radius: 10px;
    border: 1px solid var(--card-border, rgba(236,72,153,0.35));
    background: var(--nav-bg, rgba(20,10,15,0.85));
    color: var(--text-primary, #fff);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    font-size: 1.05rem;
    font-family: 'Inter', sans-serif;
}
#sheild-sidebar-toggle:hover {
    background: var(--nav-link-hover, rgba(236,72,153,0.2));
    border-color: rgba(236, 72, 153, 0.55);
}

/* ── main background ── */
.stApp,
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, var(--bg-start) 0%, var(--bg-end) 100%) !important;
}
.main { background: transparent !important; }

/* ── text colours ── */
h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; }
p, li, label, .stMarkdown p, .stMarkdown li { color: var(--text-primary) !important; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--sidebar-start) 0%, var(--sidebar-end) 100%) !important;
    top: 90px !important;
    height: calc(100vh - 90px) !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span { color: var(--text-primary) !important; }

/* ── metric cards ── */
div[data-testid="metric-container"] {
    background:    var(--card-bg)     !important;
    border: 1px solid var(--card-border) !important;
    padding: 18px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] div { color: var(--text-secondary) !important; }

/* ── dividers ── */
hr { border-color: var(--divider-color) !important; }

/* ══════════════════════════════════════════════
   PLOTLY CHART TEXT — theme-aware
   Works whether charts are in main DOM or same-origin iframes.
   The JS block also injects a stylesheet into every iframe.
   ══════════════════════════════════════════════ */
.js-plotly-plot text,
.svg-container text,
.plotly text {
    fill: var(--chart-text) !important;
    transition: fill 0.45s ease !important;
}

/* ══════════════════════════════════════════════
   DROPDOWN / SELECT / POPOVER — text visibility
   ══════════════════════════════════════════════ */

/* Select trigger box */
[data-baseweb="select"] > div {
    background: var(--card-bg)     !important;
    border-color: var(--card-border) !important;
    color: var(--text-primary)     !important;
}
/* Selected value text */
[data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"],
[data-baseweb="select"] span,
[data-baseweb="select"] input {
    color: var(--text-primary) !important;
}
/* Dropdown popover container */
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: var(--sidebar-start) !important;
    border: 1px solid var(--card-border) !important;
}
/* Individual option rows */
[data-baseweb="menu"] [role="option"],
[data-baseweb="menu"] li,
[data-baseweb="menu"] li span,
[data-baseweb="list"] li,
[data-baseweb="list"] li span {
    background: transparent !important;
    color: var(--text-primary) !important;
}
/* Hover state for option rows */
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="list"] li:hover {
    background: var(--nav-link-hover) !important;
    color: var(--text-primary) !important;
}
/* Highlighted / selected option */
[data-baseweb="menu"] [aria-selected="true"],
[data-baseweb="menu"] [data-highlighted="true"] {
    background: rgba(236, 72, 153, 0.12) !important;
    color: #ec4899 !important;
}
/* Multiselect tags */
[data-baseweb="tag"] {
    background: rgba(236, 72, 153, 0.15) !important;
    color: var(--text-primary) !important;
}
/* Radio buttons */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label,
[data-testid="stToggle"] label { color: var(--text-primary) !important; }

/* ═══════════════════════════════════════════
   STICKY NAVIGATION BAR
   ═══════════════════════════════════════════ */
.sheild-nav {
    position: fixed;
    top: 34px; left: 0; right: 0;
    z-index: 999999;
    background:    var(--nav-bg)     !important;
    border-bottom: 1px solid var(--nav-border) !important;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    display: flex;
    align-items: center;
    gap: 2px;
    padding: 0 20px;
    height: 56px;
    box-sizing: border-box;
    box-shadow: 0 4px 32px rgba(0, 0, 0, 0.5);
}

.sheild-nav-brand {
    font-size: 1rem;
    font-weight: 700;
    color: #ec4899 !important;
    letter-spacing: -0.3px;
    padding-right: 20px;
    border-right: 1px solid var(--nav-border);
    margin-right: 8px;
    white-space: nowrap;
    flex-shrink: 0;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}
.sheild-nav-brand img {
    height: 28px;
    width: auto;
    border-radius: 6px;
}

.sheild-nav a {
    color: var(--nav-link) !important;
    text-decoration: none;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 8px 16px;
    border-radius: 8px;
    white-space: nowrap;
    border-bottom: 2px solid transparent;
    display: inline-block;
}
.sheild-nav a:hover {
    color: var(--text-primary) !important;
    background: var(--nav-link-hover) !important;
    border-bottom-color: rgba(236, 72, 153, 0.45);
}

/* Theme toggle button — injected into nav by JS */
#theme-toggle-btn {
    margin-left: auto;
    background: var(--toggle-bg);
    border: 1px solid var(--toggle-border);
    color: var(--toggle-color);
    width: 38px; height: 38px;
    border-radius: 10px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    padding: 0;
}
#theme-toggle-btn:hover {
    background: var(--nav-link-hover);
    border-color: rgba(236, 72, 153, 0.4);
    color: var(--text-primary);
}
#theme-toggle-btn svg { width: 17px; height: 17px; stroke: currentColor; }

/* ── section anchors ── */
.section-anchor {
    display: block;
    height: 0;
    visibility: hidden;
    scroll-margin-top: 112px;
}

/* ── push content below emergency bar + nav ── */
.block-container {
    padding-top:    112px !important;
    padding-bottom: 40px !important;
}

/* ═══════════════════════════════════════════
   EMERGENCY HELPLINE BAR — always on top, always visible
   ═══════════════════════════════════════════ */
.sheild-emergency {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 1000000;
    height: 34px;
    background: linear-gradient(90deg, var(--accent-pink) 0%, var(--accent-green) 100%);
    display: flex;
    align-items: center;
    gap: 22px;
    padding: 0 18px;
    overflow-x: auto;
    white-space: nowrap;
    box-shadow: 0 2px 10px rgba(236,72,153,0.25);
}
.sheild-emergency::-webkit-scrollbar { height: 3px; }
.sheild-emergency-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.76rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.2px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.15);
    flex-shrink: 0;
}
.sheild-emergency-item b {
    background: rgba(255,255,255,0.28);
    padding: 1px 8px;
    border-radius: 20px;
    font-weight: 800;
}
.sheild-emergency-sep {
    color: rgba(255,255,255,0.55);
    flex-shrink: 0;
}

/* ── badge styles ── */
.risk-badge-high {
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white; padding: 4px 12px; border-radius: 20px;
    font-weight: 600; font-size: 0.8rem;
}
.risk-badge-medium {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white; padding: 4px 12px; border-radius: 20px;
    font-weight: 600; font-size: 0.8rem;
}
.risk-badge-low {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white; padding: 4px 12px; border-radius: 20px;
    font-weight: 600; font-size: 0.8rem;
}
.ut-card {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px; padding: 16px; margin: 8px 0;
}

/* ── primary buttons — brand pink→green gradient (e.g. "Find Route") ── */
button[kind="primary"], button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg, var(--accent-pink), var(--accent-green)) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(236,72,153,0.30) !important;
}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
    filter: brightness(1.06);
    box-shadow: 0 6px 22px rgba(16,185,129,0.35) !important;
}
/* ── secondary buttons — soft pink outline ── */
button[kind="secondary"] {
    border-color: var(--accent-pink) !important;
    color: var(--accent-pink) !important;
}
/* ── tabs — pink underline for active tab ── */
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-pink) !important;
    border-bottom-color: var(--accent-pink) !important;
}
[data-baseweb="tab-highlight"] {
    background-color: var(--accent-pink) !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# STICKY NAVIGATION HTML
# --------------------------------------------------

_logo_img_tag = f'<img src="{_LOGO_URI}" alt="SHEild" />' if _LOGO_URI else "🛡️"

st.markdown(f"""
<style>
.sheild-emergency-item a {{ color: #ffffff; text-decoration: none; }}
.sheild-emergency-item a:hover {{ text-decoration: underline; }}
</style>
<div class="sheild-emergency" id="sheild-emergency">
  <span class="sheild-emergency-item">🚓 Police <a href="tel:100"><b>100</b></a> / <a href="tel:112"><b>112</b></a></span>
  <span class="sheild-emergency-sep">•</span>
  <span class="sheild-emergency-item">👩 Women Helpline <a href="tel:1091"><b>1091</b></a></span>
  <span class="sheild-emergency-sep">•</span>
  <span class="sheild-emergency-item">🧒 Child Helpline <a href="tel:1098"><b>1098</b></a></span>
  <span class="sheild-emergency-sep">•</span>
  <span class="sheild-emergency-item">🏠 Domestic Abuse <a href="tel:181"><b>181</b></a></span>
  <span class="sheild-emergency-sep">•</span>
  <span class="sheild-emergency-item">🚑 Ambulance <a href="tel:108"><b>108</b></a> / <a href="tel:102"><b>102</b></a></span>
  <span class="sheild-emergency-sep">•</span>
  <span class="sheild-emergency-item">🔥 Fire <a href="tel:101"><b>101</b></a></span>
  <span class="sheild-emergency-sep">•</span>
  <span class="sheild-emergency-item">🧭 Disaster Mgmt <a href="tel:1078"><b>1078</b></a></span>
  <span class="sheild-emergency-sep">•</span>
  <span class="sheild-emergency-item">📞 Nat'l Emergency <a href="tel:112"><b>112</b></a></span>
</div>
<nav class="sheild-nav" id="sheild-nav">
  <span class="sheild-nav-brand">{_logo_img_tag} SHEild</span>
  <span style="color:var(--nav-link); font-size:0.82rem;">Women &amp; Children Safety Intelligence</span>
</nav>
""", unsafe_allow_html=True)

# --------------------------------------------------
# THEME TOGGLE BUTTON
# Runs inside a sandboxed iframe (st.components.v1.html).
# Uses window.parent to access and modify the Streamlit
# app's top-level document — works on localhost (same origin).
# A MutationObserver re-injects the button if Streamlit
# re-renders the nav on widget interaction.
# --------------------------------------------------

# Determine theme string for JS injection
_theme_js = "light" if _is_light else "dark"

components.html(f"""
<script>
(function () {{
    var SUN  = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';
    var MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

    /* ── Apply Python-driven theme class immediately on every render ── */
    var pd = window.parent.document;
    if ("{_theme_js}" === "light") {{
        pd.documentElement.classList.add("light-theme");
    }} else {{
        pd.documentElement.classList.remove("light-theme");
    }}

    /* ─────────────────────────────────────────────
       applyChartFonts — injects a <style> into every same-origin
       Plotly iframe so SVG text elements get the right fill color.
       Falls back silently for any cross-origin frame.
       Also updates any Plotly text living directly in the main DOM.
    ───────────────────────────────────────────── */
    function applyChartFonts(isLight) {{
        var col = isLight ? '#1e293b' : '#94a3b8';
        var css = 'text{{fill:' + col + '!important;transition:fill 0.45s ease!important}}';

        /* 1. Main-DOM Plotly elements */
        pd.querySelectorAll('.js-plotly-plot text, .svg-container text, .plotly text').forEach(function(el) {{
            el.style.setProperty('fill', col, 'important');
        }});

        /* 2. Any same-origin iframe that contains Plotly SVG */
        pd.querySelectorAll('iframe').forEach(function(f) {{
            try {{
                var doc = f.contentDocument;
                if (!doc || !doc.body) return;
                if (!doc.querySelector('text')) return;   /* not a chart iframe */
                /* Inject / update a persistent <style> tag */
                var s = doc.getElementById('_sheild_ct');
                if (!s) {{
                    s = doc.createElement('style');
                    s.id = '_sheild_ct';
                    (doc.head || doc.documentElement).appendChild(s);
                }}
                s.textContent = css;
                /* Also forcibly set fill on every text node that exists now */
                doc.querySelectorAll('text').forEach(function(t) {{
                    t.style.setProperty('fill', col, 'important');
                }});
            }} catch(e) {{ /* cross-origin — ignore */ }}
        }});
    }}

    /* Run immediately, then keep re-applying every second so new charts pick it up */
    applyChartFonts("{_theme_js}" === "light");
    setInterval(function () {{
        applyChartFonts(pd.documentElement.classList.contains('light-theme'));
    }}, 1000);

    function setBtn(btn, isLight) {{
        btn.innerHTML = isLight ? MOON : SUN;
        btn.title = isLight ? 'Switch to dark mode' : 'Switch to light mode';
        btn.setAttribute('aria-label', btn.title);
    }}

    function inject() {{
        var nav = pd.getElementById('sheild-nav');
        if (!nav) {{ setTimeout(inject, 200); return; }}
        if (pd.getElementById('theme-toggle-btn')) return;

        var isLight = pd.documentElement.classList.contains('light-theme');

        var btn = pd.createElement('button');
        btn.id = 'theme-toggle-btn';
        setBtn(btn, isLight);

        btn.addEventListener('click', function () {{
            var light = pd.documentElement.classList.toggle('light-theme');
            setBtn(btn, light);
            /* Apply chart fonts immediately — no page reload needed */
            applyChartFonts(light);
        }});

        nav.appendChild(btn);
    }}

    inject();

    /* Re-inject if Streamlit re-renders and removes the button */
    var observer = new pd.defaultView.MutationObserver(function () {{
        if (pd.getElementById('sheild-nav') && !pd.getElementById('theme-toggle-btn')) {{
            inject();
        }}
    }});
    observer.observe(pd.body, {{ childList: true, subtree: false }});
}})();
</script>
""", height=0)

# --------------------------------------------------
# SIDEBAR OPEN/CLOSE BUTTON
# Streamlit's own collapse/reopen arrow is unreachable here: it renders
# inside [data-testid="stHeader"], which forms its own stacking context,
# so the fixed emergency bar + nav (declared outside that header) always
# paint over it no matter what z-index it's given. We hide the native
# arrow (see CSS above) and inject our own fixed button into the parent
# document instead. Clicking it finds whichever native control actually
# exists right now (the "open" arrow when collapsed, or the collapse
# icon inside the sidebar when expanded) and clicks it programmatically —
# so we get Streamlit's real show/hide behaviour, just via a button that
# is guaranteed to stay visible and clickable.
# --------------------------------------------------

components.html("""
<script>
(function () {
    var pd = window.parent.document;

    function findNativeToggle() {
        // Reopen arrow (shown when sidebar is collapsed)
        var el = pd.querySelector('[data-testid="collapsedControl"] button')
              || pd.querySelector('[data-testid="collapsedControl"]')
              || pd.querySelector('[data-testid="stSidebarCollapsedControl"] button')
              || pd.querySelector('[data-testid="stSidebarCollapsedControl"]');
        if (el) return el;

        // Collapse control (shown inside the sidebar when it's expanded)
        el = pd.querySelector('[data-testid="stSidebarCollapseButton"] button')
          || pd.querySelector('[data-testid="stSidebarCollapseButton"]')
          || pd.querySelector('[data-testid="stSidebar"] button[kind="header"]');
        return el;
    }

    function isSidebarOpen() {
        var sb = pd.querySelector('[data-testid="stSidebar"]');
        if (!sb) return false;
        var aria = sb.getAttribute('aria-expanded');
        if (aria !== null) return aria === 'true';
        return sb.offsetWidth > 20;
    }

    function updateIcon(btn) {
        btn.textContent = isSidebarOpen() ? '\u00AB' : '\u00BB';
        btn.title = isSidebarOpen() ? 'Collapse filters panel' : 'Open filters panel';
        btn.setAttribute('aria-label', btn.title);
    }

    function inject() {
        if (pd.getElementById('sheild-sidebar-toggle')) return;

        var btn = pd.createElement('button');
        btn.id = 'sheild-sidebar-toggle';
        updateIcon(btn);

        btn.addEventListener('click', function () {
            var target = findNativeToggle();
            if (target) target.click();
            setTimeout(function () { updateIcon(btn); }, 250);
        });

        pd.body.appendChild(btn);
    }

    inject();

    // Keep the icon in sync (in case the sidebar state changes some other
    // way) and re-inject the button if Streamlit's own re-renders ever
    // strip it out.
    var syncTimer = setInterval(function () {
        var btn = pd.getElementById('sheild-sidebar-toggle');
        if (!btn) { inject(); return; }
        updateIcon(btn);
    }, 600);

    var observer = new pd.defaultView.MutationObserver(function () {
        if (!pd.getElementById('sheild-sidebar-toggle')) inject();
    });
    observer.observe(pd.body, { childList: true, subtree: false });
})();
</script>
""", height=0)

# --------------------------------------------------
# FLOATING SOS BUTTON + PANEL
# Injected directly into the parent Streamlit document (same technique
# as the theme toggle above) so it stays visible & fixed no matter which
# tab/section the user has scrolled to.
#
# What this can and can't do, by design:
#   - It CANNOT silently auto-send a message — no browser can do that
#     without a paid SMS gateway (e.g. Twilio) and a backend.
#   - It CAN grab live GPS, build a pre-filled WhatsApp/SMS message with
#     a Google Maps link to that location, and hand it to WhatsApp/the
#     Messages app already typed out — the contact just taps Send.
#   - Saved contacts persist in the browser's localStorage (client-side
#     only — nothing is sent to or stored on any server).
# --------------------------------------------------

components.html("""
<script>
(function () {
    var pd = window.parent.document;

    /* ---------- one-time styles ---------- */
    if (!pd.getElementById('_sheild_sos_style')) {
        var css = pd.createElement('style');
        css.id = '_sheild_sos_style';
        css.textContent = `
            #sheild-sos-fab {
                position: fixed;
                right: 22px;
                bottom: 26px;
                z-index: 1000002;
                width: 62px;
                height: 62px;
                border-radius: 50%;
                background: radial-gradient(circle at 30% 30%, #ff5b7a, #dc2626);
                border: none;
                color: #fff;
                font-size: 1.55rem;
                font-weight: 800;
                cursor: pointer;
                box-shadow: 0 0 0 0 rgba(220,38,38,0.55), 0 6px 18px rgba(0,0,0,0.35);
                animation: sheild-pulse 2.2s infinite;
                font-family: 'Inter', sans-serif;
            }
            @keyframes sheild-pulse {
                0%   { box-shadow: 0 0 0 0   rgba(220,38,38,0.55), 0 6px 18px rgba(0,0,0,0.35); }
                70%  { box-shadow: 0 0 0 16px rgba(220,38,38,0),   0 6px 18px rgba(0,0,0,0.35); }
                100% { box-shadow: 0 0 0 0   rgba(220,38,38,0),   0 6px 18px rgba(0,0,0,0.35); }
            }
            #sheild-sos-panel {
                position: fixed;
                right: 22px;
                bottom: 100px;
                z-index: 1000002;
                width: 300px;
                max-width: calc(100vw - 44px);
                background: var(--nav-bg);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 16px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.25);
                display: none;
                font-family: 'Inter', sans-serif;
                color: var(--text-primary);
            }
            #sheild-sos-panel.open { display: block; }
            #sheild-sos-panel h4 { margin: 0 0 10px 0; font-size: 1rem; color: var(--text-primary); }
            #sheild-sos-panel label { font-size: 0.72rem; color: var(--text-secondary); display:block; margin-top:8px; }
            #sheild-sos-panel input {
                width: 100%; box-sizing: border-box; padding: 7px 9px; margin-top: 3px;
                border-radius: 8px; border: 1px solid var(--card-border);
                background: var(--toggle-bg); color: var(--text-primary); font-size: 0.85rem;
            }
            #sheild-sos-send {
                width: 100%; margin-top: 14px; padding: 10px; border: none; border-radius: 10px;
                background: linear-gradient(135deg,#25D366,#128C7E); color: #fff; font-weight: 700;
                cursor: pointer; font-size: 0.88rem;
            }
            #sheild-sos-sms {
                width: 100%; margin-top: 8px; padding: 9px; border-radius: 10px;
                background: transparent; border: 1px solid var(--card-border); color: var(--toggle-color);
                cursor: pointer; font-size: 0.8rem;
            }
            #sheild-sos-status { font-size: 0.72rem; margin-top: 8px; min-height: 14px; }
            #sheild-sos-dial { display:flex; gap:6px; margin-top:12px; }
            #sheild-sos-dial a {
                flex: 1; text-align:center; padding: 7px 0; border-radius: 8px;
                background: var(--toggle-bg); color: var(--text-primary); text-decoration:none;
                font-size: 0.72rem; font-weight: 700;
            }
            #sheild-sos-close {
                position:absolute; top:10px; right:12px; background:none; border:none;
                color: var(--text-secondary); font-size:1rem; cursor:pointer;
            }
            #sheild-sos-note { font-size: 0.65rem; color: var(--text-secondary); margin-top: 10px; line-height: 1.3; }
        `;
        pd.head.appendChild(css);
    }

    function buildPanelHTML(name, phone) {
        return `
            <button id="sheild-sos-close">✕</button>
            <h4>🆘 Emergency SOS</h4>
            <label>Trusted contact name</label>
            <input id="sheild-sos-name" placeholder="e.g. Mom" value="${name}">
            <label>Their WhatsApp/phone number (with country code)</label>
            <input id="sheild-sos-phone" placeholder="+91XXXXXXXXXX" value="${phone}">
            <button id="sheild-sos-send">🆘 Send SOS via WhatsApp</button>
            <button id="sheild-sos-sms">💬 Send via SMS instead</button>
            <div id="sheild-sos-status"></div>
            <div id="sheild-sos-dial">
                <a href="tel:112">📞 112</a>
                <a href="tel:1091">👩 1091</a>
                <a href="tel:100">🚓 100</a>
            </div>
            <div id="sheild-sos-note">Opens WhatsApp/SMS with your live location and message already typed in —
            your contact just needs to tap Send. Number is saved only in this browser, never sent anywhere.</div>
        `;
    }

    function getSaved() {
        try {
            return {
                name: pd.defaultView.localStorage.getItem('sheild_sos_name') || '',
                phone: pd.defaultView.localStorage.getItem('sheild_sos_phone') || ''
            };
        } catch (e) { return { name: '', phone: '' }; }
    }

    function wireUp(panel) {
        var nameEl = pd.getElementById('sheild-sos-name');
        var phoneEl = pd.getElementById('sheild-sos-phone');
        var status = pd.getElementById('sheild-sos-status');

        function save() {
            try {
                pd.defaultView.localStorage.setItem('sheild_sos_name', nameEl.value);
                pd.defaultView.localStorage.setItem('sheild_sos_phone', phoneEl.value);
            } catch (e) {}
        }
        nameEl.addEventListener('input', save);
        phoneEl.addEventListener('input', save);

        pd.getElementById('sheild-sos-close').addEventListener('click', function () {
            panel.classList.remove('open');
        });

        function withLocation(cb) {
            if (!navigator.geolocation) { cb(null); return; }
            navigator.geolocation.getCurrentPosition(
                function (pos) {
                    cb('https://www.google.com/maps?q=' + pos.coords.latitude.toFixed(6) + ',' + pos.coords.longitude.toFixed(6));
                },
                function () { cb(null); },
                { enableHighAccuracy: true, timeout: 8000 }
            );
        }

        pd.getElementById('sheild-sos-send').addEventListener('click', function () {
            var phone = phoneEl.value.replace(/[^0-9]/g, '');
            if (!phone) {
                status.style.color = '#ef4444';
                status.textContent = '⚠️ Enter a contact number first.';
                return;
            }
            status.style.color = 'var(--text-secondary)';
            status.textContent = '📍 Getting your location...';
            /* Open the tab synchronously on the click so popup blockers allow it */
            var win = pd.defaultView.open('about:blank', '_blank');
            withLocation(function (mapsLink) {
                var msg = '🆘 EMERGENCY! I need help right now.' +
                    (mapsLink ? ' My live location: ' + mapsLink : ' (Location unavailable — please call me immediately.)') +
                    ' — sent via SHEild SOS';
                var url = 'https://wa.me/' + phone + '?text=' + encodeURIComponent(msg);
                if (win) { win.location.href = url; } else { pd.defaultView.location.href = url; }
                status.style.color = '#10b981';
                status.textContent = '✅ WhatsApp opened with your alert — ask them to tap Send.';
            });
        });

        pd.getElementById('sheild-sos-sms').addEventListener('click', function () {
            var phone = phoneEl.value.trim();
            if (!phone) {
                status.style.color = '#ef4444';
                status.textContent = '⚠️ Enter a contact number first.';
                return;
            }
            status.style.color = 'var(--text-secondary)';
            status.textContent = '📍 Getting your location...';
            withLocation(function (mapsLink) {
                var msg = '🆘 EMERGENCY! I need help right now.' +
                    (mapsLink ? ' My live location: ' + mapsLink : ' (Location unavailable — please call me immediately.)') +
                    ' — sent via SHEild SOS';
                var isIOS = /iPad|iPhone|iPod/.test(pd.defaultView.navigator.userAgent);
                var url = 'sms:' + phone + (isIOS ? '&' : '?') + 'body=' + encodeURIComponent(msg);
                pd.defaultView.location.href = url;
                status.style.color = '#10b981';
                status.textContent = '✅ Messages app opened with your alert pre-filled.';
            });
        });
    }

    function inject() {
        if (pd.getElementById('sheild-sos-fab')) return;

        var saved = getSaved();

        var fab = pd.createElement('button');
        fab.id = 'sheild-sos-fab';
        fab.innerHTML = 'SOS';
        fab.title = 'Emergency SOS';

        var panel = pd.createElement('div');
        panel.id = 'sheild-sos-panel';
        panel.innerHTML = buildPanelHTML(saved.name, saved.phone);

        fab.addEventListener('click', function () {
            panel.classList.toggle('open');
        });

        pd.body.appendChild(fab);
        pd.body.appendChild(panel);
        wireUp(panel);
    }

    inject();

    var observer = new pd.defaultView.MutationObserver(function () {
        if (!pd.getElementById('sheild-sos-fab')) inject();
    });
    observer.observe(pd.body, { childList: true, subtree: false });
})();
</script>
""", height=0)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

_hero_col1, _hero_col2 = st.columns([1, 6])
with _hero_col1:
    if _LOGO_URI:
        st.image(_LOGO_URI, width=84)
    else:
        st.markdown("### 🛡️")
with _hero_col2:
    st.title("SHEild — Safe Route Navigation")
    st.caption("Find the route that keeps you safest, not just the fastest one, powered by NCRB crime data. "
               "Full city-by-city safety insights are one tap away in **Explore Safety Data** below.")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    df = pd.read_csv("women_safety_dataset.csv")
except FileNotFoundError:
    st.error("Dataset not found. Ensure 'women_safety_dataset.csv' is in the project root.")
    st.stop()

REQUIRED = ["city", "state_ut", "territory_type", "district",
            "cases_2021", "cases_2022", "cases_2023",
            "population_lakhs", "crime_rate_2023", "chargesheet_rate",
            "children_cases_2023", "pocso_2023", "kidnapping_children_2023"]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"Missing columns in dataset: {missing}")
    st.stop()

df = df[df["city"].str.upper() != "TOTAL 34 CITIES"].copy()

# --------------------------------------------------
# ADD COORDINATES
# --------------------------------------------------

coords = pd.DataFrame(
    [(city, lat, lon) for city, (lat, lon) in CITY_COORDINATES.items()],
    columns=["city", "lat", "lon"]
)
df = df.merge(coords, on="city", how="left")

missing_coords = df[df["lat"].isna()]["city"].tolist()
if missing_coords:
    st.warning(f"Missing coordinates for: {missing_coords}")

# --------------------------------------------------
# CALCULATE RISK
# --------------------------------------------------

df = calculate_risk(df)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

if _LOGO_URI:
    st.sidebar.image(_LOGO_URI, width=64)
st.sidebar.title("🛡️ SHEild")
st.sidebar.caption("These filters power the tabs under **📊 Explore Safety Data** — the Safe Route Planner above works independently.")

st.sidebar.markdown("---")

territory_filter = st.sidebar.multiselect(
    "Filter by Territory Type",
    options=["State", "Union Territory"],
    default=["State", "Union Territory"]
)

df_filtered = df[df["territory_type"].isin(territory_filter)].copy()

all_uts = sorted(df_filtered["state_ut"].unique())
selected_ut = st.sidebar.selectbox("Filter by State/UT", ["All"] + all_uts)

if selected_ut != "All":
    df_display = df_filtered[df_filtered["state_ut"] == selected_ut].copy()
else:
    df_display = df_filtered.copy()

selected_city = st.sidebar.selectbox(
    "Select a City/District",
    sorted(df_display["city"].unique())
)

city_data = df[df["city"] == selected_city]

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source:** NCRB – Crime Against Women & Children in Metropolitan Cities and Union Territories (2021–2023).")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**🚨 Emergency Numbers**
- Police: **100 / 112**
- Women Helpline: **1091**
- Child Helpline: **1098**
- Domestic Abuse: **181**
- Ambulance: **108 / 102**
""")


main_tab_home, main_tab_explore = st.tabs(["🏠 Home · Safe Route", "📊 Explore Safety Data"])

with main_tab_home:
    # ==========================================================
    # SECTION 6 – SAFE ROUTE PLANNER
    # ==========================================================

    st.markdown('<div id="sec-route" class="section-anchor"></div>', unsafe_allow_html=True)

    st.header("Safe Route Planner")

    covered_cities = sorted(CITY_COORDINATES.keys())

    # ── Hidden input for GPS coordinates to communicate with Python sandbox-compatibly ──
    st.markdown("""
    <style>
    div[data-testid="stTextInput"]:has(input[aria-label="hidden_gps_coords"]) {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    hidden_gps_coords = st.text_input("hidden_gps_coords", key="hidden_gps_coords", label_visibility="collapsed")

    user_lat = None
    user_lon = None
    if hidden_gps_coords:
        try:
            import json
            gdata = json.loads(hidden_gps_coords)
            user_lat = float(gdata.get("lat"))
            user_lon = float(gdata.get("lon"))
        except Exception:
            pass

    # ── GPS Button Component ──
    # Uses React value setter to write coordinates reload-free directly back to Streamlit
    _gps_button_html = f"""
    <div style="margin-bottom:10px;">
      <button id="gps-btn" style="
          background:#ec4899; color:white; border:none; border-radius:8px;
          padding:8px 16px; font-family:Inter,sans-serif; font-size:0.85rem;
          font-weight:600; cursor:pointer; display:flex; align-items:center; gap:6px;">
          📍 Use My Current Location as Start Point
      </button>
      <div id="gps-status" style="font-family:Inter,sans-serif; font-size:0.75rem; color:#10b981; margin-top:5px; min-height:16px;"></div>
    </div>
    <script>
    document.getElementById('gps-btn').addEventListener('click', function () {{
        var btn = this, status = document.getElementById('gps-status');
        if (!navigator.geolocation) {{
            status.style.color = '#ef4444';
            status.textContent = '❌ Geolocation is not supported by this browser.';
            return;
        }}
        btn.disabled = true; btn.textContent = '⏳ Locating…';
        status.textContent = '';
        navigator.geolocation.getCurrentPosition(function (pos) {{
            var lat = pos.coords.latitude.toFixed(6);
            var lon = pos.coords.longitude.toFixed(6);
            var data = {{lat: lat, lon: lon}};
        
            var inputEl = window.parent.document.querySelector('input[aria-label="hidden_gps_coords"]');
            if (inputEl) {{
                try {{
                    var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                    nativeInputValueSetter.call(inputEl, JSON.stringify(data));
                    inputEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    status.textContent = '✅ Location captured successfully!';
                }} catch(e) {{
                    status.style.color = '#ef4444';
                    status.textContent = '⚠️ Sandbox blocked direct access. Trying fallback URL redirect...';
                    var url = new URL(window.parent.location.href);
                    url.searchParams.set('user_lat', lat);
                    url.searchParams.set('user_lon', lon);
                    window.parent.location.href = url.toString();
                }}
            }} else {{
                // Fallback if not found
                var url = new URL(window.parent.location.href);
                url.searchParams.set('user_lat', lat);
                url.searchParams.set('user_lon', lon);
                window.parent.location.href = url.toString();
            }}
            btn.disabled = false; btn.innerHTML = '📍 Use My Current Location as Start Point';
        }}, function (err) {{
            status.style.color = '#ef4444';
            status.textContent = '⚠️ ' + err.message;
            btn.disabled = false; btn.innerHTML = '📍 Use My Current Location as Start Point';
        }}, {{enableHighAccuracy: true, timeout: 10000}});
    }});
    </script>
    """
    components.html(_gps_button_html, height=55)

    # Also check query parameters as fallback
    _qp_user_lat = st.query_params.get("user_lat")
    _qp_user_lon = st.query_params.get("user_lon")
    if _qp_user_lat and _qp_user_lon:
        user_lat = float(_qp_user_lat)
        user_lon = float(_qp_user_lon)

    if user_lat and user_lon:
        st.success(
            f"📍 Using your current GPS location as the start point: "
            f"({user_lat:.4f}, {user_lon:.4f})"
        )

    route_col1, route_col2 = st.columns(2)

    with route_col1:
        st.markdown("**Start Point**")
        origin_mode = st.radio(
            "Origin input",
            ["Use GPS location", "Pick a covered city", "Enter an address"],
            label_visibility="collapsed",
            index=0 if (user_lat and user_lon) else 1,
            key="origin_mode"
        )
        origin_city_pick = None
        origin_address = None
        if origin_mode == "Pick a covered city":
            origin_city_pick = st.selectbox("From city", covered_cities, key="origin_city")
        elif origin_mode == "Enter an address":
            origin_address = st.text_input(
                "From address", placeholder="e.g. Connaught Place, Delhi", key="origin_addr"
            )

    with route_col2:
        st.markdown("**Destination**")
        dest_mode = st.radio(
            "Destination input",
            ["Pick a covered city", "Enter an address"],
            label_visibility="collapsed",
            key="dest_mode"
        )
        dest_city_pick = None
        dest_address = None
        if dest_mode == "Pick a covered city":
            dest_city_pick = st.selectbox(
                "To city", covered_cities, index=min(1, len(covered_cities) - 1), key="dest_city"
            )
        else:
            dest_address = st.text_input(
                "To address", placeholder="e.g. Gateway of India, Mumbai", key="dest_addr"
            )

    safety_weight_pct = st.slider(
        "Priority — Fastest ⟷ Safest",
        min_value=0, max_value=100, value=50, step=10,
        help=(
            "0 = shortest/fastest route regardless of crime data. "
            "100 = prioritize routes that stay furthest from higher-crime cities "
            "and favor better-lit, busier roads."
        )
    )
    safety_weight = safety_weight_pct / 100

    include_lighting = st.checkbox(
        "🔦 Also analyze street lighting & road type (via OpenStreetMap)",
        value=True,
        help=(
            "Checks whether the roads on each route are tagged as lit, and whether "
            "they're major/busy roads vs minor or isolated ones — used as a proxy for "
            "dimly-lit or empty stretches, since crime data alone can't tell us that. "
            "Only works for shorter routes (under ~60km) and needs a live connection "
            "to OpenStreetMap; uncheck for a faster, crime-data-only result."
        )
    )

    find_route_clicked = st.button("🔍 Find Route", type="primary")

    if find_route_clicked:
        # ---- Resolve origin coordinates ----
        origin_coords = None
        if origin_mode == "Use GPS location":
            if user_lat and user_lon:
                origin_coords = (user_lat, user_lon)
            else:
                st.error(
                    "No GPS location captured yet — click 'Use My Current Location as Start Point' "
                    "above and allow the browser's permission prompt first."
                )
        elif origin_mode == "Pick a covered city":
            origin_coords = CITY_COORDINATES.get(origin_city_pick)
        else:
            with st.spinner(f"Locating '{origin_address}'..."):
                origin_coords = geocode_place(origin_address)
            if origin_coords is None:
                st.error(f"Couldn't find a location for '{origin_address}'. Try a more specific address.")

        # ---- Resolve destination coordinates (biased toward origin, if we have it) ----
        dest_coords = None
        if origin_coords and dest_mode == "Enter an address":
            with st.spinner(f"Locating '{dest_address}'..."):
                dest_coords = geocode_place(dest_address, bias_coords=origin_coords)
            if dest_coords is None:
                st.error(f"Couldn't find a location for '{dest_address}'. Try a more specific address, or check the spelling.")
        elif dest_mode == "Pick a covered city":
            dest_coords = CITY_COORDINATES.get(dest_city_pick)
        else:
            with st.spinner(f"Locating '{dest_address}'..."):
                dest_coords = geocode_place(dest_address)
            if dest_coords is None:
                st.error(f"Couldn't find a location for '{dest_address}'. Try a more specific address, or check the spelling.")

        if origin_coords and dest_coords:
            with st.spinner("Fetching route alternatives from the road network..."):
                raw_routes = get_route_alternatives(origin_coords, dest_coords)

            if not raw_routes:
                st.error(
                    "Couldn't fetch routes. Check that your Google Routes API key is active "
                    "and has the Routes API enabled, or try again in a moment."
                )
                st.session_state.pop("route_results", None)
            else:
                with st.spinner("Scoring routes against NCRB data" + (" and street-level features..." if include_lighting else "...")):
                    ranked = rank_routes(
                        raw_routes, df,
                        safety_weight=safety_weight,
                        include_street_lighting=include_lighting,
                    )
                st.session_state["route_results"] = {
                    "ranked": ranked,
                    "origin_coords": origin_coords,
                    "dest_coords": dest_coords,
                    "origin_name": origin_address or origin_city_pick or "My Location",
                    "dest_name": dest_address or dest_city_pick or "Destination",
                    "include_lighting": include_lighting,
                }

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    def _risk_badge(avg_risk):
        """Soft, non-alarming qualitative label for a route's crime-proximity score."""
        if avg_risk < 0.35:
            return "🟢 Low", "#10b981"
        elif avg_risk < 0.65:
            return "🟡 Moderate", "#f59e0b"
        else:
            return "🟠 Elevated", "#f97316"

















    if "route_results" in st.session_state:
        result = st.session_state["route_results"]
        ranked = result["ranked"]
        origin_coords = result["origin_coords"]
        dest_coords = result["dest_coords"]

        st.markdown("---")
        st.subheader(f"{len(ranked)} Route Alternative(s) Found")

        best = ranked[0]
        tag_str = " · ".join(best.get("tags", [])) or "Recommended"
        badge_label, badge_color = _risk_badge(best["risk_info"]["avg_risk"])

        st.success(f"""
    ### ✅ Recommended: {tag_str}
    **{best.get('road_summary', 'Route')}**
    - **Distance:** {best['distance_km']} km
    - **Area Crime-Data Signal:** {badge_label}
    """)

        lighting = best.get("lighting_info", {})
        if lighting.get("available"):
            lit_txt = f"{lighting['lit_coverage_pct']}% explicitly tagged as lit" if lighting.get("lit_coverage_pct") is not None else "lighting tags not widely available for this area"
            st.caption(
                f"🔦 Street check: **{lighting['major_road_pct']}%** of this route runs along major/busier roads · {lit_txt}. "
                f"(Based on OpenStreetMap tagging — coverage varies by area.)"
            )
        elif include_lighting:
            st.caption("🔦 Street lighting/road-type data wasn't available for this route (route may be too long, or OpenStreetMap didn't respond in time).")

        if best.get("risk_tied_with_alternative"):
            st.caption(
                "⚖️ Note: the top two routes score within 1% of each other overall — "
                "given available data, they're effectively equally safe. The pick above "
                "is the faster of the two near-tied options."
            )

        if best["risk_info"]["nearby_cities"]:
            with st.expander("ℹ️ See the crime-data detail behind this score"):
                if len(best["risk_info"]["nearby_cities"]) == 1:
                    only_city = best["risk_info"]["nearby_cities"][0]["city"]
                    st.caption(
                        f"**{only_city}** is the only one of the NCRB-covered cities/districts "
                        f"within the 40 km search radius of this route, so it's driving this score."
                    )
                near_df = pd.DataFrame(best["risk_info"]["nearby_cities"])
                near_df.columns = ["City", "Distance from Route (km)", "Risk Level", "Risk Score"]
                st.dataframe(near_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Reminder: NCRB data here is city/district-level, not street-level — "
                    "this reflects the wider area, not the specific road."
                )
        else:
            st.caption("No NCRB-covered city/district falls within reach of this route — crime-data signal is neutral here.")

        # --------------------------------------------------
        # COMPARE ALL ALTERNATIVES — visual cards + chart
        # --------------------------------------------------
        st.markdown("---")
        st.markdown("**Compare All Alternatives**")

        cmp_cols = st.columns(len(ranked))
        for i, (col, r) in enumerate(zip(cmp_cols, ranked)):
            with col:
                badge_label, badge_color = _risk_badge(r["risk_info"]["avg_risk"])
                tags_html = "".join(
                    f'<span style="background:#ec4899;color:white;border-radius:6px;'
                    f'padding:2px 8px;font-size:0.75rem;margin-right:4px;">{t}</span>'
                    for t in r.get("tags", [])
                )
                st.markdown(f"""
    <div style="border:1px solid rgba(148,163,184,0.3); border-radius:12px; padding:14px; height:100%;">
      <div style="margin-bottom:6px;">{tags_html}</div>
      <div style="font-weight:600; font-size:0.95rem; margin-bottom:8px;">{r.get('road_summary', f'Route {i+1}')}</div>
      <div style="font-size:1.4rem; font-weight:700;">{r['distance_km']} km</div>
      <div style="color:{badge_color}; font-weight:600; margin-top:4px;">{badge_label} crime-data signal</div>
    </div>
    """, unsafe_allow_html=True)
                safety_pct = round((1 - r.get("norm_safety", 0)) * 100)
                st.progress(max(0, min(100, safety_pct)) / 100, text=f"Relative safety: {safety_pct}%")
                lt = r.get("lighting_info", {})
                if lt.get("available"):
                    st.caption(f"🔦 {lt['major_road_pct']}% major/busy roads")

        # ---- Map ----
        st.markdown("---")
        st.markdown("**Route Map**")

        route_colors = ["#10b981", "#3b82f6", "#f59e0b", "#a855f7"]
        route_map = go.Figure()

        for i, r in enumerate(ranked):
            lats = [c[0] for c in r["coords"]]
            lons = [c[1] for c in r["coords"]]
            label = r.get("road_summary", f"Route {i + 1}") + (f" — {', '.join(r['tags'])}" if r.get("tags") else "")
            route_map.add_trace(go.Scattermap(
                lat=lats, lon=lons, mode="lines",
                line=dict(width=5 if i == 0 else 3, color=route_colors[i % len(route_colors)]),
                opacity=1.0 if i == 0 else 0.55,
                name=label,
                hoverinfo="name"
            ))

        route_map.add_trace(go.Scattermap(
            lat=[origin_coords[0]], lon=[origin_coords[1]],
            mode="markers+text", marker=dict(size=16, color="#10b981"),
            text=["Start"], textposition="top right", name="Start"
        ))
        route_map.add_trace(go.Scattermap(
            lat=[dest_coords[0]], lon=[dest_coords[1]],
            mode="markers+text", marker=dict(size=16, color="#ef4444"),
            text=["Destination"], textposition="top right", name="Destination"
        ))

        risk_color = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
        for r_level, color in risk_color.items():
            level_df = df[(df["risk_level"] == r_level) & (~df["lat"].isna()) & (~df["lon"].isna())]
            if not level_df.empty:
                route_map.add_trace(go.Scattermap(
                    lat=level_df["lat"].tolist(),
                    lon=level_df["lon"].tolist(),
                    mode="markers",
                    marker=dict(size=12, color=color, opacity=0.8),
                    name=f"NCRB {r_level} Risk Zone",
                    text=[f"{c} ({r_level} Risk)" for c in level_df["city"]],
                    hoverinfo="text",
                    showlegend=True
                ))


        mid_lat = (origin_coords[0] + dest_coords[0]) / 2
        mid_lon = (origin_coords[1] + dest_coords[1]) / 2

        # ---- Auto-fit the view to everything actually plotted ----
        # (route lines, start/end markers, and all NCRB city markers) so the
        # map opens already showing everything, like Google Maps does when
        # you search a route — instead of a fixed zoom level that might crop
        # half your cities out.
        all_lats, all_lons = [], []
        for trace in route_map.data:
            if trace.lat is not None:
                all_lats.extend([v for v in trace.lat if v is not None])
            if trace.lon is not None:
                all_lons.extend([v for v in trace.lon if v is not None])

        if all_lats and all_lons:
            lat_span = max(all_lats) - min(all_lats)
            lon_span = max(all_lons) - min(all_lons)
            span = max(lat_span, lon_span, 0.0005)
            # Empirical degree-span -> zoom lookup (higher zoom = more zoomed in)
            _zoom_table = [
                (40, 3), (20, 4), (10, 5), (5, 6), (2, 7), (1, 8),
                (0.5, 9), (0.25, 10), (0.1, 11), (0.05, 12),
                (0.02, 13), (0.01, 14), (0, 15),
            ]
            auto_zoom = next(z for threshold, z in _zoom_table if span > threshold)
            center_lat = (max(all_lats) + min(all_lats)) / 2
            center_lon = (max(all_lons) + min(all_lons)) / 2
        else:
            auto_zoom = 9
            center_lat, center_lon = mid_lat, mid_lon

        route_map.update_layout(
            map=dict(
                style="carto-darkmatter" if not _is_light else "carto-positron",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=auto_zoom
            ),
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(bgcolor=chart_bg, font=dict(color=chart_text)),
            paper_bgcolor=chart_bg
        )
        # scrollZoom + the zoom/pan buttons in the toolbar give full Google-Maps-style
        # navigation: scroll or pinch to zoom, drag to pan, buttons to reset.
        st.plotly_chart(
            route_map, use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False}
        )

        with st.expander("🗺️ Map showing a blank/empty background instead of streets?"):
            st.markdown("""
    This map needs two things to show actual street tiles (not just the colored lines/markers):
    1. **A live internet connection** while the app is running — the street tiles load from an
       external map-tile service each time, they aren't bundled with the app.
    2. **A recent Plotly version.** The street-tile map used here needs Plotly 5.24 or newer.
       If you installed this project a while ago, update it:
       ```
       pip install --upgrade plotly
       ```
       then restart `streamlit run app.py`.

    If it's still blank after both of those, your network (campus/office Wi-Fi, firewall, VPN)
    may be blocking the tile provider — try a mobile hotspot to confirm.
    """)

        st.caption(
            "Green route = top recommendation for your chosen Fastest/Safest priority. "
            "Markers show nearby NCRB-covered cities colored by risk level. "
            "Routing via Google Routes API · Geocoding via Google Geocoding API."
        )

        # ==========================================================
        # LIVE NAVIGATION ASSISTANT
        # ==========================================================

        st.markdown("---")
        st.subheader("🧭 Live Navigation Assistant")
        st.markdown(
            "Click **▶ Start Navigation** to begin turn-by-turn GPS navigation along the "
            "recommended route. The assistant tracks your real position, announces upcoming "
            "turns, and alerts you when entering higher-risk areas."
        )

        # Build serialisable versions of data for JS
        best_steps = best.get("turn_steps", [])
        ncrb_cities_for_nav = [
            {
                "city": row["city"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "risk_level": row["risk_level"],
                "risk_score": float(row["risk_score"]),
            }
            for _, row in df.iterrows()
            if not pd.isna(row.get("lat")) and not pd.isna(row.get("lon"))
        ]
        best_polyline = [{"lat": lat, "lon": lon} for lat, lon in best["coords"]]

        # Google Maps JS key (for the embedded map tile in the navigation widget).
        # Checks Streamlit's secrets.toml first (recommended — also works after
        # deploying to Streamlit Cloud), falling back to an OS environment
        # variable if you'd rather set it that way instead.
        _key_source = None
        try:
            _gmaps_key = st.secrets.get("maps_api", "")
            if _gmaps_key:
                _key_source = "Streamlit secrets.toml"
        except Exception as _secrets_err:
            _gmaps_key = ""
            print(f"[SHEild] Couldn't read .streamlit/secrets.toml — {type(_secrets_err).__name__}: {_secrets_err}")
        if not _gmaps_key:
            _gmaps_key = os.environ.get("maps_api", "")
            if _gmaps_key:
                _key_source = "OS environment variable"

        # Diagnostic goes to the TERMINAL only (where `streamlit run` is running) —
        # never shown in the browser/app itself.
        if _gmaps_key:
            print(f"[SHEild] Google Maps key found via {_key_source}, ends in …{_gmaps_key[-6:]}")
        else:
            print(
                "[SHEild] No Google Maps key found — checked .streamlit/secrets.toml "
                "(key 'maps_api') and the 'maps_api' environment variable, found neither. "
                "Falling back to the schematic map."
            )

        nav_data_json = json.dumps({
            "steps": best_steps,
            "polyline": best_polyline,
            "ncrb_cities": ncrb_cities_for_nav,
            "origin": {"lat": origin_coords[0], "lon": origin_coords[1]},
            "dest": {"lat": dest_coords[0], "lon": dest_coords[1]},
            "distance_km": best["distance_km"],
            "duration_min": best["duration_min"],
            "road_summary": best.get("road_summary", ""),
            "gmaps_key": _gmaps_key,
            "is_light": _is_light,
        })

        nav_widget_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        font-family: 'Inter', -apple-system, sans-serif;
        background: {'#f8fafc' if _is_light else '#0f172a'};
        color: {'#0f172a' if _is_light else '#f8fafc'};
        height: 100%; overflow: hidden;
      }}
      #nav-shell {{
        display: flex; flex-direction: column; height: 100vh; padding: 0;
      }}
      /* ── Start button ── */
      #start-btn {{
        background: linear-gradient(135deg, #ec4899, #ec4899);
        color: white; border: none; border-radius: 12px;
        padding: 14px 28px; font-size: 1rem; font-weight: 700;
        cursor: pointer; margin: 16px auto; display: block;
        box-shadow: 0 4px 20px rgba(236,72,153,0.4);
        transition: transform .15s, box-shadow .15s;
      }}
      #start-btn:hover {{ transform: translateY(-2px); box-shadow: 0 6px 28px rgba(236,72,153,0.55); }}
      #start-btn:disabled {{ opacity: .5; cursor: default; transform: none; }}

      /* ── HUD ── */
      #nav-hud {{
        display: none; flex-direction: column; height: 100%;
      }}
      /* Next-turn card */
      #turn-card {{
        background: {'rgba(15,23,42,0.97)' if not _is_light else 'rgba(248,250,252,0.97)'};
        border-bottom: 1px solid {'rgba(255,255,255,0.1)' if not _is_light else 'rgba(15,23,42,0.1)'};
        padding: 12px 16px; display: flex; align-items: center; gap: 16px;
        flex-shrink: 0;
      }}
      #maneuver-icon {{
        font-size: 2.6rem; line-height: 1; width: 52px; text-align: center; flex-shrink: 0;
      }}
      #turn-info {{ flex: 1; min-width: 0; }}
      #turn-dist {{
        font-size: 1.6rem; font-weight: 800;
        color: #ec4899; line-height: 1.1;
      }}
      #turn-instruction {{
        font-size: 0.95rem; font-weight: 500; margin-top: 3px;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        color: {'#0f172a' if _is_light else '#f8fafc'};
      }}
      #turn-after {{
        font-size: 0.78rem; color: {'#64748b' if _is_light else '#94a3b8'}; margin-top: 2px;
      }}
      /* ETA strip */
      #eta-strip {{
        background: {'#10b981' if True else '#059669'};
        color: white; display: flex; justify-content: space-around;
        align-items: center; padding: 8px 16px; flex-shrink: 0;
        font-weight: 700; font-size: 0.95rem;
      }}
      #eta-strip span {{ display: flex; flex-direction: column; align-items: center; }}
      #eta-strip small {{ font-weight: 400; font-size: 0.7rem; opacity: 0.85; }}
      /* Safety alert banner */
      #safety-alert {{
        display: none;
        background: linear-gradient(90deg, #ef4444, #f97316);
        color: white; padding: 10px 16px; font-size: 0.87rem; font-weight: 600;
        text-align: center; flex-shrink: 0;
        animation: pulse-border 1.5s infinite;
      }}
      @keyframes pulse-border {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.82; }}
      }}
      /* Map container */
      #map-container {{ flex: 1; position: relative; min-height: 0; }}
      #nav-map {{ width: 100%; height: 100%; }}
      /* GPS accuracy badge */
      #gps-badge {{
        position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
        background: rgba(0,0,0,0.65); color: white; padding: 4px 12px;
        border-radius: 20px; font-size: 0.72rem; pointer-events: none;
        backdrop-filter: blur(4px);
      }}
      /* Arrived overlay */
      #arrived-overlay {{
        display: none; position: absolute; inset: 0;
        background: rgba(16,185,129,0.92); color: white;
        flex-direction: column; align-items: center; justify-content: center;
        font-size: 2rem; font-weight: 800; gap: 12px; text-align: center;
        padding: 24px;
      }}
      #arrived-overlay.show {{ display: flex; }}
      /* Step list (expandable) */
      #steps-toggle {{
        background: {'rgba(248,250,252,0.9)' if _is_light else 'rgba(15,23,42,0.9)'};
        border-top: 1px solid {'rgba(15,23,42,0.1)' if _is_light else 'rgba(255,255,255,0.1)'};
        padding: 6px 16px; font-size: 0.8rem; cursor: pointer;
        color: {'#475569' if _is_light else '#94a3b8'}; text-align: center;
        flex-shrink: 0; user-select: none;
      }}
      #steps-list {{
        display: none; background: {'#f1f5f9' if _is_light else '#1e293b'};
        max-height: 220px; overflow-y: auto; flex-shrink: 0;
      }}
      #steps-list.open {{ display: block; }}
      .step-row {{
        padding: 8px 16px; border-bottom: 1px solid {'rgba(15,23,42,0.07)' if _is_light else 'rgba(255,255,255,0.07)'};
        font-size: 0.82rem; display: flex; gap: 10px; align-items: flex-start;
      }}
      .step-row.active {{ background: rgba(236,72,153,0.12); font-weight: 600; }}
      .step-icon {{ font-size: 1rem; flex-shrink: 0; margin-top: 1px; }}
      .step-text {{ flex: 1; }}
      .step-dist {{ color: {'#64748b' if _is_light else '#94a3b8'}; font-size: 0.76rem; margin-top: 2px; }}
    </style>
    </head>
    <body>
    <div id="nav-shell">
      <button id="start-btn">▶ Start Navigation</button>

      <div id="nav-hud">
        <!-- Next-turn card -->
        <div id="turn-card">
          <div id="maneuver-icon">↑</div>
          <div id="turn-info">
            <div id="turn-dist">—</div>
            <div id="turn-instruction">Calculating…</div>
            <div id="turn-after"></div>
          </div>
        </div>

        <!-- ETA strip -->
        <div id="eta-strip">
          <span><span id="eta-time">—</span><small>ETA</small></span>
          <span><span id="eta-dist">—</span><small>Remaining</small></span>
          <span><span id="eta-mins">—</span><small>min left</small></span>
        </div>

        <!-- Safety alert -->
        <div id="safety-alert" id="safety-alert">⚠️ <span id="alert-text"></span></div>

        <!-- Map -->
        <div id="map-container">
          <div id="nav-map"></div>
          <div id="gps-badge">📡 Acquiring GPS…</div>
          <div id="arrived-overlay">
            <div>🎉</div>
            <div>You have arrived!</div>
            <div style="font-size:1rem;font-weight:400;margin-top:8px;">at your destination</div>
          </div>
        </div>

        <!-- Step list toggle -->
        <div id="steps-toggle" onclick="toggleSteps()">▼ All turns (tap to expand)</div>
        <div id="steps-list" id="steps-list"></div>
      </div>
    </div>

    <script>
    // ──────────────────────────────────────────────
    // DATA from Python (baked in at render time)
    // ──────────────────────────────────────────────
    const NAV_DATA = {nav_data_json};
    const STEPS    = NAV_DATA.steps;
    const POLYLINE = NAV_DATA.polyline;   // [{{lat,lon}},...]
    const CITIES   = NAV_DATA.ncrb_cities;
    const ORIGIN   = NAV_DATA.origin;
    const DEST     = NAV_DATA.dest;
    const GMAPS_KEY = NAV_DATA.gmaps_key;
    const IS_LIGHT  = NAV_DATA.is_light;

    // ──────────────────────────────────────────────
    // STATE
    // ──────────────────────────────────────────────
    let map, userMarker, positionCircle, watchId;
    let currentStepIdx = 0;
    let spokenAt = {{}};          // stepIdx -> {{500:bool,200:bool,50:bool}}
    let alertedCities = {{}};
    let lastPos = null;
    let arrived = false;
    let totalRemainingM = (NAV_DATA.distance_km || 0) * 1000;
    let stepsOpen = false;

    // ──────────────────────────────────────────────
    // MANEUVER → EMOJI ICON
    // ──────────────────────────────────────────────
    const ICON_MAP = {{
      'TURN_LEFT': '↰', 'TURN_RIGHT': '↱',
      'TURN_SHARP_LEFT': '⬅', 'TURN_SHARP_RIGHT': '➡',
      'TURN_SLIGHT_LEFT': '↖', 'TURN_SLIGHT_RIGHT': '↗',
      'MERGE': '⇒', 'FORK_LEFT': '↙', 'FORK_RIGHT': '↘',
      'ROUNDABOUT_LEFT': '↺', 'ROUNDABOUT_RIGHT': '↻',
      'FERRY': '⛴', 'UTURN_LEFT': '↩', 'UTURN_RIGHT': '↪',
      'STRAIGHT': '↑', 'NAME_CHANGE': '↑', 'CONTINUE': '↑',
      'DEPART': '🚦', 'DESTINATION': '🏁',
      'RAMP_LEFT': '↙', 'RAMP_RIGHT': '↘',
    }};
    function maneuverIcon(m) {{ return ICON_MAP[m] || '↑'; }}

    // ──────────────────────────────────────────────
    // HAVERSINE distance in metres
    // ──────────────────────────────────────────────
    function haversineM(lat1, lon1, lat2, lon2) {{
      const R = 6371000;
      const dLat = (lat2 - lat1) * Math.PI / 180;
      const dLon = (lon2 - lon1) * Math.PI / 180;
      const a = Math.sin(dLat/2)**2 +
                Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLon/2)**2;
      return 2 * R * Math.asin(Math.sqrt(a));
    }}

    // Nearest point on polyline to a lat/lon → {{lat,lon,idx,progress_m}}
    function snapToRoute(lat, lon) {{
      let bestDist = Infinity, bestPt = null, bestIdx = 0;
      for (let i = 0; i < POLYLINE.length; i++) {{
        const d = haversineM(lat, lon, POLYLINE[i].lat, POLYLINE[i].lon);
        if (d < bestDist) {{ bestDist = d; bestPt = POLYLINE[i]; bestIdx = i; }}
      }}
      return {{ pt: bestPt, idx: bestIdx, offRoute: bestDist > 80 }};
    }}

    // Remaining route distance from polyline index i
    function remainingDistM(fromIdx) {{
      let d = 0;
      for (let i = fromIdx; i < POLYLINE.length - 1; i++) {{
        d += haversineM(POLYLINE[i].lat, POLYLINE[i].lon, POLYLINE[i+1].lat, POLYLINE[i+1].lon);
      }}
      return d;
    }}

    // ──────────────────────────────────────────────
    // FORMAT helpers
    // ──────────────────────────────────────────────
    function fmtDist(m) {{
      if (m >= 1000) return (m/1000).toFixed(1) + ' km';
      return Math.round(m/10)*10 + ' m';
    }}
    function fmtETA(mins) {{
      const h = Math.floor(mins/60), m = Math.round(mins%60);
      return h > 0 ? `${{h}}h ${{m}}m` : `${{m}} min`;
    }}

    // ──────────────────────────────────────────────
    // SPEECH
    // ──────────────────────────────────────────────
    function speak(text, interrupt) {{
      if (!window.speechSynthesis) return;
      if (interrupt) window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = 'en-IN';
      u.rate = 1.05;
      window.speechSynthesis.speak(u);
    }}

    // ──────────────────────────────────────────────
    // UPDATE HUD
    // ──────────────────────────────────────────────
    function updateHUD(userLat, userLon) {{
      if (arrived) return;

      const snap = snapToRoute(userLat, userLon);
      const remM = remainingDistM(snap.idx);
      totalRemainingM = remM;

      // ETA
      const speedKmh = (lastPos && lastPos.speed && lastPos.speed > 0)
        ? lastPos.speed * 3.6 : 20; // default 20 km/h walking/slow
      const etaMins = (remM / 1000) / speedKmh * 60;
      document.getElementById('eta-dist').textContent = fmtDist(remM);
      document.getElementById('eta-mins').textContent = Math.round(etaMins);
      const etaDate = new Date(Date.now() + etaMins * 60000);
      document.getElementById('eta-time').textContent =
        etaDate.toLocaleTimeString([], {{hour:'2-digit', minute:'2-digit'}});

      // Arrived?
      const distToDest = haversineM(userLat, userLon, DEST.lat, DEST.lon);
      if (distToDest < 40) {{
        arrived = true;
        speak('You have arrived at your destination!', true);
        document.getElementById('arrived-overlay').classList.add('show');
        if (watchId) navigator.geolocation.clearWatch(watchId);
        return;
      }}

      // Find current step
      // Advance step if user has passed it (distance to step start < 30m AND not last)
      while (currentStepIdx < STEPS.length - 1) {{
        const s = STEPS[currentStepIdx];
        const dToStep = haversineM(userLat, userLon, s.lat, s.lon);
        const nextDist = STEPS[currentStepIdx + 1]
          ? haversineM(userLat, userLon, STEPS[currentStepIdx + 1].lat, STEPS[currentStepIdx + 1].lon)
          : Infinity;
        if (dToStep < 25 || nextDist < dToStep) {{ currentStepIdx++; }}
        else break;
      }}

      const step = STEPS[currentStepIdx];
      const nextStep = STEPS[currentStepIdx + 1] || null;
      const distToStep = haversineM(userLat, userLon, step.lat, step.lon);

      // Turn card
      document.getElementById('maneuver-icon').textContent = maneuverIcon(step.maneuver || 'STRAIGHT');
      document.getElementById('turn-dist').textContent = fmtDist(distToStep);
      document.getElementById('turn-instruction').textContent = step.instruction;
      document.getElementById('turn-after').textContent =
        nextStep ? 'Then: ' + nextStep.instruction.substring(0, 48) : '';

      // Voice — announce at 500, 200, 50 m
      if (!spokenAt[currentStepIdx]) spokenAt[currentStepIdx] = {{}};
      const spoken = spokenAt[currentStepIdx];
      if (distToStep <= 500 && !spoken[500]) {{
        speak('In ' + fmtDist(500) + ', ' + step.instruction, false);
        spoken[500] = true;
      }} else if (distToStep <= 200 && !spoken[200]) {{
        speak('In ' + fmtDist(200) + ', ' + step.instruction, false);
        spoken[200] = true;
      }} else if (distToStep <= 50 && !spoken[50]) {{
        speak(step.instruction, true);
        spoken[50] = true;
      }}

      // Highlight active step in list
      document.querySelectorAll('.step-row').forEach((el, i) => {{
        el.classList.toggle('active', i === currentStepIdx);
      }});

      // Safety zone check
      CITIES.forEach(c => {{
        if (!c.lat || !c.lon) return;
        const d = haversineM(userLat, userLon, c.lat, c.lon);
        if (d < 10000 && c.risk_level === 'High' && !alertedCities[c.city]) {{
          alertedCities[c.city] = true;
          const msg = `Caution: entering a higher-risk area near ${{c.city}}. Stay alert.`;
          document.getElementById('alert-text').textContent = msg;
          document.getElementById('safety-alert').style.display = 'block';
          speak(msg, false);
          setTimeout(() => {{ document.getElementById('safety-alert').style.display = 'none'; }}, 8000);
        }}
      }});

      // GPS badge
      if (snap.offRoute) {{
        document.getElementById('gps-badge').textContent = '⚠️ Off route by more than 80 m';
        document.getElementById('gps-badge').style.background = 'rgba(239,68,68,0.8)';
      }} else {{
        document.getElementById('gps-badge').textContent = '📡 GPS active';
        document.getElementById('gps-badge').style.background = 'rgba(0,0,0,0.65)';
      }}

      // Move map
      if (map) {{
        map.panTo({{lat: userLat, lng: userLon}});
        userMarker.setPosition({{lat: userLat, lng: userLon}});
        positionCircle.setCenter({{lat: userLat, lng: userLon}});
      }}
    }}

    // ──────────────────────────────────────────────
    // STEP LIST
    // ──────────────────────────────────────────────
    function buildStepList() {{
      const list = document.getElementById('steps-list');
      list.innerHTML = '';
      STEPS.forEach((s, i) => {{
        const row = document.createElement('div');
        row.className = 'step-row' + (i === 0 ? ' active' : '');
        row.innerHTML = `
          <div class="step-icon">${{maneuverIcon(s.maneuver || 'STRAIGHT')}}</div>
          <div class="step-text">
            <div>${{s.instruction}}</div>
            ${{s.distance_m > 0 ? `<div class="step-dist">${{fmtDist(s.distance_m)}}</div>` : ''}}
          </div>`;
        list.appendChild(row);
      }});
    }}
    function toggleSteps() {{
      stepsOpen = !stepsOpen;
      document.getElementById('steps-list').classList.toggle('open', stepsOpen);
      document.getElementById('steps-toggle').textContent =
        stepsOpen ? '▲ All turns (tap to collapse)' : '▼ All turns (tap to expand)';
    }}

    // ──────────────────────────────────────────────
    // GOOGLE MAPS INIT
    // ──────────────────────────────────────────────
    function initMap() {{
      const center = {{lat: ORIGIN.lat, lng: ORIGIN.lon}};
      const mapStyle = IS_LIGHT ? [] : [
        {{elementType:'geometry',stylers:[{{color:'#1e293b'}}]}},
        {{elementType:'labels.text.fill',stylers:[{{color:'#94a3b8'}}]}},
        {{featureType:'road',elementType:'geometry',stylers:[{{color:'#334155'}}]}},
        {{featureType:'road',elementType:'labels.text.fill',stylers:[{{color:'#94a3b8'}}]}},
        {{featureType:'water',elementType:'geometry',stylers:[{{color:'#0f172a'}}]}},
        {{featureType:'poi',stylers:[{{visibility:'off'}}]}},
        {{featureType:'transit',stylers:[{{visibility:'off'}}]}},
      ];

      map = new google.maps.Map(document.getElementById('nav-map'), {{
        zoom: 16,
        center: center,
        mapTypeId: 'roadmap',
        disableDefaultUI: true,
        zoomControl: true,
        styles: mapStyle,
      }});

      // Draw route polyline
      const path = POLYLINE.map(p => ({{lat: p.lat, lng: p.lon}}));
      new google.maps.Polyline({{
        path: path,
        geodesic: true,
        strokeColor: '#ec4899',
        strokeOpacity: 0.9,
        strokeWeight: 6,
        map: map,
      }});

      // Start/end markers
      new google.maps.Marker({{
        position: {{lat: ORIGIN.lat, lng: ORIGIN.lon}},
        map: map,
        title: 'Start',
        icon: {{ url: 'http://maps.google.com/mapfiles/ms/icons/green-dot.png' }},
      }});
      new google.maps.Marker({{
        position: {{lat: DEST.lat, lng: DEST.lon}},
        map: map,
        title: 'Destination',
        icon: {{ url: 'http://maps.google.com/mapfiles/ms/icons/red-dot.png' }},
      }});

      // User position marker (blue dot)
      userMarker = new google.maps.Marker({{
        position: center,
        map: map,
        title: 'You',
        icon: {{
          path: google.maps.SymbolPath.CIRCLE,
          scale: 10,
          fillColor: '#3b82f6',
          fillOpacity: 1,
          strokeColor: '#ffffff',
          strokeWeight: 3,
        }},
        zIndex: 999,
      }});

      // Accuracy circle
      positionCircle = new google.maps.Circle({{
        map: map,
        center: center,
        radius: 0,
        strokeColor: '#3b82f6',
        strokeOpacity: 0.3,
        strokeWeight: 1,
        fillColor: '#3b82f6',
        fillOpacity: 0.08,
      }});

      // NCRB city markers & threat overlay zones
      CITIES.forEach(c => {{
        if (!c.lat || !c.lon) return;
        const colorMap = {{High:'#ef4444', Medium:'#f59e0b', Low:'#10b981'}};
        const color = colorMap[c.risk_level] || '#94a3b8';
    
        // Draw a subtle "vicinity" halo around each covered city, not a
        // full-city-sized zone — with many covered cities close together
        // (e.g. Delhi's districts), large overlapping circles compound their
        // opacity and wash the whole map red. A small halo + the solid dot
        // marker below is enough to flag the city without doing that.
        const radiusMap = {{High: 2000, Medium: 1500, Low: 1000}};
        const opacityMap = {{High: 0.10, Medium: 0.06, Low: 0.03}};
        new google.maps.Circle({{
          map: map,
          center: {{lat: c.lat, lng: c.lon}},
          radius: radiusMap[c.risk_level] || 1000,
          fillColor: color,
          fillOpacity: opacityMap[c.risk_level] || 0.03,
          strokeColor: color,
          strokeOpacity: 0.35,
          strokeWeight: 1,
          clickable: false,
        }});

        // Draw central dot marker
        new google.maps.Marker({{
          position: {{lat: c.lat, lng: c.lon}},
          map: map,
          title: c.city + ' (' + c.risk_level + ' Risk, Score: ' + c.risk_score + ')',
          icon: {{
            path: google.maps.SymbolPath.CIRCLE,
            scale: 6,
            fillColor: color,
            fillOpacity: 0.9,
            strokeColor: '#ffffff',
            strokeWeight: 1.5,
          }},
        }});
      }});

      // Start GPS watch
      startGPS();
    }}

    function initOSMMap() {{
      // Fallback: draw a simple SVG overview map when no Google key is available
      const svgEl = document.getElementById('nav-map');
      if (!POLYLINE.length) return;
      const lats = POLYLINE.map(p=>p.lat), lons = POLYLINE.map(p=>p.lon);
      const minLat=Math.min(...lats), maxLat=Math.max(...lats);
      const minLon=Math.min(...lons), maxLon=Math.max(...lons);
      const W=700, H=500;
      const toX = lon => (lon-minLon)/(maxLon-minLon||1)*W;
      const toY = lat => H-(lat-minLat)/(maxLat-minLat||1)*H;
      let pts = POLYLINE.map(p=>`${{toX(p.lon)}},${{toY(p.lat)}}`).join(' ');
      svgEl.innerHTML = `<svg viewBox="0 0 ${{W}} ${{H}}" style="width:100%;height:100%;background:#1e293b">
        <polyline points="${{pts}}" fill="none" stroke="#ec4899" stroke-width="3"/>
        <circle cx="${{toX(ORIGIN.lon)}}" cy="${{toY(ORIGIN.lat)}}" r="8" fill="#10b981"/>
        <circle cx="${{toX(DEST.lon)}}" cy="${{toY(DEST.lat)}}" r="8" fill="#ef4444"/>
        <circle id="user-dot" cx="${{toX(ORIGIN.lon)}}" cy="${{toY(ORIGIN.lat)}}" r="6" fill="#3b82f6" stroke="#fff" stroke-width="2"/>
        <text x="10" y="20" fill="#94a3b8" font-size="12" font-family="Inter,sans-serif">Overview map (no Google Maps key set)</text>
      </svg>`;
      svgEl._toX = toX; svgEl._toY = toY;
      startGPS();
    }}

    function updateOSMDot(lat, lon) {{
      const dot = document.getElementById('user-dot');
      const svgEl = document.getElementById('nav-map');
      if (dot && svgEl._toX) {{
        dot.setAttribute('cx', svgEl._toX(lon));
        dot.setAttribute('cy', svgEl._toY(lat));
      }}
    }}

    // ──────────────────────────────────────────────
    // GPS WATCH
    // ──────────────────────────────────────────────
    function startGPS() {{
      if (!navigator.geolocation) {{
        document.getElementById('gps-badge').textContent = '❌ Geolocation not available';
        return;
      }}
      watchId = navigator.geolocation.watchPosition(
        function(pos) {{
          lastPos = pos.coords;
          const lat = pos.coords.latitude;
          const lon = pos.coords.longitude;
          const acc = Math.round(pos.coords.accuracy);
          document.getElementById('gps-badge').textContent = `📡 GPS ±${{acc}}m`;
          updateHUD(lat, lon);
          if (!GMAPS_KEY) updateOSMDot(lat, lon);
        }},
        function(err) {{
          document.getElementById('gps-badge').textContent = '⚠️ GPS: ' + err.message;
        }},
        {{
          enableHighAccuracy: true,
          maximumAge: 2000,
          timeout: 10000,
        }}
      );
    }}

    // ──────────────────────────────────────────────
    // START BUTTON
    // ──────────────────────────────────────────────
    document.getElementById('start-btn').addEventListener('click', function() {{
      this.disabled = true;
      this.textContent = '⏳ Starting…';
      buildStepList();
      document.getElementById('nav-hud').style.display = 'flex';
      this.style.display = 'none';
      // Trigger initial HUD with origin position
      updateHUD(ORIGIN.lat, ORIGIN.lon);

      if (GMAPS_KEY) {{
        // Load Google Maps JS API dynamically with graceful fallback
        const script = document.createElement('script');
        script.src = `https://maps.googleapis.com/maps/api/js?key=${{GMAPS_KEY}}&callback=initMap&loading=async`;
        script.async = true;
        script.defer = true;
        // If Maps JS fails (API not enabled, referrer restriction, invalid key)
        // fall back to the SVG overview map silently instead of showing the Google error screen.
        script.onerror = function() {{
          console.warn('[SHEild] Google Maps JS failed to load — falling back to SVG overview map.');
          initOSMMap();
        }};
        // Suppress the Google-injected "Oops" error overlay that appears inside the iframe.
        // Logged to the browser console only (F12 → Console) — never shown in the app itself.
        window.gm_authFailure = function() {{
          console.warn('[SHEild] Google Maps auth failure — check API enabled / billing / website restriction (localhost:8501). Falling back to schematic map.');
          if (map) return; // already initialised somehow
          initOSMMap();
        }};
        document.body.appendChild(script);
      }} else {{
        initOSMMap();
      }}
    }});

    </script>
    </body>
    </html>
    """

        components.html(nav_widget_html, height=720, scrolling=False)

with main_tab_explore:
    st.divider()

    st.markdown('<div id="sec-explore" class="section-anchor"></div>', unsafe_allow_html=True)

    st.header("📊 Explore Safety Data")
    st.caption("Dive into the full NCRB-based city dashboard, risk map, analytics, children & UT focus, and the safety advisor — organized into tabs below.")

    tab_overview, tab_map, tab_analytics, tab_children, tab_advisor = st.tabs([
        "🏙️ Overview & City Dashboard", "🗺️ Risk Map", "📊 Analytics", "👧 Children & UT Focus", "🧭 Safety Advisor"
    ])

    with tab_overview:
        # --------------------------------------------------
        # KPI CARDS
        # --------------------------------------------------

        st.header("Dashboard Overview")

        metrics = dashboard_metrics(df_display)
        ut_count = df_display[df_display["territory_type"] == "Union Territory"]["state_ut"].nunique()
        children_total = int(df_display["children_cases_2023"].sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Locations", metrics["total_cities"])
        c2.metric("UTs Covered", ut_count)
        c3.metric("Highest Risk", metrics["highest_risk_city"])
        c4.metric("High Risk Zones", metrics["high_risk"])
        c5.metric("Children Cases (2023)", f"{children_total:,}")

        st.divider()
        # ==========================================================
        # SECTION 1 – CITY DASHBOARD
        # ==========================================================

        st.markdown('<div id="sec-city" class="section-anchor"></div>', unsafe_allow_html=True)

        st.header("City Dashboard")
        st.subheader(f"Location: {selected_city}")

        if not city_data.empty:
            row = city_data.iloc[0]
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Cases (2023)", f"{int(row['cases_2023']):,}")
            col2.metric("Crime Rate (per lakh)", round(float(row["crime_rate_2023"]), 1))
            col3.metric("Risk Score", round(float(row["risk_score"]), 2))
            col4.metric("Safety Score", f"{round(float(row['safety_score']), 1)}/100")

            st.markdown("---")

            info_col, trend_col = st.columns([1, 2])

            with info_col:
                st.markdown(f"""
        **Location Details**
        - **State / UT:** {row['state_ut']}
        - **Territory Type:** {row['territory_type']}
        - **District:** {row['district']}
        - **Population:** {row['population_lakhs']} lakh
        - **Chargesheet Rate:** {row['chargesheet_rate']}%

        **Risk Classification**
        - **Risk Level:** {row['risk_level']}
        - **Predicted 2024 Cases:** {predict_next_year(row):,}

        **Children Safety (2023)**
        - **Total Cases:** {int(row['children_cases_2023']):,}
        - **POCSO Cases:** {int(row['pocso_2023']):,}
        - **Kidnapping:** {int(row['kidnapping_children_2023']):,}
        """)

            with trend_col:
                trend_df = pd.DataFrame({
                    "Year": ["2021", "2022", "2023"],
                    "Women Cases": [
                        int(row["cases_2021"]),
                        int(row["cases_2022"]),
                        int(row["cases_2023"])
                    ],
                    "Children Cases": [
                        int(row["children_cases_2021"]),
                        int(row["children_cases_2022"]),
                        int(row["children_cases_2023"])
                    ]
                })

                trend_fig = go.Figure()
                trend_fig.add_trace(go.Scatter(
                    x=trend_df["Year"], y=trend_df["Women Cases"],
                    mode="lines+markers", name="Women Cases",
                    line=dict(color="#ec4899", width=3),
                    marker=dict(size=10)
                ))
                trend_fig.add_trace(go.Scatter(
                    x=trend_df["Year"], y=trend_df["Children Cases"],
                    mode="lines+markers", name="Children Cases",
                    line=dict(color="#8b5cf6", width=3),
                    marker=dict(size=10)
                ))
                trend_fig.update_layout(
                    title=f"Crime Trend — {selected_city}",
                    xaxis_title="Year",
                    yaxis_title="Reported Cases",
                    plot_bgcolor=chart_bg,
                    paper_bgcolor=chart_bg,
                    font=dict(color=chart_text),
                    legend=dict(bgcolor=chart_bg)
                )
                st.plotly_chart(trend_fig, use_container_width=True)

            risk = row["risk_level"]
            if risk == "High":
                st.error(f"""
        ### High Risk Zone
        **{selected_city}** ({row['district']} district, {row['state_ut']}) shows elevated crime indicators.
        - High volume of reported cases relative to population
        - Crime rate significantly above average
        - Worsening or high trend over 2021–2023
        """)
            elif risk == "Medium":
                st.warning(f"""
        ### Medium Risk Zone
        **{selected_city}** has moderate safety concerns. Exercise standard precautions.
        """)
            else:
                st.success(f"""
        ### Lower Risk Zone
        **{selected_city}** shows comparatively lower crime indicators. Basic safety practices remain important.
        """)
        else:
            st.error("City data not found.")

        st.divider()


    with tab_map:
        # ==========================================================
        # SECTION 2 – RISK MAP
        # ==========================================================

        st.markdown('<div id="sec-map" class="section-anchor"></div>', unsafe_allow_html=True)

        st.header("Risk Map")
        st.subheader("Women Safety Risk Map — India")

        map_df = df_display.dropna(subset=["lat", "lon"])

        col_map1, col_map2 = st.columns([3, 1])

        with col_map1:
            map_view = st.radio(
                "Colour by", ["Risk Level", "Territory Type"], horizontal=True
            )

        if map_view == "Risk Level":
            color_col = "risk_level"
            color_map = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
        else:
            color_col = "territory_type"
            color_map = {"State": "#3b82f6", "Union Territory": "#a855f7"}

        map_fig = px.scatter_map(
            map_df,
            lat="lat",
            lon="lon",
            hover_name="city",
            hover_data={
                "state_ut": True,
                "territory_type": True,
                "district": True,
                "risk_level": True,
                "risk_score": ":.2f",
                "cases_2023": True,
                "children_cases_2023": True,
                "crime_rate_2023": True,
                "lat": False,
                "lon": False
            },
            color=color_col,
            size="cases_2023",
            size_max=30,
            zoom=4,
            height=650,
            color_discrete_map=color_map
        )
        map_fig.update_layout(
            map_style="carto-darkmatter",
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(map_fig, use_container_width=True)

        st.caption("Bubble size = total women cases (2023) | Colour = risk level or territory type")

        st.divider()


    with tab_analytics:
        # ==========================================================
        # SECTION 3 – ANALYTICS
        # ==========================================================

        st.markdown('<div id="sec-analytics" class="section-anchor"></div>', unsafe_allow_html=True)

        st.header("Analytics Dashboard")

        a1, a2 = st.columns(2)

        with a1:
            st.subheader("Top 10 Highest Risk Locations")
            top10 = df_display.sort_values("risk_score", ascending=False).head(10)[
                ["city", "state_ut", "territory_type", "district", "risk_level", "risk_score", "cases_2023", "crime_rate_2023"]
            ]
            st.dataframe(top10, use_container_width=True, hide_index=True)

        with a2:
            st.subheader("Top 10 Safest Locations")
            bot10 = df_display.sort_values("risk_score").head(10)[
                ["city", "state_ut", "territory_type", "district", "risk_level", "safety_score", "cases_2023"]
            ]
            st.dataframe(bot10, use_container_width=True, hide_index=True)

        st.markdown("---")

        bar_col, pie_col = st.columns(2)

        with bar_col:
            st.subheader("Risk Score by Location")
            top15 = df_display.sort_values("risk_score", ascending=False).head(15)
            bar = px.bar(
                top15,
                x="city",
                y="risk_score",
                color="risk_level",
                title="Top 15 Locations by Risk Score",
                color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                hover_data=["state_ut", "territory_type", "district"]
            )
            bar.update_layout(
                plot_bgcolor=chart_bg,
                paper_bgcolor=chart_bg,
                font=dict(color=chart_text),
                xaxis_tickangle=-45
            )
            st.plotly_chart(bar, use_container_width=True)

        with pie_col:
            st.subheader("Risk Distribution")
            pie = px.pie(
                df_display,
                names="risk_level",
                title="Risk Category Distribution",
                color="risk_level",
                color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                hole=0.4
            )
            pie.update_layout(
                plot_bgcolor=chart_bg,
                paper_bgcolor=chart_bg,
                font=dict(color=chart_text)
            )
            st.plotly_chart(pie, use_container_width=True)

        st.markdown("---")
        st.subheader("State / UT Summary")

        ut_summary = df_display.groupby(["state_ut", "territory_type"]).agg(
            Locations=("city", "count"),
            Total_Women_Cases_2023=("cases_2023", "sum"),
            Total_Children_Cases_2023=("children_cases_2023", "sum"),
            Avg_Crime_Rate=("crime_rate_2023", "mean"),
            High_Risk_Zones=("risk_level", lambda x: (x == "High").sum())
        ).reset_index().round(1)
        ut_summary.columns = [
            "State/UT", "Territory Type", "Locations",
            "Women Cases (2023)", "Children Cases (2023)",
            "Avg Crime Rate", "High Risk Zones"
        ]
        ut_summary = ut_summary.sort_values("Women Cases (2023)", ascending=False)
        st.dataframe(ut_summary, use_container_width=True, hide_index=True)

        st.divider()


    with tab_children:
        # ==========================================================
        # SECTION 4 – CHILDREN & UT FOCUS
        # ==========================================================

        st.markdown('<div id="sec-children" class="section-anchor"></div>', unsafe_allow_html=True)

        st.header("Children Safety & Union Territory Deep Dive")

        st.info("""
        **National Context (NCRB 2023)**
        India registered **1,77,335** crimes against children in 2023 (+9.2% from 2022).
        Major categories: **Kidnapping & Abduction** (79,884 cases, 45%) · **POCSO Act** (67,694 cases, 38%)
        """)

        st.markdown("---")
        st.subheader("Union Territories — Complete Overview")

        ut_df = df[df["territory_type"] == "Union Territory"].copy()

        ut_cols = st.columns(4)
        ut_cols[0].metric("UT Locations", len(ut_df))
        ut_cols[1].metric("Total UT Women Cases (2023)", f"{int(ut_df['cases_2023'].sum()):,}")
        ut_cols[2].metric("Total UT Children Cases (2023)", f"{int(ut_df['children_cases_2023'].sum()):,}")
        ut_cols[3].metric("UT High Risk Zones", int((ut_df["risk_level"] == "High").sum()))

        st.markdown("---")

        st.subheader("All Union Territory Districts — Crime Data")
        ut_table = ut_df[[
            "city", "state_ut", "district", "cases_2021", "cases_2022", "cases_2023",
            "crime_rate_2023", "children_cases_2023", "pocso_2023",
            "kidnapping_children_2023", "risk_level", "safety_score"
        ]].sort_values("cases_2023", ascending=False).copy()

        ut_table.columns = [
            "City/Town", "Union Territory", "District",
            "Women Cases 2021", "Women Cases 2022", "Women Cases 2023",
            "Crime Rate (per lakh)", "Children Cases 2023",
            "POCSO Cases 2023", "Children Kidnapping 2023",
            "Risk Level", "Safety Score"
        ]
        st.dataframe(ut_table, use_container_width=True, hide_index=True)

        st.markdown("---")

        ch1, ch2 = st.columns(2)

        with ch1:
            st.subheader("Women Cases by Union Territory (2023)")
            ut_agg = ut_df.groupby("state_ut").agg(
                Women_Cases=("cases_2023", "sum"),
                Children_Cases=("children_cases_2023", "sum"),
                POCSO=("pocso_2023", "sum"),
                Locations=("city", "count")
            ).reset_index().sort_values("Women_Cases", ascending=True)

            bar_ut = px.bar(
                ut_agg,
                x="Women_Cases",
                y="state_ut",
                orientation="h",
                color="Women_Cases",
                color_continuous_scale="Reds",
                title="Total Reported Women Cases by UT (2023)",
                labels={"Women_Cases": "Cases", "state_ut": "Union Territory"}
            )
            bar_ut.update_layout(
                plot_bgcolor=chart_bg,
                paper_bgcolor=chart_bg,
                font=dict(color=chart_text),
                coloraxis_showscale=False
            )
            st.plotly_chart(bar_ut, use_container_width=True)

        with ch2:
            st.subheader("Children Crime Breakdown by UT (2023)")
            ut_children = ut_df.groupby("state_ut").agg(
                POCSO=("pocso_2023", "sum"),
                Kidnapping=("kidnapping_children_2023", "sum")
            ).reset_index()

            fig_children = go.Figure()
            fig_children.add_trace(go.Bar(
                name="POCSO Cases",
                x=ut_children["state_ut"],
                y=ut_children["POCSO"],
                marker_color="#a855f7"
            ))
            fig_children.add_trace(go.Bar(
                name="Kidnapping",
                x=ut_children["state_ut"],
                y=ut_children["Kidnapping"],
                marker_color="#ec4899"
            ))
            fig_children.update_layout(
                barmode="group",
                title="Children Crime by UT — POCSO vs Kidnapping (2023)",
                plot_bgcolor=chart_bg,
                paper_bgcolor=chart_bg,
                font=dict(color=chart_text),
                xaxis_tickangle=-30
            )
            st.plotly_chart(fig_children, use_container_width=True)

        st.markdown("---")
        st.subheader("Delhi Districts — Detailed Breakdown")

        delhi_df = df[df["state_ut"] == "Delhi"].sort_values("cases_2023", ascending=False)

        dl1, dl2 = st.columns(2)

        with dl1:
            delhi_bar = px.bar(
                delhi_df,
                x="city",
                y="cases_2023",
                color="risk_level",
                title="Delhi Districts — Women Cases (2023)",
                color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                hover_data=["district", "crime_rate_2023", "children_cases_2023"]
            )
            delhi_bar.update_layout(
                plot_bgcolor=chart_bg,
                paper_bgcolor=chart_bg,
                font=dict(color=chart_text),
                xaxis_tickangle=-35
            )
            st.plotly_chart(delhi_bar, use_container_width=True)

        with dl2:
            delhi_children = px.bar(
                delhi_df,
                x="city",
                y=["pocso_2023", "kidnapping_children_2023"],
                title="Delhi Districts — Children Crime (2023)",
                barmode="group",
                color_discrete_map={
                    "pocso_2023": "#a855f7",
                    "kidnapping_children_2023": "#ec4899"
                },
                labels={
                    "value": "Cases",
                    "variable": "Crime Type",
                    "pocso_2023": "POCSO",
                    "kidnapping_children_2023": "Kidnapping"
                }
            )
            delhi_children.update_layout(
                plot_bgcolor=chart_bg,
                paper_bgcolor=chart_bg,
                font=dict(color=chart_text),
                xaxis_tickangle=-35
            )
            st.plotly_chart(delhi_children, use_container_width=True)

        st.markdown("---")
        st.subheader("Jammu & Kashmir — District Breakdown")

        jk_df = df[df["state_ut"] == "Jammu & Kashmir"].sort_values("cases_2023", ascending=False)
        jk_table = jk_df[[
            "city", "district", "cases_2021", "cases_2022", "cases_2023",
            "crime_rate_2023", "children_cases_2023", "pocso_2023",
            "kidnapping_children_2023", "risk_level"
        ]].copy()
        jk_table.columns = [
            "City/Town", "District", "Women 2021", "Women 2022", "Women 2023",
            "Crime Rate", "Children 2023", "POCSO", "Kidnapping", "Risk"
        ]
        st.dataframe(jk_table, use_container_width=True, hide_index=True)

        jk_fig = px.bar(
            jk_df,
            x="city",
            y="cases_2023",
            color="risk_level",
            title="J&K Districts — Women Cases (2023)",
            color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
            hover_data=["district", "crime_rate_2023"]
        )
        jk_fig.update_layout(
            plot_bgcolor=chart_bg,
            paper_bgcolor=chart_bg,
            font=dict(color=chart_text),
            xaxis_tickangle=-45
        )
        st.plotly_chart(jk_fig, use_container_width=True)

        st.divider()


    with tab_advisor:
        # ==========================================================
        # SECTION 5 – SAFETY ADVISOR
        # ==========================================================

        st.markdown('<div id="sec-advisor" class="section-anchor"></div>', unsafe_allow_html=True)

        st.header("Safety Advisor")
        st.subheader(f"Assessment for: {selected_city}")

        if not city_data.empty:
            row = city_data.iloc[0]
            risk = row["risk_level"]
            score = round(float(row["risk_score"]), 2)
            safety = round(float(row["safety_score"]), 1)

            adv_col1, adv_col2 = st.columns([1, 2])

            with adv_col1:
                st.metric("Risk Score", score)
                st.metric("Safety Score", f"{safety}/100")
                st.metric("Predicted Cases (2024)", f"{predict_next_year(row):,}")
                st.metric("Children Cases (2023)", f"{int(row['children_cases_2023']):,}")

            with adv_col2:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=safety,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Safety Score", "font": {"color": chart_text}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": chart_text},
                        "bar": {"color": "#10b981" if risk == "Low" else "#f59e0b" if risk == "Medium" else "#ef4444"},
                        "steps": [
                            {"range": [0, 33], "color": "rgba(239,68,68,0.15)"},
                            {"range": [33, 66], "color": "rgba(245,158,11,0.15)"},
                            {"range": [66, 100], "color": "rgba(16,185,129,0.15)"}
                        ],
                        "threshold": {
                            "line": {"color": chart_text, "width": 2},
                            "thickness": 0.75,
                            "value": safety
                        },
                        "bgcolor": chart_bg
                    },
                    number={"font": {"color": chart_text}}
                ))
                gauge.update_layout(
                    paper_bgcolor=chart_bg,
                    height=220,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(gauge, use_container_width=True)

            st.markdown("---")

            if risk == "High":
                st.error(f"""
        ### High Risk Assessment
        **{selected_city}** ({row['district']}, {row['state_ut']}) is in the **HIGH RISK** category.

        **Observed Indicators:**
        - Crime volume: **{int(row['cases_2023']):,} reported cases in 2023**
        - Crime rate: **{row['crime_rate_2023']} per lakh population** (national avg: 66.2)
        - Crime trend: {"Increasing" if row['cases_2023'] > row['cases_2021'] else "Declining"}
        - Children at risk: **{int(row['children_cases_2023']):,} cases** including {int(row['pocso_2023'])} POCSO
        """)
            elif risk == "Medium":
                st.warning(f"""
        ### Medium Risk Assessment
        **{selected_city}** has moderate safety concerns. Stay aware and take standard precautions.

        - Cases 2023: **{int(row['cases_2023']):,}**
        - Crime rate: **{row['crime_rate_2023']}** per lakh population
        """)
            else:
                st.success(f"""
        ### Lower Risk Zone
        **{selected_city}** shows comparatively lower crime indicators based on NCRB 2023 data.

        - Cases 2023: **{int(row['cases_2023']):,}**
        - Safety Score: **{safety}/100**
        """)

            st.subheader("Safety Recommendations")
            recommendations = generate_recommendation(risk)
            for rec in recommendations:
                st.write(rec)


        else:
            st.error("City data not found.")

        st.divider()

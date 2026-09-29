import base64
import glob
import io
import json
import logging
import os
import re
import textwrap
import time
import yaml
from fpdf import FPDF
from groq import Groq
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

st.set_page_config(
    page_title="Kidney Stone Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Needs to exist before the home-page background block below (which checks it)
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "home"

st.markdown(
    """
<style>
    /* ---- App-wide light gradient background (calm, clean, hospital-white feel) ---- */
    .stApp {
        background: linear-gradient(180deg, #ffffff 0%, #f4fbfa 45%, #eafaf7 100%);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }

    @keyframes main-header-rise {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main-header {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%);
        border: 1px solid #d5f0ec;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(15, 118, 110, 0.06), 0 12px 28px rgba(15, 118, 110, 0.1);
        position: relative;
        overflow: hidden;
        animation: main-header-rise 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .main-header::after {
        content: "";
        position: absolute;
        left: 50%;
        bottom: 0;
        transform: translateX(-50%);
        width: 90px;
        height: 4px;
        border-radius: 4px 4px 0 0;
        background: linear-gradient(90deg, #0d9488, #0ea5e9);
    }
    @media (prefers-reduced-motion: reduce) {
        .main-header { animation: none; }
    }
    .main-header h1 {
        color: #0d9488;
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
    }
    .main-header-icon {
        width: 58px;
        height: 58px;
        margin: 0 auto 0.8rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.7rem;
        border-radius: 50%;
        background: linear-gradient(135deg, #ffffff, #e6fbf7);
        border: 1px solid rgba(13, 148, 136, 0.25);
        box-shadow: 0 8px 18px rgba(13, 148, 136, 0.18);
        animation: hero-float-3d 5s ease-in-out infinite;
    }
    @media (prefers-reduced-motion: reduce) {
        .main-header-icon { animation: none; }
    }
    .main-header p {
        color: #5b7370;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }
    .sidebar-datasets {
        text-align: left;
        margin-bottom: 0.6rem;
        margin-top: -0.8rem;
    }
    .sidebar-datasets .datasets-title {
        color: #0d9488;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
        display: block;
    }
    .sidebar-datasets .dataset-box {
        background: rgba(13, 148, 136, 0.06);
        border: 1px solid rgba(13, 148, 136, 0.25);
        border-radius: 8px;
        padding: 0.5rem 0.7rem;
        color: #38534f;
        font-size: 0.78rem;
    }
    .sidebar-datasets .dataset-box .dataset-item {
        padding: 0.25rem 0;
    }
    .sidebar-datasets .dataset-box .dataset-item:not(:last-child) {
        border-bottom: 1px solid rgba(13, 148, 136, 0.15);
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f2fbf9);
        border: 1px solid #d9efec;
        border-radius: 14px;
        padding: 1.4rem 1rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(15, 118, 110, 0.08), 0 8px 20px rgba(15, 118, 110, 0.07);
        margin-bottom: 1rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        /* Lets the value/label size respond to the card's own box width
           (e.g. when 3 metric cards squeeze into a narrow mobile column)
           rather than only the viewport width. */
        container-type: inline-size;
        container-name: metric-card;
    }
    .metric-card:hover {
        transform: perspective(700px) rotateX(4deg) translateY(-4px) scale(1.015);
        box-shadow: 0 2px 4px rgba(13, 148, 136, 0.15), 0 20px 34px rgba(13, 148, 136, 0.22);
        border-color: rgba(13, 148, 136, 0.4);
    }
    .metric-card .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0d9488;
        line-height: 1;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #517470;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    /* Card itself is narrow (e.g. stacked 3-up on a small phone) — shrink
       the numbers so they don't wrap or overflow the card. */
    @container metric-card (max-width: 110px) {
        .metric-card { padding: 1rem 0.6rem; }
        .metric-card .metric-value { font-size: 1.6rem; }
        .metric-card .metric-label { font-size: 0.7rem; letter-spacing: 0.5px; }
    }

    .status-detected {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
        padding: 0.5rem 1.4rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        box-shadow: 0 4px 16px rgba(220, 38, 38, 0.25);
    }
    .status-clear {
        background: linear-gradient(135deg, #15803d, #22c55e);
        color: white;
        padding: 0.5rem 1.4rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        box-shadow: 0 4px 16px rgba(34, 197, 94, 0.25);
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0d9488;
        border-left: 4px solid #0d9488;
        padding-left: 0.75rem;
        margin-bottom: 1rem;
    }

    .stFileUploader > div {
        border: 2px dashed #0d9488 !important;
        border-radius: 12px !important;
        background: rgba(13, 148, 136, 0.03) !important;
        transition: background 0.18s ease, border-color 0.18s ease;
    }
    .stFileUploader > div:hover {
        background: rgba(13, 148, 136, 0.07) !important;
        border-color: #0f766e !important;
    }

    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff, #f0fdfa);
        border-right: 1px solid #d9efec;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label {
        color: #3f5c58 !important;
    }

    hr { border-color: #d9efec !important; }

    .info-box {
        background: rgba(14, 165, 233, 0.06);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        color: #0369a1;
        font-size: 0.9rem;
        line-height: 1.65;
    }
    .warn-box {
        background: rgba(217, 119, 6, 0.07);
        border: 1px solid rgba(217, 119, 6, 0.3);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        color: #b45309;
        font-size: 0.9rem;
    }

    .footer-box {
        background: linear-gradient(135deg, #ffffff 0%, #fbfefe 100%);
        border: 1px solid #d9efec;
        border-left: 4px solid #0d9488;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        color: #4a6360;
        font-size: 0.82rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(15, 118, 110, 0.05), 0 8px 20px rgba(15, 118, 110, 0.06);
    }
    .footer-creators {
        margin-top: 0.9rem;
        padding-top: 0.9rem;
        border-top: 1px solid #d9efec;
        color: #3f5c58;
        font-size: 0.85rem;
    }
    .footer-creators .creators-title {
        color: #0d9488;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-size: 0.72rem;
        display: block;
        margin-bottom: 0.35rem;
    }
    .footer-creators .creator-name {
        display: inline-block;
        background: rgba(13, 148, 136, 0.06);
        border: 1px solid rgba(13, 148, 136, 0.25);
        border-radius: 50px;
        padding: 0.25rem 0.9rem;
        margin: 0.2rem;
        color: #1e293b;
        font-size: 0.82rem;
        transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
    }
    .footer-creators .creator-name:hover {
        background: #0d9488;
        color: #ffffff;
        border-color: #0d9488;
        transform: translateY(-3px) scale(1.04);
        box-shadow: 0 8px 16px rgba(13, 148, 136, 0.28);
    }
    .footer-creators .creator-name:hover a {
        color: #ffffff !important;
    }
    @media (prefers-reduced-motion: reduce) {
        .footer-creators .creator-name:hover { transform: none; }
    }
    .footer-creators .creator-links {
        margin-left: 0.5rem;
        padding-left: 0.5rem;
        border-left: 1px solid rgba(13, 148, 136, 0.3);
        font-size: 0.75rem;
    }
    .footer-creators .creator-links a {
        color: #0d9488;
        text-decoration: none;
        font-weight: 600;
    }
    .footer-creators .creator-links a:hover {
        text-decoration: underline;
    }

    /* ---- Explore the Project resource cards ---- */
    .resource-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 0.75rem;
        margin: 0.6rem 0 1.4rem 0;
    }
    .resource-card {
        background: linear-gradient(135deg, #ffffff 0%, #f6fdfc 100%);
        border: 1px solid #d9efec;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 2px rgba(15, 118, 110, 0.06), 0 6px 14px rgba(15, 118, 110, 0.05);
        transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
    }
    .resource-card:hover {
        border-color: rgba(13, 148, 136, 0.35);
        box-shadow: 0 2px 4px rgba(15, 118, 110, 0.1), 0 16px 28px rgba(15, 118, 110, 0.18);
        transform: perspective(700px) rotateX(3deg) translateY(-4px) scale(1.015);
    }
    .resource-card .resource-head {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.4rem;
    }
    .resource-card .resource-icon {
        flex: none;
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        background: rgba(13, 148, 136, 0.08);
        border-radius: 8px;
    }
    .resource-card h4 {
        color: #0f172a;
        font-size: 0.86rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.25;
    }
    .resource-card p {
        color: #64748b;
        font-size: 0.76rem;
        line-height: 1.4;
        margin: 0 0 0.5rem 0;
    }
    .resource-card a {
        color: #0d9488;
        font-weight: 700;
        font-size: 0.76rem;
        text-decoration: none;
    }
    .resource-card a:hover {
        text-decoration: underline;
    }
    /* ---- Dark monospace code/formula box (e.g. class-weight formula) ---- */
    .code-box {
        background: #0f172a;
        color: #7dd3fc;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        white-space: pre-wrap;
        margin: 0.6rem 0;
    }
    /* ---- Leakage-check pill row (Train ∩ Val = ∅, etc.) ---- */
    .leakage-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin: 0.8rem 0;
    }
    .leakage-pill {
        background: rgba(13, 148, 136, 0.06);
        border: 1px solid rgba(13, 148, 136, 0.3);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.85rem;
        color: #0f766e;
        font-weight: 700;
        flex: 1 1 0;
        text-align: center;
    }

    /* Hide only the Deploy button; keep the 3-dot menu visible */
    [data-testid="stAppDeployButton"] {
        display: none;
    }
    div[data-testid="stToolbarActions"] button[kind="header"] {
        display: none;
    }

    /* ---- Landing page ---- */
    .landing-wrap {
        position: relative;
        overflow: hidden;
        min-height: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 0.7rem 2rem 1.4rem 2rem;
        border-radius: 16px;
        background: radial-gradient(circle at 50% 30%, rgba(13,148,136,0.05), transparent 60%);
    }
    /* ---- Pointer-following spotlight glow over the hero ---- */
    @property --spot-size {
        syntax: "<length-percentage>";
        inherits: true;
        initial-value: 0%;
    }
    .landing-wrap::before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background: radial-gradient(
            circle at var(--mx, 50%) var(--my, 30%),
            rgba(13, 148, 136, 0.22),
            transparent var(--spot-size, 0%)
        );
        transition: --spot-size 0.3s ease;
    }
    .landing-wrap:hover::before {
        --spot-size: 42%;
    }
    @media (prefers-reduced-motion: reduce) {
        .landing-wrap::before { transition: none; }
    }

    /* ---------------------------------------------------------------
       Dock the "Start Detection" CTA onto the bottom of the hero card
       instead of leaving it floating separately over the background
       art below. Pulled up flush against the hero (negative margin)
       with matching rounded-bottom corners and a frosted card
       background, so it reads as one continuous card with the hero
       above it rather than a disconnected button.
    --------------------------------------------------------------- */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-hero_cta_dock) {
        margin: 0 !important;
        min-height: 0 !important;
    }
    .st-key-hero_cta_dock {
        margin: -1px 0 1.4rem 0 !important;
        padding: 1rem 2rem 1.3rem 2rem !important;
        background: rgba(255, 255, 255, 0.35) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px !important;
        border-top: 1px solid rgba(13, 148, 136, 0.15) !important;
        box-shadow: 0 10px 24px rgba(15, 118, 110, 0.08);
    }
    .st-key-hero_cta_dock .stButton > button {
        box-shadow: 0 6px 18px rgba(13, 148, 136, 0.3) !important;
    }
    .landing-topbar {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 999px;
        padding: 0.25rem 0.8rem 0.25rem 0.25rem;
        width: fit-content;
        box-shadow: 0 4px 14px rgba(15, 118, 110, 0.08);
    }
    .landing-topbar .topbar-icon-chip {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(13, 148, 136, 0.25);
        box-shadow: 0 4px 14px rgba(15, 118, 110, 0.18);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        padding: 5px;
    }
    .landing-topbar img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        flex-shrink: 0;
    }
    .landing-topbar span {
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        line-height: 1.3;
        color: #0d9488;
        text-align: left;
    }
    .hero-title-row h1 {
        margin-bottom: 0 !important;
    }
    .landing-content {
        position: relative;
        z-index: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .landing-wrap h1 {
        color: #0d9488;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 0.6rem;
    }
    .landing-wrap .landing-sub {
        color: #4b6763;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        max-width: 700px;
    }
    .landing-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-bottom: 0.5rem;
    }
    .landing-badge {
        flex: 0 0 auto;
        background: rgba(14, 165, 233, 0.05);
        border: 1px solid rgba(14, 165, 233, 0.25);
        border-radius: 10px;
        padding: 0.6rem 1.1rem;
        color: #111827;
        font-weight: 600;
        font-size: 0.82rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
    }
    .landing-badge:hover {
        transform: translateY(-2px);
        border-color: rgba(14, 165, 233, 0.55);
        background: rgba(14, 165, 233, 0.1);
    }
    .landing-badge b {
        color: #111827;
        font-weight: 700;
    }
    /* Headline stats (the actual numbers worth leading with) get a
       stronger accent so they read before the supporting feature tags. */
    .landing-badge--stat {
        background: linear-gradient(135deg, rgba(13, 148, 136, 0.1), rgba(14, 165, 233, 0.08));
        border: 1px solid rgba(13, 148, 136, 0.4);
    }
    .landing-badge--stat b {
        color: #0f766e;
        font-size: 0.95rem;
    }
    .landing-badge--stat:hover {
        border-color: rgba(13, 148, 136, 0.7);
        background: linear-gradient(135deg, rgba(13, 148, 136, 0.16), rgba(14, 165, 233, 0.13));
    }

    /* ---- Centered hero: pill badge + big title + subtitle + stat bar ---- */
    .hero-centered {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 0.5rem;
        width: 100%;
        max-width: 900px;
        margin: 0 auto;
        padding: 0.1rem 1rem 0.3rem;
        position: relative;
        z-index: 1;
    }
    .hero-logo-centered {
        flex: 0 0 auto !important;
        margin-bottom: 0.1rem;
    }
    .hero-logo-centered .hero-illustration-card {
        max-width: 84px;
    }
    /* ---- Hero content rises up into place on load, staggered ---- */
    @keyframes hero-rise-in {
        from { opacity: 0; transform: translateY(34px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .hero-pill, .hero-big-title, .hero-subtitle-centered, .hero-stat-bar, .hero-feature-tags {
        animation: hero-rise-in 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .hero-pill { animation-delay: 0.02s; }
    .hero-big-title { animation-delay: 0.12s; }
    .hero-subtitle-centered { animation-delay: 0.26s; }
    .hero-stat-bar { animation-delay: 0.38s; }
    .hero-feature-tags { animation-delay: 0.5s; }
    @media (prefers-reduced-motion: reduce) {
        .hero-pill, .hero-big-title, .hero-subtitle-centered, .hero-stat-bar, .hero-feature-tags {
            animation: none;
        }
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(13, 148, 136, 0.35);
        color: #0d9488;
        font-weight: 700;
        font-size: 0.74rem;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .hero-pill:hover {
        transform: perspective(500px) rotateX(6deg) translateY(-2px) scale(1.03);
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.14);
    }
    .hero-big-title {
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.16;
        margin: 0.15rem 0 0.2rem 0;
        max-width: 700px;
        text-shadow: 0 0 18px rgba(255, 255, 255, 0.95), 0 0 6px rgba(255, 255, 255, 0.9), 0 2px 4px rgba(255, 255, 255, 0.7);
    }
    .hero-big-title .title-dark { color: #0f172a; }
    .hero-big-title .title-accent { color: #0d9488; }
    .hero-subtitle-centered {
        max-width: 620px;
        color: #334155;
        font-size: 0.88rem;
        line-height: 1.5;
        margin: 0 auto 0.2rem auto;
        text-shadow: 0 0 14px rgba(255, 255, 255, 0.95), 0 0 4px rgba(255, 255, 255, 0.9);
    }
    .hero-stat-bar {
        display: flex;
        flex-wrap: wrap;
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 14px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 8px 20px rgba(15, 23, 42, 0.06);
        overflow: hidden;
        margin-top: 0.1rem;
        width: 100%;
    }
    .hero-stat {
        flex: 1 1 120px;
        padding: 0.6rem 0.9rem;
        text-align: center;
        border-right: 1px solid rgba(15, 23, 42, 0.07);
        transition: background 0.18s ease, transform 0.18s ease;
    }
    .hero-stat:last-child { border-right: none; }
    .hero-stat:hover {
        background: rgba(13, 148, 136, 0.06);
        transform: perspective(600px) rotateX(5deg) translateY(-2px) scale(1.02);
    }
    .hero-stat-value {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0d9488;
    }
    .hero-stat-label {
        font-size: 0.62rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.15rem;
    }
    .hero-feature-tags {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.5rem;
        margin-top: 0.9rem;
    }
    .feature-tag {
        flex: 0 0 auto;
        background: rgba(14, 165, 233, 0.05);
        border: 1px solid rgba(14, 165, 233, 0.25);
        border-radius: 999px;
        padding: 0.4rem 1rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: #111827;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }
    .feature-tag:hover {
        transform: perspective(500px) rotateX(6deg) translateY(-3px) scale(1.04);
        border-color: rgba(14, 165, 233, 0.5);
        box-shadow: 0 8px 16px rgba(14, 165, 233, 0.18);
    }
    @media (prefers-reduced-motion: reduce) {
        .hero-pill:hover, .hero-stat:hover, .feature-tag:hover { transform: none; }
    }
    @media (max-width: 700px) {
        .hero-big-title { font-size: 1.5rem; }
        .hero-stat { flex: 1 1 45%; border-right: none; border-bottom: 1px solid rgba(15,23,42,0.07); }
    }

    /* ---- Hero split layout (text + logo badge) — legacy, currently unused
       by the centered hero above, kept in case an older layout is restored ---- */
    .hero-split {
        display: flex;
        align-items: center;
        gap: 2.2rem;
        text-align: left;
        width: 100%;
        max-width: 1220px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    .hero-split-text {
        flex: 1.15;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        position: relative;
        z-index: 1;
    }
    .hero-split-text h1 {
        color: #0d9488;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 0.3px;
        margin-bottom: 0.5rem;
        line-height: 1.15;
        cursor: default;
        background: linear-gradient(90deg, #0d9488, #14b8a6, #0ea5e9, #0d9488);
        background-size: 300% 100%;
        background-position: 0% 0;
        -webkit-background-clip: text;
        background-clip: text;
        /* currentColor (= the solid teal above) is what actually paints the
           glyphs at rest, so the title looks like normal solid bold text —
           the gradient behind it only becomes visible once the fill turns
           transparent on hover, letting the clipped gradient show through. */
        -webkit-text-fill-color: currentColor;
        transition: background-position 0.7s ease, -webkit-text-fill-color 0.3s ease;
    }
    .hero-title-row:hover h1 {
        background-position: 100% 0;
        -webkit-text-fill-color: transparent;
    }
    .hero-emoji {
        display: inline-block;
        transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .hero-emoji:hover {
        transform: rotate(-18deg) scale(1.2);
    }
    @media (prefers-reduced-motion: reduce) {
        .hero-split-text h1, .hero-emoji { transition: none; }
    }
    .hero-split-text .landing-sub {
        text-align: justify;
        text-justify: inter-word;
        max-width: 100%;
        margin-bottom: 0.9rem;
        font-size: 0.95rem;
    }
    .hero-split-text .landing-badges {
        width: 100%;
        justify-content: flex-start;
        margin-bottom: 0;
    }
    .hero-illustration-wrap {
        flex: 0.4;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .hero-illustration-card {
        position: relative;
        width: 100%;
        max-width: 210px;
        border-radius: 22px;
        overflow: hidden;
        background: linear-gradient(160deg, #f0fdfa 0%, #e0f2fe 100%);
        box-shadow: 0 10px 26px rgba(15, 118, 110, 0.16), 0 0 0 6px rgba(13,148,136,0.08);
        border: 1px solid rgba(13, 148, 136, 0.2);
        display: block;
        padding: 10px;
        animation: hero-float-3d 5s ease-in-out infinite;
        transform-style: preserve-3d;
    }
    @keyframes hero-float-3d {
        0%, 100% {
            transform: perspective(600px) translateY(0) rotateY(0deg) rotateX(0deg);
            box-shadow: 0 10px 26px rgba(15, 118, 110, 0.16), 0 0 0 6px rgba(13,148,136,0.08);
        }
        50% {
            transform: perspective(600px) translateY(-10px) rotateY(8deg) rotateX(3deg);
            box-shadow: 0 22px 38px rgba(15, 118, 110, 0.22), 0 0 0 6px rgba(13,148,136,0.08);
        }
    }
    @media (prefers-reduced-motion: reduce) {
        .hero-illustration-card { animation: none; }
        .metric-card:hover, .resource-card:hover, .about-card:hover {
            transform: translateY(-2px);
        }
    }
    .hero-illustration-card img {
        width: 100%;
        height: auto;
        display: block;
        border-radius: 14px;
    }
    @media (max-width: 900px) {
        .hero-split {
            flex-direction: column;
            text-align: center;
        }
        .hero-split-text {
            align-items: center;
        }
        .hero-split-text h1 {
            font-size: 1.6rem;
            white-space: normal;
        }
        .hero-split-text .landing-sub,
        .hero-split-text .landing-badges {
            text-align: center;
            justify-content: center;
            align-items: center;
        }
        .hero-illustration-card {
            max-width: 110px;
        }
    }

    /* ---- How It Works section ---- */
    .how-wrap {
        display: flex;
        align-items: center;
        gap: 2.5rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 20px rgba(15,23,42,0.05);
    }
    .how-img-col {
        flex: 0.8;
        display: flex;
        justify-content: center;
    }
    .how-img-col img {
        width: 100%;
        max-width: 320px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(15, 118, 110, 0.15);
    }
    .how-steps-col {
        flex: 1.2;
    }
    .how-steps-col h3 {
        color: #0d9488;
        font-size: 1.4rem;
        margin-bottom: 1rem;
    }
    .how-step {
        display: flex;
        align-items: flex-start;
        gap: 0.9rem;
        margin-bottom: 1.1rem;
    }
    .how-step-num {
        flex-shrink: 0;
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0d9488, #0284c7);
        color: #ffffff;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
    }
    .how-step-text b {
        display: block;
        color: #1e293b;
        font-size: 1rem;
        margin-bottom: 0.15rem;
    }
    .how-step-text span {
        color: #64748b;
        font-size: 0.88rem;
    }
    @media (max-width: 900px) {
        .how-wrap {
            flex-direction: column;
            padding: 1.5rem;
        }
    }

    /* ---- Try a Demo Sample section ---- */
    .demo-wrap {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 20px rgba(15,23,42,0.05);
    }
    .demo-wrap h3 {
        color: #0d9488;
        font-size: 1.2rem;
        margin-bottom: 0.3rem;
    }
    .demo-wrap p {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .st-key-demo_sample_grid {
        max-height: 300px;
        overflow-y: auto;
        padding: 0.2rem 0.6rem 0.2rem 0.2rem;
    }
    .st-key-demo_sample_grid::-webkit-scrollbar {
        width: 6px;
    }
    .st-key-demo_sample_grid::-webkit-scrollbar-track {
        background: transparent;
    }
    .st-key-demo_sample_grid::-webkit-scrollbar-thumb {
        background: rgba(13, 148, 136, 0.35);
        border-radius: 10px;
    }
    .st-key-demo_sample_grid .stButton > button {
        height: auto !important;
        padding: 0 !important;
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        overflow: hidden !important;
        background: #f8fafc !important;
    }
    .st-key-demo_sample_grid .stButton > button:hover {
        border-color: #0d9488 !important;
        transform: translateY(-2px);
    }
    .st-key-demo_sample_grid .stButton > button p {
        font-size: 0.78rem !important;
        font-weight: 600;
        color: #334155 !important;
        padding: 0.3rem 0;
    }
    .demo-thumb {
        width: 100%;
        height: 90px;
        overflow: hidden;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        background: #000;
        margin-bottom: 0.35rem;
    }
    .demo-thumb img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }


    /* Responsive tweaks for small / narrow viewports */
    @media (max-width: 900px) {
        .landing-bg-icon, .landing-logo {
            display: none;
        }
        .landing-wrap h1 {
            font-size: 2.1rem;
        }
        .st-key-top_nav_bar {
            right: 1rem !important;
            top: 3.2rem !important;
        }
        .st-key-top_brand_bar {
            display: none !important;
        }
    }

    /* ---- About section (Project Overview / Features / Tech Stack) ---- */
    .about-wrap {
        max-width: 100%;
        margin: 1.5rem 0 2rem 0;
    }
    .about-card {
        background: linear-gradient(135deg, #ffffff 0%, #f6fdfc 100%);
        border: 1px solid #d9efec;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 16px rgba(15, 118, 110, 0.06);
        transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
    }
    .about-card:hover {
        border-color: rgba(13, 148, 136, 0.35);
        box-shadow: 0 12px 26px rgba(15, 118, 110, 0.12);
        transform: perspective(900px) rotateX(1.5deg) translateY(-2px);
    }
    .about-card h3 {
        color: #0d9488;
        font-size: 1.15rem;
        font-weight: 800;
        margin: 0 0 0.9rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .about-card p {
        color: #3f5c58;
        font-size: 0.95rem;
        line-height: 1.75;
        margin: 0;
    }
    /* ---- Architecture section cards ---- */
    .arch-card {
        background: linear-gradient(135deg, #ffffff 0%, #f6fdfc 100%);
        border: 1px solid #d9efec;
        border-left: 4px solid #0d9488;
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(15, 118, 110, 0.06);
        transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
    }
    .arch-card:hover {
        border-color: rgba(13, 148, 136, 0.4);
        box-shadow: 0 8px 24px rgba(15, 118, 110, 0.12);
        transform: translateY(-2px);
    }
    .arch-card h4 {
        color: #0d9488;
        font-size: 1.08rem;
        font-weight: 800;
        margin: 0 0 0.7rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .arch-card p {
        color: #33504c;
        font-size: 0.92rem;
        line-height: 1.75;
        margin: 0 0 0.6rem 0;
    }
    .arch-card p:last-child {
        margin-bottom: 0;
    }
    .arch-card ul {
        margin: 0.3rem 0 0.6rem 0;
        padding-left: 1.2rem;
    }
    .arch-card li {
        color: #33504c;
        font-size: 0.9rem;
        line-height: 1.7;
        margin-bottom: 0.4rem;
    }
    .arch-card li:last-child {
        margin-bottom: 0;
    }
    .arch-card .arch-highlight {
        color: #0f766e;
        font-weight: 600;
    }
    /* ---- Result plot images (Performance / Classifier tabs) ---- */
    .st-key-result_plots_grid [data-testid="stImage"],
    .st-key-clf_plots_grid [data-testid="stImage"] {
        margin-bottom: 1.3rem;
    }
    .st-key-result_plots_grid [data-testid="stImage"] img,
    .st-key-clf_plots_grid [data-testid="stImage"] img {
        width: 100% !important;
        height: auto !important;
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(15,23,42,0.04);
        padding: 0.6rem;
        box-sizing: border-box;
    }
    /* ---- Numbered flow-chip pipeline (e.g. Preprocessing Pipeline) ---- */
    .flow-chip-row {
        display: flex;
        align-items: stretch;
        gap: 0.55rem;
        margin-bottom: 0.7rem;
    }
    .flow-chip {
        flex: 1 1 0;
        min-width: 0;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        gap: 0.4rem;
        background: rgba(239, 246, 255, 0.7);
        border: 1px solid #dbeafe;
        border-radius: 8px;
        padding: 0.6rem 0.7rem;
        font-size: 0.83rem;
        font-weight: 600;
        color: #1e3a8a;
        white-space: normal;
        line-height: 1.25;
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    .flow-chip:hover {
        background: #dbeafe;
        border-color: #93c5fd;
    }
    .flow-chip .chip-num {
        color: #2563eb;
        font-weight: 800;
        flex-shrink: 0;
    }
    .flow-arrow {
        flex: 0 0 auto;
        display: flex;
        align-items: center;
        color: #94a3b8;
        font-size: 1.05rem;
    }
    @media (max-width: 900px) {
        .flow-chip-row {
            flex-wrap: wrap;
        }
        .flow-chip {
            flex: 1 1 40%;
        }
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.9rem;
    }
    .feature-item {
        background: rgba(13, 148, 136, 0.04);
        border: 1px solid rgba(13, 148, 136, 0.18);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        color: #33504c;
        font-size: 0.88rem;
        line-height: 1.55;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    }
    .feature-item:hover {
        transform: translateY(-3px);
        border-color: rgba(13, 148, 136, 0.45);
        background: rgba(13, 148, 136, 0.08);
        box-shadow: 0 8px 18px rgba(13, 148, 136, 0.12);
    }
    .feature-item b {
        color: #0d9488;
        display: block;
        margin-bottom: 0.3rem;
        font-size: 0.92rem;
    }
    .tech-stack-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    .tech-groups-row {
        display: flex;
        flex-wrap: wrap;
        gap: 2.4rem;
        align-items: flex-start;
    }
    .tech-groups-row .tech-group-block {
        flex: 0 1 auto;
        min-width: 0;
    }
    .tech-pill {
        background: rgba(14, 165, 233, 0.06);
        border: 1px solid rgba(14, 165, 233, 0.28);
        border-radius: 50px;
        padding: 0.4rem 1rem;
        color: #0284c7;
        font-size: 0.82rem;
        font-weight: 600;
        transition: transform 0.15s ease, background 0.15s ease, border-color 0.15s ease;
    }
    .tech-pill:hover {
        transform: translateY(-2px);
        background: rgba(14, 165, 233, 0.12);
        border-color: rgba(14, 165, 233, 0.5);
    }
    .tech-group-label {
        color: #64827e;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
        margin: 0.9rem 0 0.5rem 0;
    }
    .tech-group-label:first-child {
        margin-top: 0;
    }

    /* ---------------------------------------------------------------
       Reduce Streamlit's large default top gap so page content (and
       the brand bar below) sits close to the nav row instead of far
       down the page. This is a real layout change, not a visual hack,
       so it doesn't clip or overshoot like a negative-offset trick can.
    --------------------------------------------------------------- */
    .block-container {
        padding-top: 3rem !important;
    }

    /* ---------------------------------------------------------------
       Top-right nav bar, pinned beside Streamlit's native 3-dot menu.
       st.container(key="top_nav_bar") gets a stable ".st-key-top_nav_bar"
       class in modern Streamlit — that's what we target here.
    --------------------------------------------------------------- */
    .st-key-top_nav_bar {
        position: fixed !important;
        top: 0.9rem !important;
        right: 1.3rem !important;
        z-index: 999999 !important;
        width: auto !important;
        background: rgba(255, 255, 255, 0.88) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        padding: 0.22rem 0.4rem !important;
        border-radius: 999px !important;
        border: 1px solid rgba(13, 148, 136, 0.18) !important;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.09) !important;
    }
    .st-key-top_nav_bar [data-testid="stHorizontalBlock"] {
        gap: 0.6rem !important;
    }
    .st-key-top_nav_bar [data-testid="stColumn"] {
        width: auto !important;
        min-width: fit-content !important;
        flex: 0 0 auto !important;
    }
    .st-key-top_nav_bar .stButton > button {
        border-radius: 999px !important;
        padding: 0.28rem 0.7rem !important;
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px;
        min-height: 0 !important;
        height: auto !important;
        white-space: nowrap !important;
        transition: all 0.18s ease !important;
    }
    .st-key-top_nav_bar .stButton > button:hover {
        transform: translateY(-2px) scale(1.03) !important;
    }
    .st-key-top_nav_bar .stButton > button:active {
        transform: translateY(0) scale(0.98) !important;
    }
    .st-key-top_nav_bar .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
        color: #1e293b !important;
        font-weight: 700 !important;
    }
    .st-key-top_nav_bar .stButton > button[kind="secondary"]:hover {
        background: rgba(13, 148, 136, 0.1) !important;
        border-color: rgba(13, 148, 136, 0.3) !important;
        color: #0d9488 !important;
    }
    .st-key-top_nav_bar .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0d9488, #0f766e) !important;
        border: 1px solid #0d9488 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.3) !important;
        text-decoration: none !important;
    }
    .st-key-top_nav_bar .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 22px rgba(13, 148, 136, 0.45) !important;
    }

    /* Collapse the empty vertical space the nav row would otherwise leave behind */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-top_nav_bar) {
        margin: 0 !important;
        min-height: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-top_brand_bar) {
        margin: 0 !important;
        min-height: 0 !important;
    }

    /* ---------------------------------------------------------------
       Top-left brand bar (logo + tagline). Lives in normal document
       flow (NOT position:fixed) so it scrolls away with the rest of
       the page instead of floating over content underneath it. The
       small negative margin-top nudges it up so it visually lines up
       with the nav row above, right at the page's top-left corner.
    --------------------------------------------------------------- */
    .st-key-top_brand_bar {
        position: static !important;
        width: auto !important;
        background: transparent !important;
        padding: 0 !important;
        margin: -5.2rem 0 0.5rem -2.5rem !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* ---------------------------------------------------------------
       Force readable dark text on plain Streamlit markdown/table
       elements that still carry the app's underlying dark-theme text
       color (this is what was showing up "invisible" on the new light
       background). Only plain prose tags are targeted here so our own
       colored badges/pills/cards (which already set their own color)
       are left alone.
    --------------------------------------------------------------- */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] b,
    [data-testid="stMarkdownContainer"] em,
    [data-testid="stMarkdownContainer"] code,
    [data-testid="stMarkdownContainer"] a,
    [data-testid="stMarkdownContainer"] blockquote {
        color: #1e293b;
    }
    [data-testid="stTable"] table,
    [data-testid="stTable"] th,
    [data-testid="stTable"] td,
    [data-testid="stTable"] span {
        color: #1e293b !important;
        background-color: #ffffff !important;
        border-color: #e2eeec !important;
    }
    [data-testid="stCaptionContainer"] {
        color: #64827e !important;
    }
    button[data-baseweb="tab"] {
        color: #4b6763 !important;
    }
    button[data-baseweb="tab"] p {
        color: inherit !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0d9488 !important;
    }

    /* ---------------------------------------------------------------
       Global theme variable overrides. Streamlit's built-in widgets
       (buttons, sliders, selectboxes, file uploader, dataframe canvas)
       read these CSS custom properties for their colors — overriding
       them here re-themes native components to match the light/teal
       palette without having to fight every widget's internal markup.
    --------------------------------------------------------------- */
    :root, .stApp {
        --primary-color: #0d9488 !important;
        --background-color: #ffffff !important;
        --secondary-background-color: #eef8f6 !important;
        --text-color: #1e293b !important;
    }

    /* Inline <code> spans inside markdown (previously dark-on-dark / invisible) */
    [data-testid="stMarkdownContainer"] code {
        background-color: #eef8f6 !important;
        color: #0f766e !important;
        border: 1px solid #d9efec;
        border-radius: 4px;
        padding: 0.1rem 0.35rem;
    }

    /* ---- Model Info "tabs" (actually plain buttons — see comment at
       its call site for why) styled to look like a pill tab bar ---- */
    .st-key-model_info_tab_bar [data-testid="stHorizontalBlock"] {
        gap: 0.5rem !important;
        flex-wrap: wrap;
    }
    .st-key-model_info_tab_bar .stButton > button {
        border-radius: 999px !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
    }
    .st-key-model_info_tab_bar .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        border: 1px solid #d9efec !important;
        color: #0f172a !important;
    }
    .st-key-model_info_tab_bar .stButton > button[kind="secondary"]:hover {
        border-color: #0d9488 !important;
        background: rgba(13, 148, 136, 0.05) !important;
    }
    .st-key-model_info_tab_bar .stButton > button[kind="primary"] {
        background: #0d9488 !important;
        border-color: #0d9488 !important;
    }

    /* Generic buttons (Generate AI Report, Start Detection, etc.) —
       nav bar buttons keep their own scoped rules above and win by
       specificity, so this only affects the rest of the app.
       Layered box-shadow gives a "raised ledge" 3D look; the ledge
       shortens on press so the button reads as physically pushed in. */
    .stButton > button, .stDownloadButton > button {
        background-color: #ffffff !important;
        color: #0f766e !important;
        border: 1px solid #0d9488 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: 0 3px 0 rgba(13, 148, 136, 0.35), 0 5px 10px rgba(15, 118, 110, 0.12) !important;
        transform: translateY(0);
        transition: transform 0.1s ease, box-shadow 0.1s ease, background-color 0.15s ease, border-color 0.15s ease !important;
    }
    .stButton > button p, .stDownloadButton > button p {
        font-weight: 700 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: rgba(13, 148, 136, 0.08) !important;
        border-color: #0f766e !important;
        color: #0f766e !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 0 rgba(13, 148, 136, 0.35), 0 10px 18px rgba(15, 118, 110, 0.18) !important;
    }
    .stButton > button:active, .stDownloadButton > button:active {
        transform: translateY(2px);
        box-shadow: 0 1px 0 rgba(13, 148, 136, 0.35), 0 2px 6px rgba(15, 118, 110, 0.14) !important;
    }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        background: linear-gradient(135deg, #14b8a6, #0d9488) !important;
        color: #ffffff !important;
        border: 1px solid #0d9488 !important;
        box-shadow: 0 4px 0 #0b5f57, 0 8px 16px rgba(13, 148, 136, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 0 #0b5f57, 0 14px 22px rgba(13, 148, 136, 0.4) !important;
    }
    .stButton > button[kind="primary"]:active, .stDownloadButton > button[kind="primary"]:active {
        box-shadow: 0 1px 0 #0b5f57, 0 3px 8px rgba(13, 148, 136, 0.3) !important;
    }
    @media (prefers-reduced-motion: reduce) {
        .stButton > button, .stDownloadButton > button { transition: background-color 0.15s ease, border-color 0.15s ease !important; }
    }

    /* File uploader dropzone + its internal "Browse files" button */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f7fbfa !important;
        border: 2px dashed #0d9488 !important;
        border-radius: 16px !important;
        padding: 1.6rem 1rem !important;
        transition: background-color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease !important;
    }
    [data-testid="stFileUploaderDropzone"] > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem !important;
        text-align: center !important;
    }
    [data-testid="stFileUploaderDropzone"] svg {
        width: 34px !important;
        height: 34px !important;
        color: #0d9488 !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: rgba(13, 148, 136, 0.06) !important;
        border-color: #0f766e !important;
        box-shadow: 0 10px 24px rgba(15, 118, 110, 0.12) !important;
        transform: translateY(-2px);
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #33504c !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #ffffff !important;
        color: #0f766e !important;
        border: 1px solid #0d9488 !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.3rem !important;
    }
    [data-testid="stFileUploaderFile"] {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border-radius: 10px !important;
    }
    @media (prefers-reduced-motion: reduce) {
        [data-testid="stFileUploaderDropzone"]:hover { transform: none; }
    }

    /* Selectbox (e.g. the report "Layout" dropdown) */
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #d9efec !important;
        color: #1e293b !important;
    }
    [data-baseweb="popover"] li {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }

    /* Sliders — soften the default red to teal.
       Streamlit bakes the accent color into each part's inline
       `style` (sometimes as a solid rgb(), sometimes as a
       linear-gradient using that same rgb()) — matching on the
       bare digit sequence catches both forms. Using the `background`
       shorthand (not just background-color) also overrides any
       gradient so no red edge is left peeking through. */
    div[data-baseweb="slider"] div[style*="255, 75, 75"],
    div[data-baseweb="slider"] div[style*="255,75,75"],
    div[data-baseweb="slider"] div[style*="255, 43, 43"],
    div[data-baseweb="slider"] div[style*="255,43,43"],
    div[data-baseweb="slider"] div[style*="#ff4b4b"] {
        background: #0d9488 !important;
        background-color: #0d9488 !important;
    }
    div[data-baseweb="slider"] [role="slider"][style*="255, 75, 75"],
    div[data-baseweb="slider"] [role="slider"][style*="255,75,75"],
    div[data-baseweb="slider"] [role="slider"][style*="255, 43, 43"],
    div[data-baseweb="slider"] [role="slider"][style*="255,43,43"],
    div[data-baseweb="slider"] [role="slider"][style*="#ff4b4b"] {
        background: #0d9488 !important;
        background-color: #0d9488 !important;
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.15) !important;
    }
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
        color: #64827e !important;
        background: transparent !important;
    }
    [data-testid="stSliderThumbValue"] {
        color: #0f766e !important;
    }

    /* Charts (bar chart etc.) — light container so axis/text stay readable */
    [data-testid="stVegaLiteChart"], .vega-embed {
        background-color: #ffffff !important;
    }
    [data-testid="stVegaLiteChart"] text, .vega-embed text {
        fill: #33504c !important;
    }

    /* ---- Severity assessment box (was inline-styled at its call site) ---- */
    .severity-box {
        background: #f7fbfa;
        border: 1px solid #e2eeec;
        border-left: 4px solid var(--severity-color, #0d9488);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0;
    }
    .severity-box .severity-title {
        color: var(--severity-color, #0d9488);
        font-weight: 700;
    }
    .severity-box .severity-body {
        color: #33504c;
        font-size: 0.95rem;
    }

    /* ---- Model-agreement comparison card (was inline-styled) ---- */
    .agreement-card {
        background: var(--agreement-bg, rgba(14, 165, 233, 0.08));
        border: 1px solid var(--agreement-border, #0369a1);
        border-left: 4px solid var(--agreement-border, #0369a1);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
        container-type: inline-size;
        container-name: agreement-card;
    }
    .agreement-card .agreement-headline {
        color: var(--agreement-border, #0369a1);
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.6rem;
    }
    .agreement-card .agreement-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .agreement-card .agreement-item {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid #e2eeec;
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
        flex: 1;
        min-width: 120px;
    }
    .agreement-card .agreement-item-label {
        color: #4f6b67;
        font-size: 0.72rem;
        text-transform: uppercase;
    }
    .agreement-card .agreement-item-value {
        color: #1e293b;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .agreement-card .agreement-note {
        color: #92650c;
        font-size: 0.85rem;
        margin-top: 0.7rem;
    }
    /* Stack the two stat items if the card itself gets squeezed narrow */
    @container agreement-card (max-width: 260px) {
        .agreement-card .agreement-row { flex-direction: column; }
        .agreement-card .agreement-item { min-width: 0; }
    }

    /* ---- Quick confidence stats row (Max/Min/Threshold — was inline-styled) ---- */
    .quickstat-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 0.8rem;
        container-type: inline-size;
        container-name: quickstat-row;
    }
    .quickstat-item {
        background: #f0f9ff;
        border: 1px solid #d8ecf7;
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
        flex: 1;
        min-width: 100px;
        text-align: center;
    }
    .quickstat-item .quickstat-label {
        color: #0284c7;
        font-size: 0.78rem;
        text-transform: uppercase;
    }
    .quickstat-item .quickstat-value {
        color: #1e293b;
        font-weight: 700;
    }
    @container quickstat-row (max-width: 260px) {
        .quickstat-row { flex-direction: column; }
        .quickstat-item { min-width: 0; }
    }

    /* ---- Scroll-driven reveal: cards fade + rise into place as you scroll
       past them, instead of just sitting there statically. Pure CSS (no JS),
       progressively enhanced — browsers without animation-timeline support
       simply show the cards normally with no animation. ---- */
    @supports (animation-timeline: view()) {
        @keyframes scroll-reveal-in {
            from { opacity: 0; transform: translateY(28px) scale(0.97); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        .resource-card, .metric-card, .arch-card, .about-card, .agreement-card {
            animation: scroll-reveal-in linear both;
            animation-timeline: view();
            animation-range: entry 0% entry 55%;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        .resource-card, .metric-card, .arch-card, .about-card, .agreement-card {
            animation: none !important;
        }
    }

    /* ---- CSS-only tooltip: hover/focus a metric card with a data-tip
       attribute to see what the metric means. No JS required. ---- */
    .metric-card[data-tip] {
        position: relative;
        cursor: help;
    }
    .metric-card[data-tip]::after {
        content: attr(data-tip);
        position: absolute;
        left: 50%;
        bottom: calc(100% + 10px);
        transform: translateX(-50%) translateY(4px);
        width: max-content;
        max-width: 220px;
        background: #0f172a;
        color: #f0fdfa;
        font-size: 0.72rem;
        font-weight: 500;
        line-height: 1.4;
        text-transform: none;
        letter-spacing: normal;
        padding: 0.5rem 0.7rem;
        border-radius: 8px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.25);
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.15s ease, transform 0.15s ease;
        z-index: 20;
    }
    .metric-card[data-tip]::before {
        content: "";
        position: absolute;
        left: 50%;
        bottom: 100%;
        transform: translateX(-50%);
        border: 6px solid transparent;
        border-top-color: #0f172a;
        margin-bottom: -2px;
        opacity: 0;
        transition: opacity 0.15s ease;
        z-index: 20;
    }
    .metric-card[data-tip]:hover::after, .metric-card[data-tip]:focus-within::after {
        opacity: 1;
        transform: translateX(-50%) translateY(0);
    }
    .metric-card[data-tip]:hover::before, .metric-card[data-tip]:focus-within::before {
        opacity: 1;
    }
</style>
""",
    unsafe_allow_html=True,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "classifier.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "runs", "detect", "train")
CLASSIFIER_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "final_model")
KFOLD_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "kfold_runs")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def _sanitize_llm_html(text: str) -> str:
    """The AI report is rendered with unsafe_allow_html=True so its Markdown
    formatting (headers, bold, lists) shows up properly. That also means any
    literal HTML the model includes would be rendered as-is. Since the model
    output isn't otherwise validated, strip anything that could execute code
    or load external content before it ever reaches st.markdown — this is a
    pragmatic regex-based pass (no extra dependency required), not a full
    HTML parser, but it removes the categories of tags/attributes that
    actually matter here (scripts, iframes, event handlers, javascript: URIs)."""
    if not text:
        return text
    # Strip whole dangerous elements, including their content.
    text = re.sub(r"(?is)<(script|style|iframe|object|embed|link|meta)\b.*?(</\1\s*>|$)", "", text)
    # Strip inline event-handler attributes like onclick="...", onerror='...'.
    text = re.sub(r"(?is)\son\w+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", "", text)
    # Neutralize javascript:/data: URIs in href/src attributes.
    text = re.sub(r"(?is)(href|src)\s*=\s*(\"|')\s*(javascript|data):[^\"']*\2", r"\1=\2#\2", text)
    return text


@st.cache_data(show_spinner=False)
def get_classifier_metrics():
    """Reads the final row of the classifier's results.csv (training-time
    accuracy curve) and returns a dict of the metrics actually achieved
    during training (no hardcoded numbers)."""
    csv_path = os.path.join(CLASSIFIER_RESULTS_DIR, "results.csv")
    if not os.path.isfile(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]

        def find_col(*keywords):
            for col in df.columns:
                if all(k.lower() in col.lower() for k in keywords):
                    return col
            return None

        top1_col = find_col("top1")
        loss_col = find_col("loss") or find_col("train", "loss")
        epoch_col = find_col("epoch")

        return {
            "top1": float(last[top1_col]) if top1_col else None,
            "final_loss": float(last[loss_col]) if loss_col else None,
            "epochs_run": int(last[epoch_col]) if epoch_col else len(df),
        }
    except Exception:
        logger.warning("get_classifier_metrics: failed to parse %s", csv_path, exc_info=True)
        return None


def compute_binary_extra_metrics(confusion_matrix_2x2):
    """Computes Balanced Accuracy and Matthews Correlation Coefficient (MCC)
    directly from a saved 2x2 confusion matrix — no retraining needed, these
    are just different summaries of the same held-out test predictions."""
    try:
        cm = np.array(confusion_matrix_2x2, dtype=float)
        if cm.shape != (2, 2):
            return None
        tn, fp = cm[0]
        fn, tp = cm[1]
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        balanced_acc = (recall_0 + recall_1) / 2

        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

        return {"balanced_accuracy": balanced_acc, "mcc": mcc}
    except Exception:
        logger.warning("compute_binary_extra_metrics: failed to compute from confusion matrix", exc_info=True)
        return None


@st.cache_data(show_spinner=False)
def get_classifier_test_metrics():
    """Reads the final held-out test evaluation (accuracy / macro precision,
    recall, F1, confusion matrix) saved by the training notebook to
    final_test_results.json — this is the untouched 335-image test set
    result, not a training/validation number, so no hardcoded figures."""
    path = os.path.join(CLASSIFIER_RESULTS_DIR, "final_test_results.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        logger.warning("get_classifier_test_metrics: failed to parse %s", path, exc_info=True)
        return None


@st.cache_data(show_spinner=False)
def get_classifier_cv_metrics():
    """Reads the 5-fold cross-validation summary saved by the training
    notebook to kfold_runs/fold_results.json and returns the per-fold
    scores plus their mean/std — this is what demonstrates the model's
    result is stable and not just a lucky single split."""
    path = os.path.join(KFOLD_RESULTS_DIR, "fold_results.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            fold_results = json.load(f)
        values = list(fold_results.values())
        if not values:
            return None
        return {
            "folds": fold_results,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    except Exception:
        logger.warning("get_classifier_cv_metrics: failed to parse %s", path, exc_info=True)
        return None


@st.cache_data(show_spinner=False)
def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        logger.warning("get_base64_image: failed to read %s", path, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Home-page fixed background image. Stays in place while the page scrolls
# (background-attachment: fixed) and shows through crisp/un-blurred in the
# gaps between sections; the card-like sections themselves (hero, How It
# Works, About cards) get a frosted-glass treatment — semi-transparent
# white + backdrop-filter blur — so the same photo appears softly blurred
# behind their text instead, keeping it readable while still visually
# tying the card to the background behind it.
# ---------------------------------------------------------------------------
_home_bg_b64 = None
_home_bg_mime = "image/jpeg"
for _home_bg_name, _home_bg_mime_candidate in (
    ("home_bg.jpg", "image/jpeg"),
    ("home_bg.jpeg", "image/jpeg"),
    ("home_bg.png", "image/png"),
):
    _home_bg_b64 = get_base64_image(os.path.join(ASSETS_DIR, _home_bg_name))
    if _home_bg_b64:
        _home_bg_mime = _home_bg_mime_candidate
        break
if _home_bg_b64 and st.session_state.nav_page == "home":
    st.markdown(
        f"""<style>
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.55), rgba(255, 255, 255, 0.55)), url("data:{_home_bg_mime};base64,{_home_bg_b64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background: transparent !important;
    }}
    .how-wrap, .about-card, .demo-wrap {{
        background: rgba(255, 255, 255, 0.62) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }}
    </style>""",
        unsafe_allow_html=True,
    )


MODEL_INFO = {
    "name": "YOLO26x",
    "variant": "xl (extra-large)",
    "epochs": 150,
    "precision": 90.77,
    "recall": 86.77,
    "map50": 91.03,
    "map50_95": 55.94,
    "f1": 88.72,
}

DET_TEST_METRICS = {
    "n_images": 246,
    "precision": 97.19,
    "recall": 92.51,
    "map50": 95.02,
    "map50_95": 73.59,
    "f1": 94.79,
}

# Fallback values shown ONLY if runs/detect/train/args.yaml isn't found on
# disk (e.g. a deployment that ships without the training run folder).
# Edit these to match your actual training run if that's the case.
TRAIN_CONFIG_FALLBACK = {
    "epochs": MODEL_INFO["epochs"],
    "batch": 16,
    "imgsz": 640,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "patience": 100,
    "seed": 0,
    "device": "0",
    "workers": 8,
}


def find_first_existing(*filenames, search_dir=None):
    """Returns the first existing file path among the filenames, or None.
    Searches RESULTS_DIR by default; pass search_dir to look elsewhere."""
    base_dir = search_dir if search_dir is not None else RESULTS_DIR
    for name in filenames:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            return path
    return None


@st.cache_data(show_spinner=False)
def load_train_config(search_dir=None):
    """Reads the hyperparameters Ultralytics auto-saves as args.yaml in the
    training run folder (e.g. runs/detect/train/args.yaml). Returns a dict
    of the fields we care about, or None if the file isn't found — the UI
    falls back to a manual TRAIN_CONFIG dict in that case."""
    base_dir = search_dir if search_dir is not None else RESULTS_DIR
    path = os.path.join(base_dir, "args.yaml")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        logger.warning("load_train_config: failed to parse %s", path, exc_info=True)
        return None

    keys = [
        "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf",
        "momentum", "weight_decay", "patience", "seed", "device", "workers",
    ]
    return {k: raw[k] for k in keys if k in raw}



@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_classifier():
    if not os.path.isfile(CLASSIFIER_PATH):
        return None
    return YOLO(CLASSIFIER_PATH)


def run_classifier(classifier, img_array):
    """Runs the stone/non_stone classifier on the full image and returns the
    predicted probability that a stone is present, or None if unavailable."""
    if classifier is None:
        return None
    try:
        clf_results = classifier.predict(source=img_array, verbose=False)
        probs = clf_results[0].probs
        names = clf_results[0].names
        stone_idx = next((idx for idx, name in names.items() if name == "stone"), None)
        if stone_idx is None:
            return None
        return float(probs.data[stone_idx])
    except Exception:
        logger.warning("run_classifier: classification inference failed", exc_info=True)
        return None


@st.cache_data(show_spinner=False)
def run_detection_cached(_model, image_bytes, conf, iou):
    """Runs YOLO detection and returns plain-data results (not the raw
    ultralytics Results object, which isn't reliably cacheable). Cached on
    (image bytes, conf, iou) so re-running the same image at the same
    thresholds — e.g. an unrelated widget triggering a rerun, or the user
    moving a slider back to a value they already tried — is instant instead
    of re-running the full model forward pass."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    start = time.perf_counter()
    results = _model.predict(source=arr, conf=conf, iou=iou, verbose=False)
    elapsed = time.perf_counter() - start

    result = results[0]
    boxes = result.boxes
    detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            box_conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = _model.names.get(cls, f"class_{cls}")
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            area_pct = (w * h) / (img.width * img.height) * 100
            detections.append({
                "Detection #": len(detections) + 1,
                "Label": label,
                "Confidence": box_conf,
                "Conf %": f"{box_conf*100:.1f}%",
                "Area %": f"{area_pct:.2f}%",
                "Location (x1,y1)": f"({int(x1)}, {int(y1)})",
                "Size (WxH)": f"{int(w)}×{int(h)} px",
            })
    # Rendered here (inside the cache) since the raw ultralytics Results
    # object itself can't be reliably cached/returned — only the drawn
    # RGB array (plain numpy data) survives Streamlit's cache round-trip.
    annotated_rgb = result.plot()[..., ::-1] if detections else None
    return detections, elapsed, annotated_rgb


@st.cache_data(show_spinner=False)
def run_classifier_cached(_classifier, image_bytes):
    """Cached wrapper around run_classifier, keyed only on the image bytes
    (the classifier doesn't take conf/iou thresholds)."""
    img_array = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    return run_classifier(_classifier, img_array)


@st.cache_resource(show_spinner=False)
def get_ai_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"], timeout=30.0)


def generate_ai_report(stone_count, avg_conf, severity_label, detections, clf_stone_prob=None, classifier_threshold=None):
    """Sends detection summary to an LLM and returns a generated medical-style report."""
    client = get_ai_client()

    detection_lines = (
        "\n".join([
            f"- Stone #{d['Detection #']}: confidence {d['Conf %']},"
            f" area {d['Area %']}, size {d['Size (WxH)']},"
            f" location {d['Location (x1,y1)']}"
            for d in detections
        ])
        if detections
        else "No stones detected."
    )

    if clf_stone_prob is not None and classifier_threshold is not None:
        clf_flag = "stone likely present" if clf_stone_prob >= classifier_threshold else "no stone likely"
        agrees = (stone_count > 0) == (clf_stone_prob >= classifier_threshold)
        classifier_line = (
            f"- Classification model result: {clf_flag} "
            f"({clf_stone_prob*100:.1f}% stone probability, threshold "
            f"{classifier_threshold*100:.0f}%) — {'agrees' if agrees else 'disagrees'} with the detection model."
        )
    else:
        classifier_line = "- Classification model result: not available for this image."

    prompt = f"""
You are a radiology assistant. Based on the following kidney stone detection
results, write a professional medical report in Markdown format:

- Stones found: {stone_count}
- Average confidence: {avg_conf*100:.1f}%
- Severity: {severity_label}
- Per-stone details:
{detection_lines}
{classifier_line}

The report should have three sections:
1. Findings
2. Impression
3. Recommendation

Briefly mention the classification model's result in the Findings
section — note whether it agrees or disagrees with the detection model, since
a disagreement is clinically relevant and worth flagging for review.

Do not use Markdown tables (lines with "|" or "---" separators) anywhere in
the report — this text is also rendered into a PDF that cannot lay out
tables, so use plain bullet or numbered lists for the per-stone details
instead.

End with a disclaimer stating that this report is for informational purposes
only, is not a final medical diagnosis, and that a qualified radiologist or
urologist should be consulted for an actual clinical decision.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# LLMs commonly reach for "nicer" typographic punctuation (en-dashes, smart
# quotes, non-breaking/thin spaces, bullet dots) that the core PDF fonts'
# Latin-1 encoding can't represent — those were silently turning into a
# literal "?" wherever they appeared. Normalizing them to their plain-ASCII
# equivalents first means the "?" fallback is only ever hit by truly
# unsupported characters (which should now be rare).
_PDF_UNICODE_NORMALIZE = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2026": "...",
    "\u2022": "-", "\u2023": "-", "\u25cf": "-", "\u25e6": "-",
    "\u2000": " ", "\u2001": " ", "\u2002": " ", "\u2003": " ",
    "\u2004": " ", "\u2005": " ", "\u2006": " ", "\u2007": " ",
    "\u2008": " ", "\u2009": " ", "\u200a": " ", "\u202f": " ",
    "\u00a0": " ", "\u200b": "",
}


def _pdf_safe(text: str) -> str:
    """Strips markdown symbols and forces text into Latin-1 (core PDF fonts only support Latin-1)."""
    text = re.sub(r"[#*`_]", "", text)
    for uni_char, ascii_eq in _PDF_UNICODE_NORMALIZE.items():
        text = text.replace(uni_char, ascii_eq)
    return text.encode("latin-1", "replace").decode("latin-1")


def _prepare_report_text_for_pdf(text: str) -> str:
    """Makes LLM-generated Markdown safe for FPDF's multi_cell to wrap.

    multi_cell wraps on whitespace only — a Markdown table row like
    "| Stone | Confidence | ... |" is fine (it has spaces), but the
    separator row "|-------|-----------|...-|" is one long token with no
    spaces at all, and multi_cell crashes with "Not enough horizontal
    space to render a single character" when it hits a token wider than
    the page. This converts table rows into plain space-separated text
    and, as a safety net, force-breaks any other very long unbroken run
    of characters so nothing can trigger that crash again.
    """
    out_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        # Drop markdown table separator rows like "|---|:--:|---|"
        if stripped and re.fullmatch(r"\|?[\s:\-|]+\|?", stripped) and "|" in stripped:
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            # "| a | b | c |" -> "a   |   b   |   c" (space-padded so it can wrap)
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            line = "   |   ".join(cells)
        out_lines.append(line)
    text = "\n".join(out_lines)

    # Belt-and-suspenders: break up any run of 40+ non-space characters
    # (long URLs/ids/etc.) so it can never again be a single unbreakable word.
    def _break_long_token(match):
        token = match.group(0)
        return " ".join(token[i:i + 40] for i in range(0, len(token), 40))

    return re.sub(r"\S{41,}", _break_long_token, text)


def _write_wrapped(pdf, text, width_chars=95, line_height=6):
    """Writes text to the PDF pre-wrapped by Python's textwrap instead of
    relying on FPDF's own multi_cell() word-wrap.

    Two separate FPDF quirks were causing "Not enough horizontal space to
    render a single character" crashes here:
    1. multi_cell()'s own internal wrapping is fragile with certain
       content (long unbroken tokens, Markdown table separator rows) —
       pre-wrapping with textwrap (which always force-splits on `width`)
       sidesteps that entirely.
    2. multi_cell() can leave the cursor's x position sitting near the
       right page margin afterwards instead of resetting it, so the
       *next* multi_cell() call starts with almost no room left and
       throws the same error even for perfectly short text. Explicitly
       resetting x to the left margin after every line closes that gap.
    """
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            pdf.ln(line_height / 2)
            pdf.set_x(pdf.l_margin)
            continue
        wrapped = textwrap.wrap(
            paragraph, width=width_chars, break_long_words=True, break_on_hyphens=False
        ) or [""]
        for line in wrapped:
            pdf.multi_cell(0, line_height, line)
            pdf.set_x(pdf.l_margin)


def build_pdf_report(filename, stone_count, avg_conf, severity_label, detections, ai_report_text):
    """Builds a downloadable PDF summarizing detection results and the AI-generated report."""
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(13, 148, 136)
    pdf.cell(0, 10, "Kidney Stone Detection Report", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _pdf_safe(f"File: {filename}")[:90], ln=True)
    pdf.cell(0, 6, _pdf_safe(f"Stones Found: {stone_count}"), ln=True)
    pdf.cell(0, 6, _pdf_safe(f"Average Confidence: {avg_conf*100:.1f}%"), ln=True)
    pdf.cell(0, 6, _pdf_safe(f"Severity: {severity_label}"), ln=True)
    pdf.ln(4)

    if detections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Detected Stones", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for d in detections:
            line = (
                f"Stone #{d['Detection #']}: confidence {d['Conf %']}, "
                f"area {d['Area %']}, size {d['Size (WxH)']}, "
                f"location {d['Location (x1,y1)']}"
            )
            _write_wrapped(pdf, _pdf_safe(line))
        pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "AI-Generated Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    _write_wrapped(pdf, _pdf_safe(_prepare_report_text_for_pdf(ai_report_text)))
    pdf.ln(6)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    _write_wrapped(
        pdf,
        _pdf_safe(
            "Disclaimer: This report is generated for informational and educational "
            "purposes only. It is not a certified medical diagnosis. Please consult a "
            "qualified radiologist or urologist for an actual clinical decision."
        ),
        width_chars=110,
        line_height=5,
    )

    pdf_output = pdf.output(dest="S")
    # Different fpdf/fpdf2 versions return different types here: older
    # versions return a Latin-1-encoded str, newer ones return
    # bytes/bytearray directly. bytes(a_str) with no encoding argument is
    # exactly what raised "string argument without an encoding" here.
    if isinstance(pdf_output, str):
        return pdf_output.encode("latin-1", "replace")
    return bytes(pdf_output)


def get_severity(count: int, avg_conf: float) -> tuple[str, str]:
    if count == 0:
        return "No Stones Detected", "#27ae60"
    elif count == 1:
        if avg_conf >= 0.75:
            return "Single Stone – High Confidence", "#e67e22"
        else:
            return "Single Stone – Possible", "#f39c12"
    elif count <= 3:
        return "Multiple Stones – Moderate", "#e74c3c"
    else:
        return "Multiple Stones – Severe", "#c0392b"


PROJECT_LINKS = {
    # ---- TODO: paste your real URLs here once ready — nothing else in ----
    # ---- the app needs to change, these are used everywhere below.    ----
    "github_repo": "https://github.com/iste-AK00/Yolo_Based_Kidney_Detection_System",
    "kaggle_bbox_dataset": "https://www.kaggle.com/datasets/safurahajiheidari/kidney-stone-images",
    "roboflow_dataset": "https://universe.roboflow.com/east-west-university-9frzq/kidney-stone-detection-wfjba",
    "kaggle_classifier_dataset": "https://www.kaggle.com/datasets/orvile/axial-ct-imaging-dataset-kidney-stone-detection",
}

CREATOR_LINKS = {
    # ---- TODO: paste each person's GitHub / LinkedIn profile URL here. ----
    # ---- Leave as "#" for anyone who doesn't have one ready yet —      ----
    # ---- their name will just show with no links until you fill it in. ----
    "Ishteuk Ahmed Joy": {"github": "#", "linkedin": "#"},
    "Israt Jahan": {"github": "#", "linkedin": "#"},
    "Dipayon Nag": {"github": "#", "linkedin": "#"},
    "Tanim": {"github": "#", "linkedin": "#"},
}


def _build_creator_chip(name, links):
    gh = links.get("github", "#")
    li = links.get("linkedin", "#")
    parts = []
    if gh and gh != "#":
        parts.append(f'<a href="{gh}" target="_blank">GitHub</a>')
    if li and li != "#":
        parts.append(f'<a href="{li}" target="_blank">LinkedIn</a>')
    links_html = f'<span class="creator-links">{" · ".join(parts)}</span>' if parts else ""
    return f'<span class="creator-name">{name}{links_html}</span>'


_CREATOR_CHIPS_HTML = "".join(_build_creator_chip(n, l) for n, l in CREATOR_LINKS.items())

FOOTER_HTML = f"""
<div class="footer-box">
    ⚕️ <b>Medical Disclaimer:</b> This application is intended for research and educational
    purposes only. It does not constitute medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare professional for medical decisions.
    <div class="footer-creators">
        <span class="creators-title">👥 Project Created By</span>
        {_CREATOR_CHIPS_HTML}
    </div>
</div>
"""


# ---------------------------------------------------------------------------
# Navigation state & UI (fixed, top-right, beside Streamlit's 3-dot menu)
# ---------------------------------------------------------------------------
def go_to(page: str):
    st.session_state.nav_page = page
    st.rerun()


nav_bar = st.container(key="top_nav_bar")
with nav_bar:
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button(
            "Home",
            key="nav_home",
            type="primary" if st.session_state.nav_page == "home" else "secondary",
        ):
            go_to("home")
    with nav_col2:
        if st.button(
            "Model Info",
            key="nav_model",
            type="primary" if st.session_state.nav_page == "model" else "secondary",
        ):
            go_to("model")
    with nav_col3:
        if st.button(
            "Upload & Detect",
            key="nav_upload",
            type="primary" if st.session_state.nav_page == "upload" else "secondary",
        ):
            go_to("upload")


# ---------------------------------------------------------------------------
# PAGE: HOME (landing)
# ---------------------------------------------------------------------------
if st.session_state.nav_page == "home":
    kidney_bg_b64 = get_base64_image(os.path.join(ASSETS_DIR, "kidney_bg.png"))
    how_img_b64 = get_base64_image(os.path.join(ASSETS_DIR, "how_it_works.jpg"))

    # Brand tagline — normal (non-fixed) block so it scrolls away with the
    # rest of the page instead of floating over content. Shown only here,
    # right above the hero card, on the home page.
    if kidney_bg_b64:
        brand_bar = st.container(key="top_brand_bar")
        with brand_bar:
            st.markdown(
                f'''<div class="landing-topbar">
<div class="topbar-icon-chip"><img src="data:image/png;base64,{kidney_bg_b64}" /></div>
<span>Small Organs,<br>Big Responsibility</span>
</div>''',
                unsafe_allow_html=True,
            )

    clf_home_metrics = get_classifier_test_metrics()
    if clf_home_metrics and clf_home_metrics.get("accuracy") is not None:
        clf_stat_value = f"{clf_home_metrics['accuracy'] * 100:.2f}%"
    else:
        clf_stat_value = "N/A"
    clf_stat_html = f'<div class="hero-stat"><div class="hero-stat-value">{clf_stat_value}</div><div class="hero-stat-label">🧠 Classifier Accuracy</div></div>'

    st.markdown(
        f"""<div class="landing-wrap" onmousemove="var r=this.getBoundingClientRect();this.style.setProperty('--mx',((event.clientX-r.left)/r.width*100)+'%');this.style.setProperty('--my',((event.clientY-r.top)/r.height*100)+'%');">
<div class="hero-centered">
<div class="hero-pill">✨ AI-assisted kidney stone detection</div>
<h1 class="hero-big-title"><span class="title-dark">Kidney Stone Detection</span><br><span class="title-accent">AI-Powered Analysis with Reporting.</span></h1>
<p class="hero-subtitle-centered">An AI-assisted diagnostic support tool that combines a custom-trained YOLO object-detection model with an classification model to identify kidney stones from medical images (CT scans), and generates automated radiology reports via LLMs.</p>
<div class="hero-stat-bar">
<div class="hero-stat"><div class="hero-stat-value">97%</div><div class="hero-stat-label">🎯 Detection Precision</div></div>
<div class="hero-stat"><div class="hero-stat-value">2,659</div><div class="hero-stat-label">🎯 Detection Images</div></div>
<div class="hero-stat"><div class="hero-stat-value">3,364</div><div class="hero-stat-label">🧠 Classifier Images</div></div>
{clf_stat_html}
<div class="hero-stat"><div class="hero-stat-value">5-Fold</div><div class="hero-stat-label">🧠 Cross-Validated</div></div>
</div>
<div class="hero-feature-tags">
<span class="feature-tag">🧬 YOLO26x Model</span>
<span class="feature-tag">🧠 YOLO11n-cls Classifier</span>
<span class="feature-tag">⚡ Real-Time Detection</span>
</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    # Primary call-to-action — docked onto the bottom of the hero card
    # (see .st-key-hero_cta_dock) so it reads as part of the same card
    # instead of floating separately over the background art below it.
    with st.container(key="hero_cta_dock"):
        cta_top1, cta_top2, cta_top3 = st.columns([1, 1, 1])
        with cta_top2:
            if st.button("🚀 Start Detection", use_container_width=True, type="primary", key="cta_start"):
                go_to("upload")

    st.markdown("<br>", unsafe_allow_html=True)

    _sample_dir = os.path.join(ASSETS_DIR, "samples")
    _sample_files = []
    if os.path.isdir(_sample_dir):
        _sample_files = sorted(
            f for f in os.listdir(_sample_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )[:6]

    if _sample_files:
        st.markdown(
            """<div class="demo-wrap">
<h3>🧪 Try a Demo Sample</h3>
<p>No CT scan handy? Pick one of the sample images below to instantly run detection — no upload needed.</p>
</div>""",
            unsafe_allow_html=True,
        )
        with st.container(key="demo_sample_grid"):
            cols = st.columns(3)
            for i, fname in enumerate(_sample_files):
                with cols[i % 3]:
                    _ext = fname.lower().rsplit(".", 1)[-1]
                    _mime = "image/png" if _ext == "png" else "image/jpeg"
                    _thumb_b64 = get_base64_image(os.path.join(_sample_dir, fname))
                    if _thumb_b64:
                        st.markdown(
                            f'<div class="demo-thumb">'
                            f'<img src="data:{_mime};base64,{_thumb_b64}" /></div>',
                            unsafe_allow_html=True,
                        )
                    if st.button(f"Sample {i + 1}", key=f"demo_sample_{i}", use_container_width=True):
                        st.session_state["selected_sample"] = os.path.join(_sample_dir, fname)
                        go_to("upload")

    if how_img_b64:
        st.markdown(
            f"""<div class="how-wrap">
<div class="how-img-col"><img src="data:image/png;base64,{how_img_b64}" /></div>
<div class="how-steps-col">
<h3>How It Works</h3>
<div class="how-step">
<div class="how-step-num">1</div>
<div class="how-step-text"><b>Upload a CT Scan</b><span>Upload an abdominal CT slice (JPG/PNG) through the Upload &amp; Detect page.</span></div>
</div>
<div class="how-step">
<div class="how-step-num">2</div>
<div class="how-step-text"><b>Two Models Analyze It</b><span>A YOLO26x detector localizes stones with bounding boxes, while a separate classification model judges the full image.</span></div>
</div>
<div class="how-step">
<div class="how-step-num">3</div>
<div class="how-step-text"><b>Get an AI Report</b><span>Review confidence, severity, and a downloadable AI-generated radiology-style report as a PDF.</span></div>
</div>
</div>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """<div class="demo-wrap">
<h3>🔄 End-to-End Pipeline</h3>
<p>From the moment you upload a scan to the moment you get a report.</p>
<div class="flow-chip-row">
    <span class="flow-chip"><span class="chip-num">01</span> Upload CT Scan</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip"><span class="chip-num">02</span> Configure Detection Settings</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip"><span class="chip-num">03</span> Preprocessing</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip"><span class="chip-num">04</span> YOLO26x Detection</span>
</div>
<div class="flow-chip-row">
    <span class="flow-chip"><span class="chip-num">05</span> Classification Model</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip"><span class="chip-num">06</span> Cross-Check Agreement</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip"><span class="chip-num">07</span> Detailed Detection Results Table</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip"><span class="chip-num">08</span> Confidence per Detection Chart</span>
</div>
<div class="flow-chip-row" style="margin-bottom:0;">
    <span class="flow-chip"><span class="chip-num">09</span> Severity Assessment</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip"><span class="chip-num">10</span> AI Report Generation (LLM)</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip"><span class="chip-num">11</span> Full Analysis Report (Image + Model Info)</span>
    <span class="flow-arrow">→</span>
    <span class="flow-chip"><span class="chip-num">12</span> PDF Report Download</span>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """<div class="about-wrap">
<div class="about-card">
<h3>📌 Project Overview</h3>
<p>This project is developed as part of an undergraduate thesis in Computer
Science and Engineering (CSE). It leverages modern Computer Vision techniques
to automate the detection, localization, and classification of kidney stones,
helping clinicians with faster and more reliable initial screening.</p>
</div>

<div class="about-card">
<h3>✨ Key Features</h3>
<div class="feature-grid">
<div class="feature-item"><b>🎯 Real-Time Detection</b>Accurate detection of kidney stones with bounding box overlays.</div>
<div class="feature-item"><b>📈 Performance Evaluation</b>High precision and mAP metrics on benchmark datasets.</div>
<div class="feature-item"><b>🖥️ Interactive Web Interface</b>Streamlit-based UI for seamless image upload and visualization.</div>
<div class="feature-item"><b>📝 AI Medical Report Generation</b>Automated summary reports powered by LLM integration.</div>
</div>
</div>

<div class="about-card">
<h3>🛠️ Tech Stack &amp; Libraries</h3>
<div class="tech-groups-row">
<div class="tech-group-block">
<div class="tech-group-label">Language</div>
<div class="tech-stack-row">
<span class="tech-pill">Python</span>
</div>
</div>
<div class="tech-group-block">
<div class="tech-group-label">Frameworks</div>
<div class="tech-stack-row">
<span class="tech-pill">Streamlit</span>
<span class="tech-pill">PyTorch</span>
<span class="tech-pill">Ultralytics YOLO</span>
</div>
</div>
<div class="tech-group-block">
<div class="tech-group-label">Libraries</div>
<div class="tech-stack-row">
<span class="tech-pill">OpenCV</span>
<span class="tech-pill">PIL</span>
<span class="tech-pill">NumPy</span>
<span class="tech-pill">Pandas</span>
</div>
</div>
<div class="tech-group-block">
<div class="tech-group-label">Model Architecture</div>
<div class="tech-stack-row">
<span class="tech-pill">YOLO Series (Custom Trained)</span>
</div>
</div>
</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    <div class="about-card" style="margin-top:1.4rem;">
    <h3>🔗 Explore the Project</h3>
    <p style="margin-bottom:1rem;">Source code and datasets behind this project.</p>
    <div class="resource-grid">
        <div class="resource-card">
            <div class="resource-head"><div class="resource-icon">📁</div><h4>Source Repository</h4></div>
            <p>Explore the complete project source code on GitHub.</p>
            <a href="{PROJECT_LINKS['github_repo']}" target="_blank">View on GitHub →</a>
        </div>
        <div class="resource-card">
            <div class="resource-head"><div class="resource-icon">🗄️</div><h4>Detection Dataset (Kaggle)</h4></div>
            <p>Bounding-box annotated kidney stone CT images used to train YOLO26x.</p>
            <a href="{PROJECT_LINKS['kaggle_bbox_dataset']}" target="_blank">Open Kaggle Dataset →</a>
        </div>
        <div class="resource-card">
            <div class="resource-head"><div class="resource-icon">🗄️</div><h4>Detection Dataset (Roboflow)</h4></div>
            <p>Additional annotated CT dataset from Roboflow Universe.</p>
            <a href="{PROJECT_LINKS['roboflow_dataset']}" target="_blank">Open Roboflow Dataset →</a>
        </div>
        <div class="resource-card">
            <div class="resource-head"><div class="resource-icon">🧪</div><h4>Classifier Dataset (Kaggle)</h4></div>
            <p>Axial CT imaging dataset used to train the classification model.</p>
            <a href="{PROJECT_LINKS['kaggle_classifier_dataset']}" target="_blank">Open Kaggle Dataset →</a>
        </div>
    </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------------
# PAGE: MODEL INFO
# ---------------------------------------------------------------------------
if st.session_state.nav_page == "model":
    st.markdown(
        """
    <div class="main-header">
        <div class="main-header-icon">🧠</div>
        <h1>About The Model</h1>
        <p>Details on the computer vision model powering this detection system</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Hide the sidebar entirely on this page — full-width layout, dataset details live in the Dataset Info tab
    st.markdown(
        "<style>section[data-testid='stSidebar'] {display: none;}</style>",
        unsafe_allow_html=True,
    )

    tab_options = [
        "📋 Model Summary",
        "🏗️ Architecture",
        "📈 Performance Evaluation",
        "🧠 Classification Model",
        "📊 Dataset Info",
    ]
    # Plain buttons instead of st.tabs() or st.radio(). st.tabs() renders
    # every tab's content on every single run (it only hides the inactive
    # ones with CSS client-side) — so all 5 sections' images, glob scans,
    # and metric computations were running on every page load regardless
    # of which tab was visible, which is what made this page slow.
    # st.radio() would fix that too, but its native radio-dot indicator
    # can't be reliably hidden across Streamlit versions. Buttons use the
    # same pattern as the working top nav bar — no such quirk.
    if "model_info_tab" not in st.session_state:
        st.session_state["model_info_tab"] = tab_options[0]
    with st.container(key="model_info_tab_bar"):
        tab_cols = st.columns(len(tab_options))
        for tab_col, tab_label in zip(tab_cols, tab_options):
            with tab_col:
                if st.button(
                    tab_label,
                    key=f"model_info_tab_btn_{tab_label}",
                    type="primary" if st.session_state["model_info_tab"] == tab_label else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["model_info_tab"] = tab_label
                    st.rerun()
    selected_tab = st.session_state["model_info_tab"]

    # --- Model Summary ---
    if selected_tab == "📋 Model Summary":
        st.markdown('<div class="section-title">Model Used</div>', unsafe_allow_html=True)
        st.table(pd.DataFrame({
            "Property": ["Model", "Variant", "Framework", "Task", "Class(es)", "Epochs Trained"],
            "Value": [
                MODEL_INFO["name"],
                MODEL_INFO["variant"],
                "Ultralytics YOLO",
                "Object Detection",
                "stone",
                str(MODEL_INFO["epochs"]),
            ],
        }).set_index("Property"))

        st.markdown('<div class="section-title" style="margin-top:1.4rem;">Training Configuration</div>', unsafe_allow_html=True)

        _train_cfg = load_train_config()
        _cfg_source = "Auto-detected" if _train_cfg else "Default"
        if not _train_cfg:
            _train_cfg = TRAIN_CONFIG_FALLBACK
        st.caption(f"{_cfg_source} — hyperparameters used to train {MODEL_INFO['name']}.")

        _cfg_fields = [
            ("Epochs", _train_cfg.get("epochs", "—")),
            ("Batch Size", _train_cfg.get("batch", "—")),
            ("Image Size", f"{_train_cfg.get('imgsz', '—')}px"),
            ("Optimizer", str(_train_cfg.get("optimizer", "—")).upper()),
            ("Initial LR", _train_cfg.get("lr0", "—")),
            ("Final LR", _train_cfg.get("lrf", "—")),
            ("Momentum", _train_cfg.get("momentum", "—")),
            ("Weight Decay", _train_cfg.get("weight_decay", "—")),
            ("Patience", _train_cfg.get("patience", "—")),
            ("Seed", _train_cfg.get("seed", "—")),
        ]
        _cfg_cols = st.columns(5)
        for i, (label, val) in enumerate(_cfg_fields):
            with _cfg_cols[i % 5]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:1.15rem;">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)
            if i % 5 == 4 and i != len(_cfg_fields) - 1:
                _cfg_cols = st.columns(5)

        st.markdown('<div class="section-title" style="margin-top:1.4rem;">Preprocessing Pipeline</div>', unsafe_allow_html=True)
        st.caption("What happens to your image between upload and detection.")
        st.markdown(
            """
            <div class="arch-card">
            <div class="flow-chip-row">
                <span class="flow-chip"><span class="chip-num">01</span> Upload Image</span>
                <span class="flow-arrow">→</span>
                <span class="flow-chip"><span class="chip-num">02</span> Convert to RGB</span>
                <span class="flow-arrow">→</span>
                <span class="flow-chip"><span class="chip-num">03</span> NumPy Array</span>
                <span class="flow-arrow">→</span>
                <span class="flow-chip"><span class="chip-num">04</span> Letterbox Resize (640×640)</span>
            </div>
            <div class="flow-chip-row" style="margin-bottom:0;">
                <span class="flow-chip"><span class="chip-num">05</span> Normalize Pixels (0–255 → 0–1)</span>
                <span class="flow-arrow">→</span>
                <span class="flow-chip"><span class="chip-num">06</span> Model Inference</span>
                <span class="flow-arrow">→</span>
                <span class="flow-chip"><span class="chip-num">07</span> Confidence Filter</span>
                <span class="flow-arrow">→</span>
                <span class="flow-chip"><span class="chip-num">08</span> IoU Non-Max Suppression</span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title" style="margin-top:1.4rem;">How Detection Works</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
            1. The uploaded image is passed to the model.<br>
            2. The model predicts candidate bounding boxes with confidence scores.<br>
            3. Boxes below the <b>Confidence Threshold</b> are discarded.<br>
            4. Overlapping boxes are merged using <b>Non-Maximum Suppression (IoU)</b>.<br>
            5. Remaining boxes are drawn on the image and summarized in the results table.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title" style="margin-top:1.4rem;">AI Report Generation</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
            After detection, an LLM (Llama 3.3 70B via Groq) can generate a
            structured, radiology-style report summarizing the findings —
            for informational purposes only.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title" style="margin-top:1.4rem;">Deployment Architecture</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="arch-card">
            <div class="flow-chip-row" style="margin-bottom:0;">
                <span class="flow-chip"><span class="chip-num">01</span> Browser</span>
                <span class="flow-arrow">→</span>
                <span class="flow-chip"><span class="chip-num">02</span> Streamlit UI</span>
                <span class="flow-arrow">→</span>
                <span class="flow-chip"><span class="chip-num">03</span> Python Backend (same process)</span>
                <span class="flow-arrow">→</span>
                <span class="flow-chip"><span class="chip-num">04</span> YOLO26x + Classifier Inference</span>
                <span class="flow-arrow">→</span>
                <span class="flow-chip"><span class="chip-num">05</span> Result Rendered Back to Page</span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Unlike a split frontend/backend deployment, this project runs as a single "
            "Streamlit application — the UI and the model inference logic execute in the "
            "same Python process, with no separate API server."
        )

    # --- Architecture ---
    if selected_tab == "🏗️ Architecture":
        st.markdown('<div class="section-title">YOLO26 Architecture</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
        This system uses <b>{MODEL_INFO['name']}</b> — the <b>xl (extra-large)</b> variant of
        <b>YOLO26</b>, the latest generation in the YOLO (You Only Look Once) family of
        single-stage, anchor-free object detectors. It is fine-tuned specifically to identify
        kidney stones in CT-scan imagery, processing the entire image in a single forward pass
        and predicting all bounding boxes and confidence scores simultaneously.
        </div>
        """, unsafe_allow_html=True)

        arch_img_path = os.path.join(ASSETS_DIR, "architecture.png")
        if os.path.isfile(arch_img_path):
            arch_c1, arch_c2, arch_c3 = st.columns([1, 5, 1])
            with arch_c2:
                st.image(arch_img_path, caption="YOLO26 Architecture Diagram", use_container_width=True)
        else:
            st.markdown("""
            <div class="warn-box">
                ⚠️ Architecture diagram not found. Upload an image below,
                or place one at <code>assets/architecture.png</code> in the
                project folder — it will then show up here automatically.
            </div>
            """, unsafe_allow_html=True)
            uploaded_arch_img = st.file_uploader(
                "Upload Architecture Diagram",
                type=["png", "jpg", "jpeg"],
                key="arch_uploader",
            )
            if uploaded_arch_img is not None:
                uarch_c1, uarch_c2, uarch_c3 = st.columns([1, 5, 1])
                with uarch_c2:
                    st.image(uploaded_arch_img, caption="YOLO26 Architecture Diagram", use_container_width=True)

        st.markdown("""
        <p style="color:#33504c;font-size:0.95rem;margin:0.2rem 0 1rem 0;">
        The network is organized into three stages — <b>Backbone</b>, <b>Neck</b>, and <b>Head</b>:
        </p>

        <div class="arch-card">
        <h4>1️⃣ Backbone (Feature Extraction)</h4>
        <p>
        The input image (640×640×3) is passed through a stack of strided <b>Conv</b> layers
        (each halving the spatial resolution: P1 → P5) interleaved with <b>C3k2</b> blocks —
        CSP-style blocks made of stacked 3×3 convolutions that extract increasingly abstract
        features while keeping the parameter count efficient. At the deepest stage (P5), two
        specialized blocks refine the features further:
        </p>
        <ul>
            <li><b>SPPF (Spatial Pyramid Pooling – Fast)</b> — pools features at multiple receptive
            field sizes so the network sees both local detail and global context.</li>
            <li><b>C2PSA (Parallel Spatial Attention)</b> — an attention block that lets the network
            focus on the most relevant spatial regions, useful for small, low-contrast objects
            such as kidney stones.</li>
        </ul>
        </div>

        <div class="arch-card">
        <h4>2️⃣ Neck (Multi-Scale Feature Fusion)</h4>
        <p>The neck follows a <b>PAN-FPN</b> (Path Aggregation + Feature Pyramid) design:</p>
        <ul>
            <li>A <b>top-down path</b> upsamples deep, semantically-rich features (P5) and
            concatenates them with shallower, higher-resolution features (P4, P3), refined at
            each step by C3k2 blocks.</li>
            <li>A <b>bottom-up path</b> then re-downsamples these fused features back through P4
            and P5, so every scale ends up with both fine spatial detail and strong semantic
            context.</li>
        </ul>
        <p>
        This fusion is what allows the model to detect kidney stones of very different sizes —
        from tiny sub-5 mm stones to large clusters — within the same image.
        </p>
        </div>

        <div class="arch-card">
        <h4>3️⃣ Head (Multi-Scale Detection)</h4>
        <p>
        Three parallel <b>Detect</b> heads operate on the fused P3, P4, and P5 feature maps
        (80×80, 40×40, and 20×20 respectively), each independently predicting bounding boxes,
        objectness, and class confidence at that scale. Detecting at three resolutions
        simultaneously means small stones (best seen at P3) and larger ones (best seen at P5)
        are both captured reliably in a single pass.
        </p>
        </div>

        <div class="arch-card">
        <h4>📐 Model Scaling</h4>
        <p>
        YOLO26 defines five variants (<b>n, s, m, l, xl</b>) scaled by a <b>depth multiplier
        (d)</b>, <b>width multiplier (w)</b>, and <b>max channel cap (mc)</b>. This project uses
        the <span class="arch-highlight">xl variant</span> — the deepest and widest
        configuration — trading some inference speed for the highest accuracy, which is
        appropriate for an offline/near-real-time diagnostic-support tool rather than a strict
        real-time video pipeline.
        </p>
        </div>

        <div class="arch-card">
        <h4>⚡ Why YOLO26 specifically (vs. earlier YOLO versions)</h4>
        <p>
        Independent latency-vs-accuracy benchmarks (COCO mAP50-95 vs. TensorRT FP16 latency)
        show YOLO26 achieving a better accuracy-per-millisecond trade-off than YOLO11, YOLOv10,
        YOLOv9, and YOLOv8 across every model size — at any given latency budget, YOLO26 reaches
        a higher mAP. This makes it a strong choice for a medical imaging task where both
        detection accuracy (missing a stone is costly) and reasonable inference speed matter.
        </p>
        </div>
        """, unsafe_allow_html=True)

        perf_img_path = os.path.join(ASSETS_DIR, "performance_comparison.png")
        if os.path.isfile(perf_img_path):
            st.markdown('<div class="section-title" style="margin-top:1rem;">YOLO Version Comparison</div>', unsafe_allow_html=True)
            perf_c1, perf_c2, perf_c3 = st.columns([1, 5, 1])
            with perf_c2:
                st.image(
                    perf_img_path,
                    caption="YOLO26 vs. YOLO11 / YOLOv10 / YOLOv9 / YOLOv8 — latency vs. COCO mAP50-95",
                    use_container_width=True,
                )
        else:
            st.caption(
                "Optional: place a latency-vs-accuracy comparison chart at "
                "`assets/performance_comparison.png` to display it here."
            )

    # --- Performance Evaluation ---
    if selected_tab == "📈 Performance Evaluation":
        st.markdown('<div class="section-title">Metrics Summary</div>', unsafe_allow_html=True)
        st.caption(f"Results on the validation set after {MODEL_INFO['epochs']} epochs of training.")

        pm1, pm2, pm3, pm4, pm5 = st.columns(5)
        _metric_tooltips = {
            "Precision": "Of everything the model flagged as a stone, the percent that actually were.",
            "Recall": "Of all the real stones in the images, the percent the model actually found.",
            "mAP@50": "Mean average precision at 50% overlap threshold — a standard detection accuracy score.",
            "mAP@50-95": "Mean average precision averaged over stricter overlap thresholds (50%–95%) — a tougher, more demanding score.",
            "F1-Score": "The balance between Precision and Recall as a single number.",
        }
        for col, label, val in zip(
            [pm1, pm2, pm3, pm4, pm5],
            ["Precision", "Recall", "mAP@50", "mAP@50-95", "F1-Score"],
            [MODEL_INFO["precision"], MODEL_INFO["recall"], MODEL_INFO["map50"], MODEL_INFO["map50_95"], MODEL_INFO["f1"]],
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card" data-tip="{_metric_tooltips[label]}" tabindex="0">
                    <div class="metric-value" style="font-size:1.7rem;">{val:.1f}%</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Result Plots</div>', unsafe_allow_html=True)

        result_plots = {
            "Confusion Matrix": ["confusion_matrix.png"],
            "Confusion Matrix (Normalized)": ["confusion_matrix_normalized.png"],
            "Overall Results (loss / mAP curves)": ["results.png"],
            "Precision-Recall Curve": ["BoxPR_curve.png", "PR_curve.png"],
            "F1 Curve": ["BoxF1_curve.png", "F1_curve.png"],
            "Precision Curve": ["BoxP_curve.png", "P_curve.png"],
            "Recall Curve": ["BoxR_curve.png", "R_curve.png"],
            "Dataset Label Distribution": ["labels.jpg"],
        }

        any_found = False
        with st.container(key="result_plots_grid"):
            plot_cols = st.columns(2)
            i = 0
            for label, filenames in result_plots.items():
                found_path = find_first_existing(*filenames)
                if found_path:
                    any_found = True
                    with plot_cols[i % 2]:
                        st.markdown(f"**{label}**")
                        st.image(found_path, use_container_width=True)
                    i += 1

        if not any_found:
            st.markdown("""
            <div class="warn-box">
                ⚠️ No training result plots were found. Please verify the
                files exist under <code>runs/detect/train/</code>.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Sample Validation Predictions</div>', unsafe_allow_html=True)
        val_pred_paths = sorted(
            glob.glob(os.path.join(RESULTS_DIR, "val_batch*_pred.jpg"))
        )
        if val_pred_paths:
            vp_cols = st.columns(min(3, len(val_pred_paths)))
            for idx, vp_path in enumerate(val_pred_paths):
                with vp_cols[idx % len(vp_cols)]:
                    st.image(vp_path, use_container_width=True)
        else:
            st.caption("No validation prediction samples found.")

            st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Held-Out Test Set Results</div>', unsafe_allow_html=True)
        st.caption(f"Evaluated on {DET_TEST_METRICS['n_images']} untouched held-out test images — never used in training or validation.")

        tm1, tm2, tm3, tm4, tm5 = st.columns(5)
        for col, label, val in zip(
            [tm1, tm2, tm3, tm4, tm5],
            ["Precision", "Recall", "mAP@50", "mAP@50-95", "F1-Score"],
            [DET_TEST_METRICS["precision"], DET_TEST_METRICS["recall"], DET_TEST_METRICS["map50"], DET_TEST_METRICS["map50_95"], DET_TEST_METRICS["f1"]],
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card" data-tip="{_metric_tooltips[label]}" tabindex="0">
                    <div class="metric-value" style="font-size:1.7rem;">{val:.2f}%</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        test_plots = {
            "Confusion Matrix (Test)": ["confusion_matrix_test.png"],
            "Confusion Matrix Normalized (Test)": ["confusion_matrix_test_normalized.png"],
        }
        test_any_found = False
        with st.container(key="test_plots_grid"):
            test_plot_cols = st.columns(2)
            j = 0
            for label, filenames in test_plots.items():
                found_path = find_first_existing(*filenames)
                if found_path:
                    test_any_found = True
                    with test_plot_cols[j % 2]:
                        st.markdown(f"**{label}**")
                        st.image(found_path, use_container_width=True)
                    j += 1
        if not test_any_found:
            st.caption("Test-set confusion matrix images not found.")    

    # --- Classification Model ---
    if selected_tab == "🧠 Classification Model":
        clf_metrics = get_classifier_metrics()
        clf_test_metrics = get_classifier_test_metrics()
        clf_cv_metrics = get_classifier_cv_metrics()

        st.markdown('<div class="section-title">What This Model Is</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
            A <b>YOLO11n classification</b> model trained as a standalone pipeline
            alongside the detector — not a fallback or add-on. It looks at the
            whole image and predicts <code>stone</code> vs
            <code>non_stone</code>, with no bounding boxes involved. This project
            has two co-equal components: a <b>detection</b> pipeline that
            localizes stones, and this <b>classification</b> pipeline that judges
            the whole image on its own. Because whole-image classification and
            per-box localization are different tasks, their accuracy is measured
            and reported separately rather than merged into one score.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title" style="margin-top:1.4rem;">Architecture</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
            Built on <b>YOLO11n-cls</b> — the nano classification variant of the YOLO11
            family. Unlike the detector, it has no detection head at all: the backbone
            (a stack of Conv + C3k2 feature-extraction blocks, the same family used in
            the detector's backbone) feeds directly into a global average-pool and a
            single fully-connected classification layer that outputs two class
            probabilities — <code>stone</code> vs <code>non_stone</code>. Input images are
            resized to 384×384. Being "nano" (the smallest YOLO11 scale) keeps it fast
            enough for real-time inference, at the cost of the fine-grained localization
            only a full detection model can provide — which is exactly why the project
            runs both models as separate, complementary pipelines.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title" style="margin-top:1.4rem;">Methodology</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
            The dataset was split <b>80% train / 10% validation / 10% held-out test</b>,
            with the test set kept completely untouched until final evaluation — it
            never influenced training, checkpoint selection, or early stopping.
            To confirm the result is stable and not a lucky split, a
            <b>5-fold cross-validation</b> was additionally run on the training pool:
            the model was retrained from scratch 5 times, each time with a different
            fifth of the data held out for validation, to measure the mean and spread
            of accuracy across folds.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title" style="margin-top:1.4rem;">Held-Out Test Results</div>', unsafe_allow_html=True)

        if clf_test_metrics is None:
            st.markdown("""
            <div class="warn-box">
                ⚠️ Test-set results not found. Copy <code>final_test_results.json</code>
                (saved by the training notebook after evaluating the untouched test set)
                into the <code>final_model</code> folder in this app's root directory.
            </div>
            """, unsafe_allow_html=True)
        else:
            n_test_images = sum(len(row) for row in clf_test_metrics.get("confusion_matrix", [])) or None
            caption = "Evaluated on the untouched held-out test set"
            if n_test_images:
                caption += f" ({n_test_images} images)"
            st.caption(caption + ".")

            tm1, tm2, tm3, tm4 = st.columns(4)
            for col, label, key in zip(
                [tm1, tm2, tm3, tm4],
                ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"],
                ["accuracy", "precision", "recall", "f1"],
            ):
                with col:
                    val = clf_test_metrics.get(key)
                    val_str = f"{val*100:.2f}%" if val is not None else "N/A"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.6rem;">{val_str}</div>
                        <div class="metric-label">{label}</div>
                    </div>""", unsafe_allow_html=True)

            extra_metrics = compute_binary_extra_metrics(clf_test_metrics.get("confusion_matrix"))
            if extra_metrics is not None:
                tm5, tm6 = st.columns(2)
                with tm5:
                    st.markdown(f"""
                    <div class="metric-card" data-tip="Average accuracy across both classes — corrects for uneven stone/non-stone counts in the test set." tabindex="0">
                        <div class="metric-value" style="font-size:1.6rem;">{extra_metrics['balanced_accuracy']*100:.2f}%</div>
                        <div class="metric-label">Balanced Accuracy</div>
                    </div>""", unsafe_allow_html=True)
                with tm6:
                    st.markdown(f"""
                    <div class="metric-card" data-tip="Matthews Correlation Coefficient — a single -1 to 1 score of prediction quality; 0 is random guessing, 1 is perfect." tabindex="0">
                        <div class="metric-value" style="font-size:1.6rem;">{extra_metrics['mcc']:.4f}</div>
                        <div class="metric-label">MCC</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">5-Fold Cross-Validation</div>', unsafe_allow_html=True)

        if clf_cv_metrics is None:
            st.markdown("""
            <div class="warn-box">
                ⚠️ Cross-validation results not found. Copy <code>fold_results.json</code>
                from your Drive's <code>kfold_runs</code> folder into a
                <code>kfold_runs</code> folder in this app's root directory.
            </div>
            """, unsafe_allow_html=True)
        else:
            cv1, cv2 = st.columns(2)
            with cv1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:1.7rem;">{clf_cv_metrics['mean']*100:.2f}%</div>
                    <div class="metric-label">Mean Accuracy ({len(clf_cv_metrics['folds'])} folds)</div>
                </div>""", unsafe_allow_html=True)
            with cv2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:1.7rem;">± {clf_cv_metrics['std']*100:.2f}%</div>
                    <div class="metric-label">Std. Deviation</div>
                </div>""", unsafe_allow_html=True)

            fold_chips = "".join(
                f'<span class="tech-pill" style="margin:0.2rem;">Fold {k}: {v*100:.2f}%</span>'
                for k, v in sorted(clf_cv_metrics["folds"].items(), key=lambda x: int(x[0]))
            )
            st.markdown(f'<div style="margin-top:0.6rem;">{fold_chips}</div>', unsafe_allow_html=True)

        if clf_metrics is not None and clf_metrics.get("top1") is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption(
                f"Training-run accuracy curve (final epoch, {clf_metrics['epochs_run']} epochs): "
                f"{clf_metrics['top1']*100:.2f}% — see Result Plots below for the full curve."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Result Plots</div>', unsafe_allow_html=True)

        clf_plots = {
            "Confusion Matrix": ["confusion_matrix.png"],
            "Confusion Matrix (Normalized)": ["confusion_matrix_normalized.png"],
            "Training Curves (loss / accuracy)": ["results.png"],
            "ROC Curve": ["roc_curve.png"],
        }

        any_clf_found = False
        with st.container(key="clf_plots_grid"):
            clf_plot_cols = st.columns(2)
            i = 0
            for label, filenames in clf_plots.items():
                found_path = find_first_existing(*filenames, search_dir=CLASSIFIER_RESULTS_DIR)
                if found_path:
                    any_clf_found = True
                    with clf_plot_cols[i % 2]:
                        st.markdown(f"**{label}**")
                        st.image(found_path, use_container_width=True)
                    i += 1

        if not any_clf_found:
            st.markdown("""
            <div class="warn-box">
                ⚠️ No classification result plots found under
                <code>final_model/</code>.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Sample Training / Validation Batches</div>', unsafe_allow_html=True)
        clf_batch_paths = sorted(
            glob.glob(os.path.join(CLASSIFIER_RESULTS_DIR, "val_batch*_pred.jpg"))
        )
        if clf_batch_paths:
            cb_cols = st.columns(min(3, len(clf_batch_paths)))
            for idx, cb_path in enumerate(clf_batch_paths):
                with cb_cols[idx % len(cb_cols)]:
                    st.image(cb_path, use_container_width=True, caption=os.path.basename(cb_path))
        else:
            st.caption("No sample batch images found.")

    # --- Dataset Info (placed after Performance Evaluation) ---
    if selected_tab == "📊 Dataset Info":
        d1, d2 = st.columns(2, gap="large")
        with d1:
            st.markdown('<div class="section-title">Detection Dataset Sources</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="info-box">
                Two public kidney-stone object-detection datasets were merged to build the
                training set:
                <ul>
                    <li><b>Kidney Stone Images with Bounding Box Annotations</b> (Kaggle,
                    by Safura Hajiheidari) — ~1,300 CT-scan images with YOLO-format bounding
                    box annotations, itself sourced from Roboflow Universe.</li>
                    <li><b>Kidney Stone Detection</b> (Roboflow Universe, by East West
                    University) — 1,299 CT-scan images with bounding box annotations.</li>
                </ul>
                Because both datasets originate from overlapping Roboflow sources, the merged
                collection was de-duplicated and consolidated before training.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with d2:
            st.markdown('<div class="section-title">Annotation & Classes</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="info-box">
                <b>Target Class:</b> <code>stone</code> (single-class object detection)<br><br>
                Bounding boxes are represented using normalized YOLO coordinate format:
                <code>[class_id, x_center, y_center, width, height]</code>.<br>
                Both source datasets provide CT-scan images with kidney stones annotated
                across a range of sizes, shapes, and positions within the urinary system.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Detection Dataset Distribution Summary</div>', unsafe_allow_html=True)
        st.caption("Final merged dataset used for training: 2,659 images total.")

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-value">2,290</div>
                    <div class="metric-label">Training Images (86.1%)</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_d2:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-value">123</div>
                    <div class="metric-label">Validation Images (4.6%)</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_d3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-value">246</div>
                    <div class="metric-label">Test Images (9.3%)</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Classifier Training Dataset</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
            The <b>classification model</b> is trained separately from
            the detector, on a different source: the
            <b>Axial CT Imaging Dataset for AI-Powered Kidney Stone Detection</b>
            (Kaggle, by Orvile) — a large collection of axial CT slices labeled as
            <code>stone</code> or <code>non_stone</code>, originally intended for
            whole-image classification research rather than bounding-box detection.
            A subset of ~3,364 images was drawn from it for this project.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Classifier dataset split: 3,364 images total.")
        col_c0, col_c1, col_c2, col_c3 = st.columns(4)
        with col_c0:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-value">3,364</div>
                    <div class="metric-label">Total Images</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_c1:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-value">1,577</div>
                    <div class="metric-label">Stone Images</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_c2:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-value">1,787</div>
                    <div class="metric-label">Non-Stone Images</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_c3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-value" style="font-size:1.7rem;">80/10/10</div>
                    <div class="metric-label">Train / Val / Test Split</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Data Leakage Prevention</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
            The held-out test set was separated <b>first</b>, before any training or
            validation split was created — and it was never touched again until the
            single, final evaluation. Both the classifier's train/val/test split and the
            detector's dataset partitioning follow the same rule: an image assigned to
            one split can never appear in another.
            </div>
            <div class="leakage-row">
                <span class="leakage-pill">Train ∩ Val = ∅</span>
                <span class="leakage-pill">Train ∩ Test = ∅</span>
                <span class="leakage-pill">Val ∩ Test = ∅</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Class Imbalance Handling</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
            The classifier's training data is <b>near-balanced</b> — 1,577 stone images
            (46.9%) vs. 1,787 non-stone images (53.1%), roughly a 47/53 split. Since this
            is far from the kind of severe imbalance (e.g. 90/10) that would bias a model
            toward the majority class, no class-weighting or resampling was applied —
            standard, equally-weighted cross-entropy loss was used for training.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Citations</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box" style="font-size:0.82rem;">
        S. Hajiheidari, "Kidney Stone Images with Bounding Box Annotations," Kaggle, 2023.
        Available: kaggle.com/datasets/safurahajiheidari/kidney-stone-images<br><br>
        East West University, "Kidney Stone Detection Dataset," Roboflow Universe, 2023.
        Available: universe.roboflow.com/east-west-university-9frzq/kidney-stone-detection-wfjba<br><br>
        Orvile, "Axial CT Imaging Dataset for AI-Powered Kidney Stone Detection," Kaggle, 2024.
        Available: kaggle.com/datasets/orvile/axial-ct-imaging-dataset-kidney-stone-detection
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        """
    <div class="warn-box" style="margin-top:1.4rem;">
        ⚠️ This model is a research/educational tool. It is <b>not</b> a certified
        diagnostic device. Detection results and confidence scores should always
        be verified by a qualified radiologist or urologist.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------------
# PAGE: UPLOAD & DETECT
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Detections below this confidence are discarded.",
    )
    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Non-Maximum Suppression overlap threshold.",
    )
    classifier_threshold = st.slider(
        "Classification Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="How confident the classification model must be to report a "
        "stone as present.",
    )
    st.markdown("---")
    st.markdown(
        "<p style='color:#5b7370;font-size:0.85rem;'>This application is intended for "
        "research and educational purposes only. It does not constitute medical advice, "
        "diagnosis, or treatment. Always consult a qualified healthcare professional for "
        "medical decisions.</p>",
        unsafe_allow_html=True,
    )


st.markdown(
    """
<div class="main-header">
    <div class="main-header-icon">🔬</div>
    <h1>Kidney Stone Detection System</h1>
    <p>Upload a CT scan to run detection, classification, and generate an AI report</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.spinner("Loading model…"):
    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

classifier = load_classifier()


class _SampleFile(io.BytesIO):
    """Wraps a demo-sample image so it behaves like a Streamlit UploadedFile."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name
        self.size = len(data)


uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    help="Supported formats: JPG, JPEG, PNG, BMP, TIFF",
)

if uploaded_file is not None:
    # A manual upload always takes priority over a previously chosen demo sample.
    st.session_state.pop("selected_sample", None)
elif st.session_state.get("selected_sample"):
    _sample_path = st.session_state["selected_sample"]
    with open(_sample_path, "rb") as _f:
        uploaded_file = _SampleFile(_f.read(), name=os.path.basename(_sample_path))
    st.info(f"🧪 Using demo sample: **{uploaded_file.name}**")
    if st.button("✕ Clear sample and upload my own"):
        st.session_state.pop("selected_sample", None)
        st.rerun()

if uploaded_file is None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
    st.stop()

# Reset the previously generated AI report whenever a new/different image is uploaded
current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
if st.session_state.get("last_uploaded_file_id") != current_file_id:
    st.session_state.pop("ai_report", None)
    st.session_state["last_uploaded_file_id"] = current_file_id

image = Image.open(uploaded_file).convert("RGB")

# Safety cap: an unusually large upload (e.g. a multi-thousand-pixel scan
# export) can spike memory and CPU/GPU time during inference. Downscale
# proportionally rather than rejecting the upload outright, so the feature
# still works end-to-end — just at a capped, still plenty-detailed resolution.
MAX_IMAGE_DIMENSION = 4000
if max(image.size) > MAX_IMAGE_DIMENSION:
    original_size = image.size
    image = image.copy()
    image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)
    st.caption(
        f"ℹ️ Image downscaled from {original_size[0]}×{original_size[1]}px to "
        f"{image.width}×{image.height}px for processing."
    )
    _resized_buf = io.BytesIO()
    image.save(_resized_buf, format="PNG")
    image_bytes = _resized_buf.getvalue()
else:
    image_bytes = uploaded_file.getvalue()

img_array = np.array(image)

pipeline_progress = st.progress(0, text="Running detection…")
try:
    detections, elapsed_s, annotated_rgb = run_detection_cached(model, image_bytes, conf_threshold, iou_threshold)
except Exception as exc:
    pipeline_progress.empty()
    st.error(f"Detection failed: {exc}")
    st.stop()
pipeline_progress.progress(60, text="Running classification model…")

stone_count = len(detections)
avg_conf = float(np.mean([d["Confidence"] for d in detections])) if detections else 0.0
max_conf = float(np.max([d["Confidence"] for d in detections])) if detections else 0.0
min_conf = float(np.min([d["Confidence"] for d in detections])) if detections else 0.0
severity_label, severity_color = get_severity(stone_count, avg_conf)

clf_stone_prob = run_classifier_cached(classifier, image_bytes)
pipeline_progress.progress(100, text="Done")
pipeline_progress.empty()

second_opinion_flag = (
    clf_stone_prob is not None
    and stone_count == 0
    and clf_stone_prob >= classifier_threshold
)

if stone_count > 0:
    display_image = Image.fromarray(annotated_rgb)
    display_caption = (
        f"{uploaded_file.name}  ·  {image.width}×{image.height} px  · "
        f" {stone_count} stone(s) detected"
    )
else:
    display_image = image
    display_caption = (
        f"{uploaded_file.name}  ·  {image.width}×{image.height} px  ·  No stones"
        " detected"
    )

col_img, col_analysis = st.columns([1.05, 1], gap="large")

with col_img:
    st.markdown(
        f'<div class="section-title">{"Detection Result" if stone_count > 0 else "Uploaded Image"}</div>',
        unsafe_allow_html=True,
    )
    st.image(display_image, use_container_width=True, caption=display_caption)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    file_size_kb = buf.tell() / 1024

with col_analysis:
    st.markdown('<div class="section-title">Detection Summary</div>', unsafe_allow_html=True)

    if stone_count > 0:
        st.markdown('<span class="status-detected">⚠️ Kidney Stone(s) Detected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-clear">✅ No Stones Detected</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stone_count}</div>
                <div class="metric-label">Stones Found</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{avg_conf*100:.1f}%</div>
                <div class="metric-label">Avg Confidence</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{elapsed_s:.2f}s</div>
                <div class="metric-label">Inference Time</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
    <div class="severity-box" style="--severity-color:{severity_color};">
        <span class="severity-title">Severity Assessment</span><br>
        <span class="severity-body">{severity_label}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if classifier is None:
        st.caption("ℹ️ Classification model not loaded (classifier.pt not found) — Classification skipped.")
    else:
        detector_says = "Stone(s) found" if stone_count > 0 else "No stone found"
        clf_says = "Stone likely" if (clf_stone_prob or 0) >= classifier_threshold else "No stone likely"
        agrees = (stone_count > 0) == ((clf_stone_prob or 0) >= classifier_threshold)

        if second_opinion_flag:
            card_border = "#b45309"
            card_bg = "rgba(234, 179, 8, 0.08)"
            headline = "⚠️ Disagreement — worth a closer look"
        elif agrees:
            card_border = "#0369a1"
            card_bg = "rgba(14, 165, 233, 0.08)"
            headline = "✓ Models agree"
        else:
            card_border = "#b45309"
            card_bg = "rgba(234, 179, 8, 0.08)"
            headline = "⚠️ Models disagree"

        st.markdown(
            f"""
        <div class="agreement-card" style="--agreement-bg:{card_bg};--agreement-border:{card_border};">
            <div class="agreement-headline">
                Classification Result — {headline}
            </div>
            <div class="agreement-row">
                <div class="agreement-item">
                    <div class="agreement-item-label">Detection Model Says</div>
                    <div class="agreement-item-value">{detector_says}</div>
                </div>
                <div class="agreement-item">
                    <div class="agreement-item-label">Classification Model Says</div>
                    <div class="agreement-item-value">{clf_says} ({clf_stone_prob*100:.1f}%)</div>
                </div>
            </div>
            {"<div class='agreement-note'>The detection model found no stones, but the classification model reports a stone is likely present. Consider a lower confidence threshold or specialist review.</div>" if second_opinion_flag else ""}
        </div>
        """,
            unsafe_allow_html=True,
        )

    if stone_count > 0:
        st.markdown(
            f"""
            <div class="quickstat-row">
                <div class="quickstat-item">
                    <div class="quickstat-label">Max Conf</div>
                    <div class="quickstat-value">{max_conf*100:.1f}%</div>
                </div>
                <div class="quickstat-item">
                    <div class="quickstat-label">Min Conf</div>
                    <div class="quickstat-value">{min_conf*100:.1f}%</div>
                </div>
                <div class="quickstat-item">
                    <div class="quickstat-label">Threshold</div>
                    <div class="quickstat-value">{conf_threshold*100:.0f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---- AI Report Generation (full width, no side gap) ----
if stone_count > 0:
    st.markdown("---")
    header_col, control_col = st.columns([3, 1])
    with header_col:
        st.markdown('<div class="section-title">AI-Generated Report</div>', unsafe_allow_html=True)
    with control_col:
        report_columns = st.selectbox(
            "Layout",
            options=[1, 2, 3],
            index=1,
            format_func=lambda n: f"{n} Column{'s' if n > 1 else ''}",
            label_visibility="collapsed",
        )

    if st.button("🧾 Generate AI Report", use_container_width=True):
        with st.spinner("Generating AI report..."):
            try:
                ai_report = generate_ai_report(
                    stone_count, avg_conf, severity_label, detections,
                    clf_stone_prob=clf_stone_prob,
                    classifier_threshold=classifier_threshold,
                )
                st.session_state["ai_report"] = ai_report
            except Exception:
                logger.error("generate_ai_report failed", exc_info=True)
                st.error(
                    "Couldn't generate the AI report right now (the report service may be "
                    "temporarily unavailable or slow). Please try again in a moment."
                )

    if "ai_report" in st.session_state:
        report_html = _sanitize_llm_html(st.session_state["ai_report"])
        st.markdown(
            f"""
                <style>
                .ai-report-columns {{
                    column-count: {report_columns};
                    column-gap: 2.2rem;
                    width: 100%;
                }}
                .ai-report-columns h1, .ai-report-columns h2, .ai-report-columns h3 {{
                    color: #0d9488;
                    break-after: avoid;
                }}
                .ai-report-columns p, .ai-report-columns li {{
                    color: #33504c;
                    break-inside: avoid;
                }}
                .ai-report-columns ul, .ai-report-columns ol {{
                    padding-left: 1.2rem;
                }}
                </style>
                """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="ai-report-columns">\n\n{report_html}\n\n</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        try:
            pdf_bytes = build_pdf_report(
                uploaded_file.name, stone_count, avg_conf, severity_label, detections, report_html
            )
            st.download_button(
                label="⬇️ Download Report as PDF",
                data=pdf_bytes,
                file_name=f"kidney_stone_report_{uploaded_file.name.rsplit('.', 1)[0]}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")

st.markdown("---")
st.markdown('<div class="section-title">Detailed Detection Results</div>', unsafe_allow_html=True)

if detections:
    df = pd.DataFrame(detections)[[
        "Detection #",
        "Label",
        "Conf %",
        "Area %",
        "Location (x1,y1)",
        "Size (WxH)",
    ]]
    st.table(
        df.style.set_properties(**{
            "background-color": "#ffffff",
            "color": "#1e293b",
            "border": "1px solid #e2eeec",
        }).set_table_styles([{
            "selector": "th",
            "props": [
                ("background-color", "#0d9488"),
                ("color", "#ffffff"),
                ("font-weight", "bold"),
            ],
        }]).hide(axis="index")
    )

    st.markdown(
        '<div class="section-title" style="margin-top:1.2rem;">Confidence per Detection</div>',
        unsafe_allow_html=True,
    )
    chart_df = (
        pd.DataFrame({
            "Detection": [f"Stone #{d['Detection #']}" for d in detections],
            "Confidence (%)": [round(d["Confidence"] * 100, 2) for d in detections],
        })
        .sort_values("Confidence (%)", ascending=False)
        .set_index("Detection")
    )
    st.bar_chart(chart_df, color="#0d9488", height=260)

else:
    st.markdown(
        f"""
    <div class="info-box">
        ✅ No kidney stones were detected at the current confidence threshold
        (<b>{conf_threshold*100:.0f}%</b>). Try lowering the threshold in the
        sidebar if you believe stones may be present.
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown('<div class="section-title">Full Analysis Report</div>', unsafe_allow_html=True)

r1, r2 = st.columns(2, gap="medium")
with r1:
    st.markdown("**Image Information**")
    st.table(
        pd.DataFrame({
            "Property": ["Filename", "Dimensions", "Color Mode", "File Size"],
            "Value": [
                uploaded_file.name,
                f"{image.width} × {image.height} px",
                image.mode,
                f"{file_size_kb:.1f} KB",
            ],
        }).set_index("Property")
    )

with r2:
    st.markdown("**Model & Detection Info**")
    st.table(
        pd.DataFrame({
            "Property": [
                "Conf Threshold",
                "IoU Threshold",
                "Inference Time",
                "Stones Detected",
                "Avg Confidence",
            ],
            "Value": [
                f"{conf_threshold*100:.0f}%",
                f"{iou_threshold*100:.0f}%",
                f"{elapsed_s:.2f} s",
                str(stone_count),
                f"{avg_conf*100:.1f}%" if stone_count else "N/A",
            ],
        }).set_index("Property")
    )

st.markdown("---")
st.markdown(FOOTER_HTML, unsafe_allow_html=True)

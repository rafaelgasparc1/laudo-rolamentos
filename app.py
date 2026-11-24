from datetime import datetime
import os
import textwrap
from zoneinfo import ZoneInfo
import librosa
import librosa.display
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from scipy.signal import butter, hilbert, sosfiltfilt, welch
import streamlit as st

# --- URL DO SEU WEBHOOK DO GOOGLE SHEETS PESSOAL ---
WEBHOOK_SHEETS_URL = "SUA_URL_DO_GOOGLE_APPS_SCRIPT_AQUI"

# Caminho opcional para o logotipo da VINCI no repositório GitHub
LOGO_VINCI_PATH = "vinci_logo.png"

# Fuso horário oficial de Manaus (UTC-4)
TZ_MANAUS = ZoneInfo("America/Manaus")

st.set_page_config(
    page_title="VINCI | Inspeção Preditiva Aeroportuária",
    page_icon="✈️",
    layout="centered",
)

# --- ESTILO VISUAL AEROPORTUÁRIO + TRAVA DE CAIXA ALTA NA DIGITAÇÃO ---
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #071324 0%, #0d213c 100%);
            color: #f1f5f9;
        }
        .aero-banner {
            background: linear-gradient(90deg, #0a1f3d 0%, #123566 100%);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-left: 5px solid #38bdf8;
            border-radius: 10px;
            padding: 18px 20px;
            margin-bottom: 18px;
        }
        .aero-tag {
            font-size: 11px;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #7dd3fc;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .aero-title {
            font-size: 22px;
            font-weight: 700;
            color: #ffffff;
            margin: 0;
        }
        .aero-sub {
            font-size: 13px;
            color: #cbd5e1;
            margin-top: 6px;
            margin-bottom: 0;
        }
        div[data-testid="stTextInput"] input {
            text-transform: uppercase !important;
            font-weight: 700;
            letter-spacing: 0.8px;
        }
        .stButton > button {
            width: 100%;
            background-color: #0284c7;
            color: white;
            font-weight: 700;
            border-radius: 8px;
            border: none;
            padding: 0.65rem 1rem;
        }
        .stButton > button:hover {
            background-color: #0369a1;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CADASTRO DE ATIVOS / TAGS POR SUBTIPO DE BOMBA ---
CADASTRO_BOMBAS = {
    **{
        tag: {
            "tipo": "Bomba",
            "subtipo": "Água Gelada Secundária",
            "rpm_min": 2900,
            "rpm_max": 3100,
            "rpm_nominal": 3000,
            "aceita_lubrificacao": True,
            "lowcut": 2500,
            "highcut": 8000,
            "lp_env": 500,
        }
        for tag in [
            "MAOMBOM056",
            "MAOMBOM057",
            "MAOMBOM058",
            "MAOMBOM059",
            "MAOMBOM060",
            "MAOMBOM061",
        ]
    },
    **{
        tag: {
            "tipo": "Bomba",
            "subtipo": "Água Gelada Primária",
            "rpm_min": 1650,
            "rpm_max": 1750,
            "rpm_nominal": 1700,
            "aceita_lubrificacao": False,
            "lowcut": 2500,
            "highcut": 8000,
            "lp_env": 500,
        }
        for tag in [
            "MAOMBOM001",
            "MAOMBOM002",
            "MAOMBOM003",
            "MAOMBOM009",
            "MAOMBOM010",
            "MAOMBOM029",
            "MAOMBOM030",
            "MAOMBOM031",
        ]
    },
    **{
        tag: {
            "tipo": "Bomba",
            "subtipo": "Água Condensada",
            "rpm_min": 1500,
            "rpm_max": 1650,
            "rpm_nominal": 1575,
            "aceita_lubrificacao": True,
            "lowcut": 2500,
            "highcut": 8000,
            "lp_env": 500,
        }
        for tag in [
            "MAOMBOM004",
            "MAOMBOM005",
            "MAOMBOM006",
            "MAOMBOM007",
            "MAOMBOM008",
            "MAOMBOM015",
            "MAOMBOM016",
            "MAOMBOM017",
        ]
    },
}

CONFIG_PADRAO_OUTROS = {
    "Bomba": {
        "subtipo": "Não Especificado",
        "rpm_min": 1500,
        "rpm_max": 1750,
        "rpm_nominal": 1750,
        "aceita_lubrificacao": True,
        "lowcut": 2500,
        "highcut": 8000,
        "lp_env": 500,
    },
    "Moto Redutor": {
        "subtipo": "Moto Redutor",
        "rpm_min": 1500,
        "rpm_max": 1750,
        "rpm_nominal": 1750,
        "aceita_lubrificacao": True,
        "lowcut": 3000,
        "highcut": 9500,
        "lp_env": 500,
    },
}

RPM_BASE_REFERENCIA = 1750.0
FREQUENCIAS_FALHA_BASE = {"BPFO": 83.0, "BPFI": 128.4, "BSF": 53.1}
NOMES_COMPONENTES = {
    "BPFO": "Pista Externa (BPFO)",
    "BPFI": "Pista Interna (BPFI)",
    "BSF": "Elementos Rolantes (BSF)",
}
CORES_FALHA = {"BPFO": "#dc2626", "BPFI": "#2563eb", "BSF": "#16a34a"}


def obter_agora_manaus():
  return datetime.now(TZ_MANAUS)


def carregar_logo_vinci():
  if os.path.exists(LOGO_VINCI_PATH):
    try:
      return Image.open(LOGO_VINCI_PATH).convert("RGBA")
    except Exception:
      return None
  return None


def obter_perfil_ativo(trecho, tipo_selecionado):
  if trecho in CADASTRO_BOMBAS:
    perfil = dict(CADASTRO_BOMBAS[trecho])
    perfil["cadastrado"] = True
    return perfil
  perfil = dict(
      CONFIG_PADRAO_OUTROS.get(
          tipo_selecionado, CONFIG_PADRAO_OUTROS["Bomba"]
      )
  )
  perfil["tipo"] = tipo_selecionado
  perfil["cadastrado"] = False
  return perfil


logo_vinci = carregar_logo_vinci()

# --- CABEÇALHO WEB COM MARCA VINCI E TEMA AEROPORTUÁRIO ---
col_logo, col_header = st.columns([1, 3.2], vertical_alignment="center")
with col_logo:
  if logo_vinci is not None:
    st.image(logo_vinci, use_container_width=True)
  else:
    st.markdown(
        """
        <div style="background:#0f294a;border:1px solid #38bdf8;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:16px;font-weight:800;letter-spacing:2px;color:#ffffff;">VINCI</div>
            <div style="font-size:9px;letter-spacing:1px;color:#7dd3fc;">AIRPORTS OPS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_header:
  st.markdown(
      """
      <div class="aero-banner" style="margin-bottom:0;">
          <div class="aero-tag">✈️ MANAUS • OPERAÇÃO AEROPORTUÁRIA • MANUTENÇÃO PREDITIVA</div>
          <div class="aero-title">Análise Acústica de Rolamentos</div>
          <div class="aero-sub">Envoltória por Welch (Mediana), rotação por tag e emissão de laudo técnico.</div>
      </div>
      """,
      unsafe_allow_html=True,
  )

st.write("")

# --- 1. CÓDIGO DO TRECHO / TAG (TRAVADO EM MAIÚSCULAS) ---
trecho_input = st.text_input(
    "📍 Código do Trecho / Tag (Apenas letras maiúsculas — Ex: MAOMBOM061):",
    value="",
    max_chars=10,
    placeholder="EX: MAOMBOM061",
)
trecho_selecionado = trecho_input.upper().strip()

# --- 2. TIPO DE ATIVO ---
tipo_ativo_manual = st.radio(
    "⚙️ Tipo de Equipamento:",
    options=["Bomba", "Moto Redutor"],
    horizontal=True,
)

perfil_ativo = obter_perfil_ativo(trecho_selecionado, tipo_ativo_manual)
tipo_ativo = perfil_ativo["tipo"]

if trecho_selecionado:
  lub_texto = (
      "Aceita lubrificação"
      if perfil_ativo["aceita_lubrificacao"]
      else "Não aceita lubrificação"
  )
  origem_tag = (
      "Tag cadastrada"
      if perfil_ativo["cadastrado"]
      else "Tag não cadastrada (usando faixa padrão)"
  )
  st.info(
      f"**Ativo:** {tipo_ativo} ({perfil_ativo['subtipo']}) • **Faixa de"
      f" Rotação:** {perfil_ativo['rpm_min']} a {perfil_ativo['rpm_max']} RPM"
      f" • **Lubrificação:** {lub_texto}\n\n*{origem_tag}*"
  )

# --- 3. UPLOAD DE ÁUDIO (INCLUINDO MPEG) ---
uploaded_file = st.file_uploader(
    "🎙️ Selecione o áudio do estetoscópio (.mp3, .wav ou .mpeg)",
    type=["mp3", "wav", "mpeg"],
)


def enviar_para_planilha(
    data_hora, tipo, subtipo, trecho, rpm_faixa, nome_arquivo, status_geral
):
  if "SUA_URL" in WEBHOOK_SHEETS_URL:
    return
  payload = {
      "data_hora": data_hora,
      "trecho": trecho,
      "arquivo": nome_arquivo,
      "status": status_geral,
      "obs": (
          f"Ativo: {tipo} ({subtipo}) | {rpm_faixa} | Welch Mediana | Horário"
          " Manaus (AMT)"
      ),
  }
  try:
    requests.post(WEBHOOK_SHEETS_URL, json=payload, timeout=5)
  except Exception as e:
    print(f"Erro ao enviar para planilha: {e}")


def isolar_envelope_rolamento(y, sr, lowcut, highcut, lp_env):
  y = y - np.mean(y)
  max_val = np.max(np.abs(y))
  if max_val > 0:
    y = y / max_val

  nyq = 0.5 * sr
  high_efetivo = min(highcut, nyq * 0.9)
  low_efetivo = min(lowcut, high_efetivo * 0.7)

  sos_band = butter(
      4, [low_efetivo / nyq, high_efetivo / nyq], btype="band", output="sos"
  )
  sinal_isolado = sosfiltfilt(sos_band, y)

  envelope = np.abs(hilbert(sinal_isolado))

  if lp_env < nyq:
    sos_lp = butter(4, lp_env / nyq, btype="low", output="sos")
    envelope = sosfiltfilt(sos_lp, envelope)

  return sinal_isolado, envelope, low_efetivo, high_efetivo


def calcular_espectro_welch_mediana(envelope, sr):
  env_centered = envelope - np.mean(envelope)
  nperseg = min(len(env_centered), int(sr * 2.0))
  nperseg = max(256, nperseg)
  noverlap = nperseg // 2

  freqs, pxx = welch(
      env_centered,
      fs=sr,
      window="hann",
      nperseg=nperseg,
      noverlap=noverlap,
      scaling="spectrum",
      average="median",
  )
  mag = np.sqrt(np.maximum(pxx, 0.0))
  return freqs, mag


def analisar_espectro_envoltoria(fft_freqs, fft_mag, rpm_min, rpm_max):
  fator_min = rpm_min / RPM_BASE_REFERENCIA
  fator_max = rpm_max / RPM_BASE_REFERENCIA
  resultados = {}

  for codigo, freq_ref in FREQUENCIAS_FALHA_BASE.items():
    f_low = freq_ref * fator_min
    f_high = freq_ref * fator_max

    mask_1x = (fft_freqs >= f_low) & (fft_freqs <= f_high)
    margem = max(12.0, (f_high - f_low) * 1.5)
    mask_viz = (
        (fft_freqs >= max(5.0, f_low - margem))
        & (fft_freqs <= f_high + margem)
        & (~mask_1x)
    )

    ruido_fundo = (
        float(np.median(fft_mag[mask_viz]))
        if np.any(mask_viz)
        else float(np.median(fft_mag[(fft_freqs >= 10) & (fft_freqs <= 350)]))
    )
    ruido_fundo = max(ruido_fundo, 1e-9)

    if np.any(mask_1x):
      freqs_1x = fft_freqs[mask_1x]
      mags_1x = fft_mag[mask_1x]
      idx_max_1x = int(np.argmax(mags_1x))
      pico_1x = float(mags_1x[idx_max_1x])
      freq_pico_1x = float(freqs_1x[idx_max_1x])
      snr_1x = pico_1x / ruido_fundo
    else:
      pico_1x, freq_pico_1x, snr_1x = 0.0, (f_low + f_high) / 2.0, 0.0

    f2_low, f2_high = f_low * 2.0, f_high * 2.0
    mask_2x = (fft_freqs >= f2_low) & (fft_freqs <= f2_high)
    mask_viz_2x = (
        (fft_freqs >= max(5.0, f2_low - margem))
        & (fft_freqs <= f2_high + margem)
        & (~mask_2x)
    )
    ruido_2x = (
        float(np.median(fft_mag[mask_viz_2x]))
        if np.any(mask_viz_2x)
        else ruido_fundo
    )
    ruido_2x = max(ruido_2x, 1e-9)

    if np.any(mask_2x):
      freqs_2x = fft_freqs[mask_2x]
      mags_2x = fft_mag[mask_2x]
      idx_max_2x = int(np.argmax(mags_2x))
      pico_2x = float(mags_2x[idx_max_2x])
      freq_pico_2x = float(freqs_2x[idx_max_2x])
      snr_2x = pico_2x / ruido_2x
    else:
      pico_2x, freq_pico_2x, snr_2x = 0.0, (f2_low + f2_high) / 2.0, 0.0

    snr_max = max(snr_1x, snr_2x)
    harmonico_dominante = "1x" if snr_1x >= snr_2x else "2x"
    freq_dominante = freq_pico_1x if snr_1x >= snr_2x else freq_pico_2x
    pico_dominante = pico_1x if snr_1x >= snr_2x else pico_2x

    if snr_max >= 2.5:
      suspeita_falha = True
      badge = "ATENÇÃO / SUSPEITA"
      resumo_tecnico = (
          f"Pico destacado em {freq_dominante:.1f} Hz ({harmonico_dominante}),"
          f" atingindo {snr_max:.1f}x o ruído de fundo. Pode indicar estágio"
          " inicial de falha na pista ou interferência de fonte externa."
          " Exige nova auscultação para confirmação."
      )
    else:
      suspeita_falha = False
      badge = "ÍNTEGRA"
      resumo_tecnico = (
          f"Sem picos relevantes acima do ruído de fundo (máx. {snr_max:.1f}x"
          f" em {freq_dominante:.1f} Hz). Comportamento espectral compatível"
          " com pista íntegra."
      )

    resultados[codigo] = {
        "f_low": f_low,
        "f_high": f_high,
        "snr_max": snr_max,
        "freq_dominante": freq_dominante,
        "pico_dominante": pico_dominante,
        "suspeita_falha": suspeita_falha,
        "badge": badge,
        "resumo_tecnico": resumo_tecnico,
    }

  return resultados


def aplicar_marca_vinci_graficos(fig, logo_img, data_assinatura):
  header_ax = fig.add_axes([0.0, 0.965, 1.0, 0.035])
  header_ax.set_facecolor("#0a1f3d")
  header_ax.set_xticks([])
  header_ax.set_yticks([])
  for spine in header_ax.spines.values():
    spine.set_visible(False)

  header_ax.text(
      0.015,
      0.5,
      "VINCI AIRPORTS  |  INSPEÇÃO PREDITIVA AEROPORTUÁRIA — MANAUS (AMT)",
      color="#e0f2fe",
      fontsize=9.5,
      fontweight="bold",
      va="center",
      ha="left",
      family="DejaVu Sans",
  )
  header_ax.text(
      0.985,
      0.5,
      f"Emissão: {data_assinatura}",
      color="#7dd3fc",
      fontsize=9,
      va="center",
      ha="right",
      family="DejaVu Sans",
  )

  if logo_img is not None:
    logo_ax = fig.add_axes([0.88, 0.91, 0.10, 0.05])
    logo_ax.axis("off")
    logo_ax.imshow(logo_img)


def gerar_pagina_laudo(
    trecho,
    tipo,
    subtipo,
    nome_arquivo,
    data_assinatura,
    rpm_faixa_str,
    aceita_lub,
    low_ef,
    high_ef,
    fator_crista,
    analise_pistas,
    pistas_com_alerta,
    baixa_lubrificacao,
    logo_img,
):
  """Gera a Página 2 (Laudo Técnico) em formato A4 centralizado e profissional."""
  fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
  ax = fig.add_axes([0, 0, 1, 1])
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.axis("off")

  fonte = "DejaVu Sans"
  cor_navy = "#0a1f3d"
  cor_azul = "#0284c7"
  cor_texto = "#1e293b"
  cor_subtexto = "#475569"
  cor_borda = "#cbd5e1"
  cor_fundo_card = "#f8fafc"

  # --- 1. BARRA SUPERIOR INSTITUCIONAL ---
  ax.add_patch(
      FancyBboxPatch(
          (0.0, 0.945),
          1.0,
          0.055,
          boxstyle="square,pad=0",
          facecolor=cor_navy,
          edgecolor="none",
      )
  )
  ax.text(
      0.05,
      0.973,
      "VINCI AIRPORTS  |  INSPEÇÃO PREDITIVA AEROPORTUÁRIA",
      color="#f8fafc",
      fontsize=9.5,
      fontweight="bold",
      va="center",
      ha="left",
      family=fonte,
  )
  ax.text(
      0.95,
      0.973,
      f"Emissão: {data_assinatura}",
      color="#7dd3fc",
      fontsize=8.2,
      va="center",
      ha="right",
      family=fonte,
  )

  # --- 2. TÍTULO CENTRALIZADO E LOGO ---
  if logo_img is not None:
    logo_ax = fig.add_axes([0.81, 0.875, 0.13, 0.055])
    logo_ax.axis("off")
    logo_ax.imshow(logo_img)

  ax.text(
      0.50,
      0.908,
      "LAUDO TÉCNICO DE INSPEÇÃO PREDITIVA",
      color=cor_navy,
      fontsize=14.5,
      fontweight="bold",
      va="center",
      ha="center",
      family=fonte,
  )
  ax.text(
      0.50,
      0.885,
      "Análise Acústica de Rolamentos por Envoltória — Aeroporto de Manaus"
      " (MAO)",
      color=cor_subtexto,
      fontsize=8.8,
      va="center",
      ha="center",
      family=fonte,
  )
  ax.plot([0.06, 0.94], [0.868, 0.868], color=cor_azul, linewidth=1.6)

  # --- 3. SEÇÃO 1: IDENTIFICAÇÃO DO ATIVO (CARD CENTRALIZADO) ---
  ax.text(
      0.06,
      0.848,
      "1. IDENTIFICAÇÃO DO ATIVO E PARÂMETROS DE COLETA",
      color=cor_navy,
      fontsize=9.5,
      fontweight="bold",
      va="center",
      ha="left",
      family=fonte,
  )
  ax.add_patch(
      FancyBboxPatch(
          (0.06, 0.725),
          0.88,
          0.112,
          boxstyle="round,pad=0.008,rounding_size=0.01",
          facecolor=cor_fundo_card,
          edgecolor=cor_borda,
          linewidth=0.9,
      )
  )

  col_esq = [
      ("Tag / Trecho:", trecho),
      ("Equipamento:", f"{tipo} — {subtipo}"),
      ("Faixa de Rotação:", rpm_faixa_str),
      ("Aceita Lubrificação:", "Sim" if aceita_lub else "Não (Rolamento Selado)"),
  ]
  col_dir = [
      ("Unidade:", "Aeroporto de Manaus (MAO)"),
      ("Banda Isolada:", f"{low_ef:.0f} Hz a {high_ef:.0f} Hz"),
      ("Método Espectral:", "Welch (Mediana)"),
      ("Fator de Crista:", f"{fator_crista:.2f}"),
  ]

  y_pos = 0.818
  for (lbl_e, val_e), (lbl_d, val_d) in zip(col_esq, col_dir):
    ax.text(
        0.08,
        y_pos,
        lbl_e,
        color=cor_subtexto,
        fontsize=8.3,
        fontweight="bold",
        va="center",
        ha="left",
        family=fonte,
    )
    ax.text(
        0.245,
        y_pos,
        val_e,
        color=cor_texto,
        fontsize=8.3,
        va="center",
        ha="left",
        family=fonte,
    )
    ax.text(
        0.53,
        y_pos,
        lbl_d,
        color=cor_subtexto,
        fontsize=8.3,
        fontweight="bold",
        va="center",
        ha="left",
        family=fonte,
    )
    ax.text(
        0.695,
        y_pos,
        val_d,
        color=cor_texto,
        fontsize=8.3,
        va="center",
        ha="left",
        family=fonte,
    )
    y_pos -= 0.023

  ax.text(
      0.08,
      0.733,
      f"Arquivo analisado: {nome_arquivo}",
      color=cor_subtexto,
      fontsize=7.6,
      style="italic",
      va="center",
      ha="left",
      family=fonte,
  )

  # --- 4. SEÇÃO 2: INTEGRIDADE DAS PISTAS (BPFO, BPFI, BSF) ---
  ax.text(
      0.06,
      0.695,
      "2. AVALIAÇÃO ESPECTRAL POR COMPONENTE (INTEGRIDADE DAS PISTAS)",
      color=cor_navy,
      fontsize=9.5,
      fontweight="bold",
      va="center",
      ha="left",
      family=fonte,
  )

  y_card = 0.600
  for codigo in ["BPFO", "BPFI", "BSF"]:
    info = analise_pistas[codigo]
    alerta = info["suspeita_falha"]
    cor_status = "#b91c1c" if alerta else "#15803d"
    fundo_badge = "#fee2e2" if alerta else "#dcfce7"
    borda_card = "#fca5a5" if alerta else cor_borda

    ax.add_patch(
        FancyBboxPatch(
            (0.06, y_card),
            0.88,
            0.080,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            facecolor="white",
            edgecolor=borda_card,
            linewidth=0.9,
        )
    )
    # Faixa lateral colorida do componente
    ax.add_patch(
        FancyBboxPatch(
            (0.06, y_card),
            0.008,
            0.080,
            boxstyle="square,pad=0",
            facecolor=CORES_FALHA[codigo],
            edgecolor="none",
        )
    )

    titulo_comp = (
        f"{NOMES_COMPONENTES[codigo]}   •   Janela:"
        f" {info['f_low']:.2f} a {info['f_high']:.2f} Hz"
    )
    ax.text(
        0.082,
        y_card + 0.061,
        titulo_comp,
        color=cor_navy,
        fontsize=8.6,
        fontweight="bold",
        va="center",
        ha="left",
        family=fonte,
    )

    # Selo de status à direita
    ax.add_patch(
        FancyBboxPatch(
            (0.755, y_card + 0.050),
            0.172,
            0.022,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            facecolor=fundo_badge,
            edgecolor="none",
        )
    )
    ax.text(
        0.841,
        y_card + 0.061,
        info["badge"],
        color=cor_status,
        fontsize=7.3,
        fontweight="bold",
        va="center",
        ha="center",
        family=fonte,
    )

    texto_wrap = textwrap.fill(info["resumo_tecnico"], width=95)
    ax.text(
        0.082,
        y_card + 0.041,
        texto_wrap,
        color=cor_texto,
        fontsize=7.9,
        va="top",
        ha="left",
        family=fonte,
        linespacing=1.35,
    )
    y_card -= 0.095

  # --- 5. SEÇÃO 3: PARECER TÉCNICO CONSOLIDADO ---
  ax.text(
      0.06,
      0.388,
      "3. PARECER TÉCNICO CONSOLIDADO (PISTAS E LUBRIFICAÇÃO)",
      color=cor_navy,
      fontsize=9.5,
      fontweight="bold",
      va="center",
      ha="left",
      family=fonte,
  )

  houve_alerta = bool(pistas_com_alerta) or baixa_lubrificacao
  fundo_parecer = "#fff7ed" if houve_alerta else "#f0fdf4"
  borda_parecer = "#f97316" if houve_alerta else "#22c55e"
  cor_titulo_parecer = "#9a3412" if houve_alerta else "#166534"

  ax.add_patch(
      FancyBboxPatch(
          (0.06, 0.245),
          0.88,
          0.128,
          boxstyle="round,pad=0.008,rounding_size=0.01",
          facecolor=fundo_parecer,
          edgecolor=borda_parecer,
          linewidth=1.1,
      )
  )

  if pistas_com_alerta:
    txt_pistas = (
        f"Integridade das Pistas: Foram identificados picos destacados em"
        f" {', '.join(pistas_com_alerta)}. O comportamento pode corresponder a"
        " início de falha no rolamento ou a interferência externa (cavitação,"
        " fluxo hidráulico ou carga mecânica adjacente)."
    )
  else:
    txt_pistas = (
        "Integridade das Pistas: Todas as pistas avaliadas (BPFO, BPFI e BSF)"
        " apresentaram-se íntegras, sem assinaturas periódicas de início de"
        " falha no espectro de envoltória."
    )

  if baixa_lubrificacao and aceita_lub:
    txt_lub = (
        "Condição de Lubrificação: Detectado indício de baixa lubrificação"
        " (atrito/impulsividade elevada) em rolamento que aceita"
        " relubrificação. Deve ser feita a lubrificação e realizada uma nova"
        " auscultação."
    )
  elif baixa_lubrificacao and not aceita_lub:
    txt_lub = (
        "Condição de Lubrificação: Detectado nível elevado de impulsividade /"
        " atrito na banda de alta frequência. Este ativo possui rolamento que"
        " não aceita lubrificação, exigindo acompanhamento criterioso."
    )
  elif aceita_lub:
    txt_lub = (
        "Condição de Lubrificação: Parâmetros de atrito e impulsividade dentro"
        " da normalidade. Sem necessidade de intervenção imediata de"
        " lubrificação."
    )
  else:
    txt_lub = (
        "Condição de Lubrificação: Rolamento selado (não aceita lubrificação),"
        " operando com níveis estáveis na faixa avaliada."
    )

  titulo_cond = (
      "CONDIÇÃO GERAL: ATENÇÃO — VERIFICAÇÃO EM CAMPO EXIGIDA"
      if houve_alerta
      else "CONDIÇÃO GERAL: SAUDÁVEL — OPERAÇÃO NORMAL"
  )
  ax.text(
      0.08,
      0.355,
      titulo_cond,
      color=cor_titulo_parecer,
      fontsize=8.8,
      fontweight="bold",
      va="center",
      ha="left",
      family=fonte,
  )
  ax.text(
      0.08,
      0.335,
      textwrap.fill(txt_pistas, width=96),
      color=cor_texto,
      fontsize=8.0,
      va="top",
      ha="left",
      family=fonte,
      linespacing=1.35,
  )
  ax.text(
      0.08,
      0.285,
      textwrap.fill(txt_lub, width=96),
      color=cor_texto,
      fontsize=8.0,
      va="top",
      ha="left",
      family=fonte,
      linespacing=1.35,
  )

  # --- 6. SEÇÃO 4: DIRETRIZES E EXIGÊNCIAS DE CAMPO ---
  ax.text(
      0.06,
      0.215,
      "4. DIRETRIZES E EXIGÊNCIAS TÉCNICAS DE MANUTENÇÃO",
      color=cor_navy,
      fontsize=9.5,
      fontweight="bold",
      va="center",
      ha="left",
      family=fonte,
  )
  ax.add_patch(
      FancyBboxPatch(
          (0.06, 0.085),
          0.88,
          0.115,
          boxstyle="round,pad=0.008,rounding_size=0.01",
          facecolor=cor_fundo_card,
          edgecolor=cor_borda,
          linewidth=0.9,
      )
  )

  acoes = []
  if baixa_lubrificacao and aceita_lub:
    acoes.append(
        "• LUBRIFICAÇÃO OBRIGATÓRIA: Executar a lubrificação do rolamento"
        " conforme plano técnico e realizar nova auscultação logo após a"
        " estabilização."
    )
  if pistas_com_alerta or baixa_lubrificacao:
    acoes.append(
        "• NOVA AUSCULTAÇÃO EXIGIDA: O técnico deve repetir a coleta com o"
        " sensor firmemente apoiado no mancal (mín. 10 segundos) para"
        " confirmar se há falha real ou pico de fonte externa."
    )
  else:
    acoes.append(
        "• MANUTENÇÃO PREVENTIVA: Manter o ativo no ciclo normal de"
        " acompanhamento preditivo periódico."
    )
  acoes.append(
      "• RASTREABILIDADE: Manter o registro desta inspeção na planilha de"
      " controle operacional da unidade para acompanhamento de tendência."
  )

  y_acao = 0.186
  for acao in acoes:
    wrapped = textwrap.fill(acao, width=96)
    ax.text(
        0.08,
        y_acao,
        wrapped,
        color=cor_texto,
        fontsize=8.0,
        va="top",
        ha="left",
        family=fonte,
        linespacing=1.35,
    )
    y_acao -= 0.036

  # --- 7. RODAPÉ CENTRALIZADO ---
  ax.plot([0.06, 0.94], [0.055, 0.055], color=cor_borda, linewidth=0.8)
  ax.text(
      0.50,
      0.035,
      "Documento gerado automaticamente • Sistema de Manutenção Preditiva"
      " VINCI Airports — Manaus (MAO)",
      color=cor_subtexto,
      fontsize=7.5,
      va="center",
      ha="center",
      family=fonte,
  )

  return fig


def processar_automatico(caminho_audio, trecho, perfil):
  nome_arquivo = os.path.basename(caminho_audio)
  agora_manaus = obter_agora_manaus()
  data_assinatura = agora_manaus.strftime("%d/%m/%Y às %H:%M:%S (Manaus)")

  tipo = perfil["tipo"]
  subtipo = perfil["subtipo"]
  rpm_min = perfil["rpm_min"]
  rpm_max = perfil["rpm_max"]
  rpm_faixa_str = f"{rpm_min} a {rpm_max} RPM"
  aceita_lub = perfil["aceita_lubrificacao"]

  y, sr = librosa.load(caminho_audio, sr=None)
  sinal_isolado, envelope, low_ef, high_ef = isolar_envelope_rolamento(
      y, sr, perfil["lowcut"], perfil["highcut"], perfil["lp_env"]
  )

  n_fft = 2048
  D = librosa.stft(sinal_isolado, n_fft=n_fft, hop_length=512)
  S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

  fft_freqs, fft_mag = calcular_espectro_welch_mediana(envelope, sr)
  analise_pistas = analisar_espectro_envoltoria(
      fft_freqs, fft_mag, rpm_min, rpm_max
  )

  # --- PÁGINA 1: GRÁFICOS ---
  fig_graficos = plt.figure(figsize=(12, 11.5))
  plt.suptitle(
      f"ANÁLISE ISOLADA DE ROLAMENTO ({tipo.upper()} - {subtipo.upper()}) |"
      f" TRECHO: {trecho}\nRotação: {rpm_faixa_str} | Arquivo: {nome_arquivo}",
      fontsize=12,
      fontweight="bold",
      color="#0a1f3d",
      y=0.945,
  )

  ax1 = plt.subplot(3, 1, 1)
  tempo = np.linspace(0, len(y) / sr, len(envelope))
  ax1.plot(tempo, envelope, color="#0284c7", alpha=0.9)
  ax1.set_title(
      "1. Envelope Isolado de Rolamento (Banda"
      f" {low_ef:.0f} Hz - {high_ef:.0f} Hz)"
  )
  ax1.set_ylabel("Amplitude")
  ax1.grid(True, linestyle="--", alpha=0.6)

  ax2 = plt.subplot(3, 1, 2)
  img = librosa.display.specshow(
      S_db, sr=sr, x_axis="time", y_axis="log", cmap="magma", ax=ax2
  )
  ax2.set_ylim(low_ef, high_ef)
  ax2.set_title(
      "2. Espectrograma Isolado (Faixa de Rolamento:"
      f" {low_ef:.0f} Hz - {high_ef:.0f} Hz)"
  )
  ax2.set_ylabel("Freq (Hz)")
  plt.colorbar(img, ax=ax2, format="%+2.0f dB")

  ax3 = plt.subplot(3, 1, 3)
  ax3.plot(
      fft_freqs,
      fft_mag,
      color="#0f172a",
      linewidth=1.2,
      label="Espectro de Envoltória (Welch Mediana)",
      zorder=3,
  )

  for codigo, info in analise_pistas.items():
    cor = CORES_FALHA.get(codigo, "black")
    f_low, f_high = info["f_low"], info["f_high"]

    ax3.axvspan(
        f_low,
        f_high,
        color=cor,
        alpha=0.16,
        zorder=1,
        label=f"Região {codigo} ({rpm_faixa_str})",
    )
    ax3.axvline(
        x=f_high, color=cor, linestyle="--", linewidth=1.4, alpha=0.8, zorder=2
    )
    ax3.axvspan(f_low * 2, f_high * 2, color=cor, alpha=0.07, zorder=1)
    ax3.axvline(
        x=f_high * 2,
        color=cor,
        linestyle=":",
        alpha=0.5,
        linewidth=1,
        zorder=2,
    )

    if info["suspeita_falha"]:
      ax3.scatter(
          [info["freq_dominante"]],
          [info["pico_dominante"]],
          color=cor,
          s=55,
          zorder=5,
          edgecolors="black",
      )
      ax3.annotate(
          f"{codigo}: {info['freq_dominante']:.1f} Hz",
          xy=(info["freq_dominante"], info["pico_dominante"]),
          xytext=(4, 6),
          textcoords="offset points",
          fontsize=8,
          fontweight="bold",
          color=cor,
      )

  ax3.set_title(
      "3. Espectro de Envoltória — Método de Welch/Mediana (Faixa"
      f" {rpm_faixa_str})"
  )
  ax3.set_xlabel("Frequência (Hz)")
  ax3.set_ylabel("Amplitude")
  ax3.set_xlim(0, 350)
  ax3.grid(True, linestyle="--", alpha=0.6)
  ax3.legend(loc="upper right", framealpha=0.92, fontsize=8.5)

  plt.tight_layout(rect=[0, 0, 1, 0.89])
  aplicar_marca_vinci_graficos(fig_graficos, logo_vinci, data_assinatura)

  # --- INDICADORES DE LUBRIFICAÇÃO / IMPULSIVIDADE ---
  rms_env = np.sqrt(np.mean(envelope**2))
  pico_env = np.max(envelope)
  fator_crista = (pico_env / rms_env) if rms_env > 0 else 0.0

  freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
  idx_lub = np.where((freqs >= low_ef) & (freqs <= min(high_ef, 5000)))[0]
  energia_lubrificacao = (
      np.mean(S_db[idx_lub, :]) if len(idx_lub) > 0 else -100
  )

  baixa_lubrificacao = (fator_crista > 6.0) or (energia_lubrificacao > -32.0)
  pistas_com_alerta = [
      cod for cod, dados in analise_pistas.items() if dados["suspeita_falha"]
  ]

  if pistas_com_alerta and (baixa_lubrificacao and aceita_lub):
    status_geral = (
        f"ALERTA ({', '.join(pistas_com_alerta)} + Baixa Lubrificação): Pode"
        " ser início de falha ou pico externo; lubrificar e realizar nova"
        " auscultação."
    )
  elif pistas_com_alerta:
    status_geral = (
        f"ALERTA ({', '.join(pistas_com_alerta)}): Pode ser início de falha ou"
        " pico de outra fonte — EXIGIDA nova auscultação para confirmação."
    )
  elif baixa_lubrificacao and aceita_lub:
    status_geral = (
        "PISTAS ÍNTEGRAS | ALERTA LUBRIFICAÇÃO: Deve ser feita a lubrificação"
        " e realizar uma nova auscultação."
    )
  elif baixa_lubrificacao:
    status_geral = (
        "PISTAS ÍNTEGRAS | ATENÇÃO ATRITO: Impulsividade elevada (não aceita"
        " lubrificação) — realizar nova auscultação."
    )
  else:
    status_geral = (
        "SAUDÁVEL: Pistas íntegras e sem sinais de início de falha ou baixa"
        " lubrificação."
    )

  # --- PÁGINA 2: LAUDO TÉCNICO CENTRALIZADO ---
  fig_laudo = gerar_pagina_laudo(
      trecho=trecho,
      tipo=tipo,
      subtipo=subtipo,
      nome_arquivo=nome_arquivo,
      data_assinatura=data_assinatura,
      rpm_faixa_str=rpm_faixa_str,
      aceita_lub=aceita_lub,
      low_ef=low_ef,
      high_ef=high_ef,
      fator_crista=fator_crista,
      analise_pistas=analise_pistas,
      pistas_com_alerta=pistas_com_alerta,
      baixa_lubrificacao=baixa_lubrificacao,
      logo_img=logo_vinci,
  )

  # --- UNIFICAR EM UM ÚNICO PDF (2 PÁGINAS) ---
  caminho_pdf_unificado = f"VINCI_{trecho}_{tipo.replace(' ', '_')}_Relatorio.pdf"
  with PdfPages(caminho_pdf_unificado) as pdf:
    pdf.savefig(fig_graficos, bbox_inches="tight")
    pdf.savefig(fig_laudo)

  plt.close(fig_graficos)
  plt.close(fig_laudo)

  enviar_para_planilha(
      data_assinatura,
      tipo,
      subtipo,
      trecho,
      rpm_faixa_str,
      nome_arquivo,
      status_geral,
  )
  return caminho_pdf_unificado, status_geral, data_assinatura


if uploaded_file is not None:
  caminho_temp = uploaded_file.name
  with open(caminho_temp, "wb") as f:
    f.write(uploaded_file.getbuffer())

  if st.button("🚀 Gerar Relatório Técnico"):
    if not trecho_selecionado:
      st.error(
          "⚠️ Por favor, preencha o código do trecho / tag antes de gerar o"
          " relatório."
      )
    else:
      with st.spinner(
          f"Analisando {trecho_selecionado} com Welch/Mediana"
          f" ({perfil_ativo['rpm_min']} a {perfil_ativo['rpm_max']} RPM)..."
      ):
        pdf_gerado, status, emissao_str = processar_automatico(
            caminho_temp, trecho_selecionado, perfil_ativo
        )

      st.success("✅ Relatório emitido e registrado com sucesso!")
      st.info(
          f"**Ativo:** {tipo_ativo} ({perfil_ativo['subtipo']}) | **Trecho:**"
          f" `{trecho_selecionado}`\n\n"
          f"**Rotação:** {perfil_ativo['rpm_min']} a"
          f" {perfil_ativo['rpm_max']} RPM | **Emissão:** {emissao_str}\n\n"
          f"**Diagnóstico:** {status}"
      )

      with open(pdf_gerado, "rb") as file:
        st.download_button(
            label="📥 Baixar Relatório Completo (PDF - 2 Páginas)",
            data=file,
            file_name=pdf_gerado,
            mime="application/pdf",
        )

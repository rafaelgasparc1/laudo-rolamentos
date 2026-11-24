from datetime import datetime
import os
from zoneinfo import ZoneInfo
import librosa
import librosa.display
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
from PIL import Image
import requests
from scipy.signal import butter, hilbert, sosfiltfilt
import streamlit as st

# --- URL DO SEU WEBHOOK DO GOOGLE SHEETS PESSOAL ---
WEBHOOK_SHEETS_URL = "SUA_URL_DO_GOOGLE_APPS_SCRIPT_AQUI"

# Caminho opcional para o logotipo da VINCI no seu repositório GitHub
LOGO_VINCI_PATH = "vinci_logo.png"

# Fuso horário oficial de Manaus (UTC-4)
TZ_MANAUS = ZoneInfo("America/Manaus")

st.set_page_config(
    page_title="VINCI | Inspeção Preditiva Aeroportuária",
    page_icon="✈️",
    layout="centered",
)

# --- ESTILO VISUAL AEROPORTUÁRIO PROFISSIONAL ---
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


def obter_agora_manaus():
  return datetime.now(TZ_MANAUS)


def carregar_logo_vinci():
  if os.path.exists(LOGO_VINCI_PATH):
    try:
      return Image.open(LOGO_VINCI_PATH).convert("RGBA")
    except Exception:
      return None
  return None


logo_vinci = carregar_logo_vinci()

# --- CABEÇALHO COM MARCA VINCI E TEMA AEROPORTUÁRIO ---
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
          <div class="aero-sub">Isolamento de sinal, diagnóstico por faixa de RPM e emissão de relatório técnico.</div>
      </div>
      """,
      unsafe_allow_html=True,
  )

st.write("")

# --- 1. TIPO DE ATIVO (BOMBA OU MOTO REDUTOR) ---
tipo_ativo = st.radio(
    "⚙️ Tipo de Equipamento:",
    options=["Bomba", "Moto Redutor"],
    horizontal=True,
)

# --- 2. CÓDIGO DO TRECHO (FORÇA MAIÚSCULAS) ---
trecho_input = st.text_input(
    "📍 Código do Trecho (Ex: MAOAAAA123):",
    value="",
    max_chars=10,
    placeholder="Ex: MAOESLP054",
)
trecho_selecionado = trecho_input.upper().strip()

# --- 3. UPLOAD DE ÁUDIO (INCLUINDO MPEG) ---
uploaded_file = st.file_uploader(
    "🎙️ Selecione o áudio do estetoscópio (.mp3, .wav ou .mpeg)",
    type=["mp3", "wav", "mpeg"],
)

CONFIG_ATIVOS = {
    "Bomba": {"lowcut": 2500, "highcut": 8000, "lp_env": 500},
    "Moto Redutor": {"lowcut": 3000, "highcut": 9500, "lp_env": 500},
}


def enviar_para_planilha(data_hora, tipo, trecho, nome_arquivo, status_lub):
  if "SUA_URL" in WEBHOOK_SHEETS_URL:
    return
  payload = {
      "data_hora": data_hora,
      "trecho": trecho,
      "arquivo": nome_arquivo,
      "status": status_lub,
      "obs": f"Ativo: {tipo} | Horário Manaus (AMT) | Relatório VINCI",
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


def aplicar_marca_vinci_figura(fig, logo_img, data_assinatura):
  """Adiciona cabeçalho/marca VINCI no topo da figura do relatório."""
  # Faixa superior discreta estilo briefing operacional aeroportuário
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
  )
  header_ax.text(
      0.985,
      0.5,
      f"Emissão: {data_assinatura}",
      color="#7dd3fc",
      fontsize=9,
      va="center",
      ha="right",
  )

  if logo_img is not None:
    logo_ax = fig.add_axes([0.88, 0.91, 0.10, 0.05])
    logo_ax.axis("off")
    logo_ax.imshow(logo_img)


def processar_automatico(caminho_audio, tipo, trecho):
  nome_arquivo = os.path.basename(caminho_audio)
  agora_manaus = obter_agora_manaus()
  data_assinatura = agora_manaus.strftime("%d/%m/%Y às %H:%M:%S (Manaus)")

  cfg = CONFIG_ATIVOS.get(tipo, CONFIG_ATIVOS["Bomba"])
  y, sr = librosa.load(caminho_audio, sr=None)
  sinal_isolado, envelope, low_ef, high_ef = isolar_envelope_rolamento(
      y, sr, cfg["lowcut"], cfg["highcut"], cfg["lp_env"]
  )

  n_fft = 2048
  D = librosa.stft(sinal_isolado, n_fft=n_fft, hop_length=512)
  S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

  env_centered = envelope - np.mean(envelope)
  janela = np.hanning(len(env_centered))
  fft_vals = np.fft.rfft(env_centered * janela)
  fft_freqs = np.fft.rfftfreq(len(env_centered), d=1 / sr)
  fft_mag = (np.abs(fft_vals) * 2.0) / np.sum(janela)

  # --- 1. GERAR GRÁFICOS (PÁGINA 1) ---
  fig_graficos = plt.figure(figsize=(12, 11.5))
  plt.suptitle(
      f"ANÁLISE ISOLADA DE ROLAMENTO ({tipo.upper()}) - TRECHO:"
      f" {trecho}\nArquivo: {nome_arquivo}",
      fontsize=12.5,
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
      label="Espectro de Envoltória",
      zorder=3,
  )

  frequencias_falha = {"BPFO": 83.0, "BPFI": 128.4, "BSF": 53.1}
  cores = {"BPFO": "red", "BPFI": "blue", "BSF": "green"}
  fator_min = 1500 / 1750

  for nome, freq_ref in frequencias_falha.items():
    cor = cores.get(nome, "black")
    freq_min = freq_ref * fator_min
    ax3.axvspan(
        freq_min,
        freq_ref,
        color=cor,
        alpha=0.15,
        zorder=1,
        label=f"Região {nome} (1500-1750 RPM)",
    )
    ax3.axvline(
        x=freq_ref, color=cor, linestyle="--", linewidth=1.5, alpha=0.8, zorder=2
    )
    ax3.axvspan(freq_min * 2, freq_ref * 2, color=cor, alpha=0.07, zorder=1)
    ax3.axvline(
        x=freq_ref * 2, color=cor, linestyle=":", alpha=0.5, linewidth=1, zorder=2
    )

  ax3.set_title(
      "3. Espectro de Envoltória (Zonas Sombreadas: Margem 1500-1750 RPM)"
  )
  ax3.set_xlabel("Frequência (Hz)")
  ax3.set_ylabel("Amplitude")
  ax3.set_xlim(0, 300)
  ax3.grid(True, linestyle="--", alpha=0.6)
  ax3.legend(loc="upper right", framealpha=0.92, fontsize=9)

  plt.tight_layout(rect=[0, 0, 1, 0.89])
  aplicar_marca_vinci_figura(fig_graficos, logo_vinci, data_assinatura)
  caminho_graficos = "relatorio_visual_graficos.png"
  plt.savefig(caminho_graficos, dpi=150, bbox_inches="tight")

  # --- 2. DIAGNÓSTICO E LAUDO ESCRITO (PÁGINA 2) ---
  rms_env = np.sqrt(np.mean(envelope**2))
  pico_env = np.max(envelope)
  fator_crista = (pico_env / rms_env) if rms_env > 0 else 0.0

  freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
  idx_lub = np.where((freqs >= low_ef) & (freqs <= min(high_ef, 5000)))[0]
  energia_lubrificacao = (
      np.mean(S_db[idx_lub, :]) if len(idx_lub) > 0 else -100
  )

  if fator_crista > 6.0 or energia_lubrificacao > -32.0:
    status_lub = (
        f"ALERTA: Impulsividade/atrito elevado ({tipo} - banda isolada)."
    )
  else:
    status_lub = f"SAUDÁVEL: Comportamento estável ({tipo} - banda isolada)."

  fig_laudo, ax_laudo = plt.subplots(figsize=(10, 11.5))
  ax_laudo.axis("off")

  selo_texto = "" if logo_vinci is not None else "[ VINCI AIRPORTS ]\n"
  texto_laudo = (
      "========================================================================\n"
      f"{selo_texto}       LAUDO TÉCNICO DE INSPEÇÃO PREDITIVA — OPERAÇÃO"
      " AEROPORTUÁRIA     \n"
      "========================================================================\n\n"
      f"Unidade / Localidade: Aeroporto de Manaus (MAO) — Horário Local AMT\n"
      f"Tipo de Ativo: {tipo}\n"
      f"Identificação do Trecho: {trecho}\n"
      f"Arquivo de Áudio: {nome_arquivo}\n"
      f"Data e Hora da Emissão: {data_assinatura}\n"
      f"Banda Isolada: {low_ef:.0f} Hz a {high_ef:.0f} Hz | Fator de Crista:"
      f" {fator_crista:.2f}\n"
      "Parâmetro Operacional Alvo: Faixa dinâmica variável de 1500 a 1750 RPM\n\n"
      "------------------------------------------------------------------------\n"
      "1. DIAGNÓSTICO DETALHADO POR COMPONENTE / FAIXA DE FALHA:\n"
      "------------------------------------------------------------------------\n"
      "🔴 PISTA EXTERNA (BPFO) - Janela: 71.10 Hz a 83.00 Hz\n"
      "   • Status: Verificar pico no espectro de envoltória (1x e 2x).\n\n"
      "🔵 PISTA INTERNA (BPFI) - Janela: 110.00 Hz a 128.40 Hz\n"
      "   • Status: Verificar pico no espectro de envoltória (1x e 2x).\n\n"
      "🟢 ELEMENTOS ROLANTES (BSF) - Janela: 45.50 Hz a 53.10 Hz\n"
      "   • Status: Verificar pico no espectro de envoltória (1x e 2x).\n\n"
      "------------------------------------------------------------------------\n"
      "2. DIAGNÓSTICO FINAL DA CONDIÇÃO DO ATIVO:\n"
      "------------------------------------------------------------------------\n"
      f"⚠️ CONDIÇÃO GERAL: {status_lub}\n"
      "   • Filtro aplicado conforme o tipo de ativo selecionado.\n\n"
      "------------------------------------------------------------------------\n"
      "3. RECOMENDAÇÃO TÉCNICA DE MANUTENÇÃO:\n"
      "------------------------------------------------------------------------\n"
      "   • Comparar picos nas regiões sombreadas (1500-1750 RPM).\n"
      "   • Registrar histórico e reavaliar conforme plano de manutenção.\n"
      "========================================================================"
  )
  ax_laudo.text(
      0.01,
      0.92,
      texto_laudo,
      fontsize=10,
      family="monospace",
      verticalalignment="top",
      horizontalalignment="left",
      bbox=dict(
          boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#0f294a"
      ),
  )

  aplicar_marca_vinci_figura(fig_laudo, logo_vinci, data_assinatura)
  caminho_laudo = "relatorio_escrito_laudo.png"
  plt.savefig(caminho_laudo, dpi=150, bbox_inches="tight")

  # --- 3. UNIFICAR EM UM ÚNICO PDF (2 PÁGINAS) ---
  caminho_pdf_unificado = f"VINCI_{trecho}_{tipo.replace(' ', '_')}_Relatorio.pdf"
  with PdfPages(caminho_pdf_unificado) as pdf:
    pdf.savefig(fig_graficos, bbox_inches="tight")
    pdf.savefig(fig_laudo, bbox_inches="tight")

  plt.close(fig_graficos)
  plt.close(fig_laudo)

  enviar_para_planilha(data_assinatura, tipo, trecho, nome_arquivo, status_lub)
  return caminho_pdf_unificado, status_lub, data_assinatura


if uploaded_file is not None:
  caminho_temp = uploaded_file.name
  with open(caminho_temp, "wb") as f:
    f.write(uploaded_file.getbuffer())

  if st.button("🚀 Gerar Relatório Técnico"):
    if not trecho_selecionado:
      st.error(
          "⚠️ Por favor, preencha o código do trecho antes de gerar o"
          " relatório."
      )
    else:
      with st.spinner(
          "Processando áudio no horário de Manaus e gerando relatório"
          " VINCI..."
      ):
        pdf_gerado, status, emissao_str = processar_automatico(
            caminho_temp, tipo_ativo, trecho_selecionado
        )

      st.success("✅ Relatório emitido e registrado com sucesso!")
      st.info(
          f"**Ativo:** {tipo_ativo} | **Trecho:** `{trecho_selecionado}`\n\n"
          f"**Emissão:** {emissao_str}\n\n"
          f"**Diagnóstico:** {status}"
      )

      with open(pdf_gerado, "rb") as file:
        st.download_button(
            label="📥 Baixar Relatório Completo (PDF - 2 Páginas)",
            data=file,
            file_name=pdf_gerado,
            mime="application/pdf",
        )

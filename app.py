from datetime import datetime
import os
import librosa
import librosa.display
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.signal import butter, hilbert, sosfiltfilt
import streamlit as st

# --- URL DO SEU WEBHOOK DO GOOGLE SHEETS PESSOAL ---
WEBHOOK_SHEETS_URL = "SUA_URL_DO_GOOGLE_APPS_SCRIPT_AQUI"

st.set_page_config(
    page_title="Laudo Automático de Rolamentos", page_icon="🔧", layout="centered"
)

st.title("🔧 Análise Preditiva - Campo")
st.write(
    "Escolha o tipo de ativo, informe o código do trecho e envie o áudio do"
    " estetoscópio."
)

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

# --- 3. UPLOAD DE ÁUDIO ---
uploaded_file = st.file_uploader(
    "🎙️ Toque para selecionar o áudio (.mp3 ou .wav)", type=["mp3", "wav"]
)

# Configuração de bandas por tipo de ativo para isolar rolamento
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
      "obs": f"Ativo: {tipo} | Análise isolada de rolamento",
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


def processar_automatico(caminho_audio, tipo, trecho):
  nome_arquivo = os.path.basename(caminho_audio)
  data_assinatura = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")

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

  # --- 1. GERAR GRÁFICOS ---
  fig_graficos = plt.figure(figsize=(12, 11))
  plt.suptitle(
      f"ANÁLISE ISOLADA DE ROLAMENTO ({tipo.upper()}) - TRECHO:"
      f" {trecho}\n📁 {nome_arquivo} | 📅 {data_assinatura}",
      fontsize=13,
      fontweight="bold",
      y=0.96,
  )

  ax1 = plt.subplot(3, 1, 1)
  tempo = np.linspace(0, len(y) / sr, len(envelope))
  ax1.plot(tempo, envelope, color="darkorange", alpha=0.85)
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
      color="purple",
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
  ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)

  plt.tight_layout(rect=[0, 0, 1, 0.94])
  caminho_graficos = "relatorio_visual_graficos.png"
  plt.savefig(caminho_graficos, dpi=150, bbox_inches="tight")

  # --- 2. DIAGNÓSTICO E LAUDO ESCRITO ---
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

  fig_laudo, ax_laudo = plt.subplots(figsize=(10, 11))
  ax_laudo.axis("off")
  texto_laudo = (
      "========================================================================\n"
      "                   LAUDO TÉCNICO DE INSPEÇÃO PREDITIVA                 \n"
      "========================================================================\n\n"
      f"Tipo de Ativo: {tipo}\n"
      f"Identificação do Trecho: {trecho}\n"
      f"Arquivo de Áudio: {nome_arquivo}\n"
      f"Data da Geração: {data_assinatura}\n"
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
      0.98,
      texto_laudo,
      fontsize=10,
      family="monospace",
      verticalalignment="top",
      horizontalalignment="left",
      bbox=dict(
          boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="darkgray"
      ),
  )

  caminho_laudo = "relatorio_escrito_laudo.png"
  plt.savefig(caminho_laudo, dpi=150, bbox_inches="tight")

  # --- 3. UNIFICAR EM UM ÚNICO PDF (2 PÁGINAS) ---
  caminho_pdf_unificado = f"{trecho}_{tipo.replace(' ', '_')}_Relatorio.pdf"
  with PdfPages(caminho_pdf_unificado) as pdf:
    pdf.savefig(fig_graficos, bbox_inches="tight")
    pdf.savefig(fig_laudo, bbox_inches="tight")

  plt.close(fig_graficos)
  plt.close(fig_laudo)

  enviar_para_planilha(data_assinatura, tipo, trecho, nome_arquivo, status_lub)
  return caminho_pdf_unificado, status_lub


if uploaded_file is not None:
  caminho_temp = uploaded_file.name
  with open(caminho_temp, "wb") as f:
    f.write(uploaded_file.getbuffer())

  if st.button("🚀 Processar Análise"):
    if not trecho_selecionado:
      st.error(
          "⚠️ Por favor, preencha o código do trecho antes de processar a"
          " análise."
      )
    else:
      with st.spinner(
          f"Processando {tipo_ativo} e gerando relatório de 2 páginas..."
      ):
        pdf_gerado, status = processar_automatico(
            caminho_temp, tipo_ativo, trecho_selecionado
        )

      st.success("✅ Análise concluída e registrada com sucesso!")
      st.info(
          f"**Ativo:** {tipo_ativo} | **Trecho:** `{trecho_selecionado}` \n\n"
          f" **Diagnóstico:** {status}"
      )

      with open(pdf_gerado, "rb") as file:
        st.download_button(
            label="📥 Baixar Relatório Completo (PDF - 2 Páginas)",
            data=file,
            file_name=pdf_gerado,
            mime="application/pdf",
        )

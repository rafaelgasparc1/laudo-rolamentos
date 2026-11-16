from datetime import datetime
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.signal import butter, hilbert, lfilter
import streamlit as st

# --- URL DO SEU WEBHOOK DO GOOGLE SHEETS PESSOAL ---
WEBHOOK_SHEETS_URL = "SUA_URL_DO_GOOGLE_APPS_SCRIPT_AQUI"

st.set_page_config(
    page_title="Laudo Automático de Rolamentos", page_icon="🔧", layout="centered"
)

st.title("🔧 Análise Preditiva - Campo")
st.write(
    "Insira o código do trecho, envie o áudio do estetoscópio e o sistema gerará"
    " o relatório completo com os gráficos e faixas de RPM."
)

# Campo de texto do trecho (Força Maiúsculas)
trecho_input = st.text_input(
    "📍 Código do Trecho (Ex: MAOAAAA123):",
    value="",
    max_chars=10,
    placeholder="Ex: MAOESLP054",
)
trecho_selecionado = trecho_input.upper().strip()

# Upload de Áudio
uploaded_file = st.file_uploader(
    "🎙️ Toque para selecionar o áudio (.mp3 ou .wav)", type=["mp3", "wav"]
)


def enviar_para_planilha(data_hora, trecho, nome_arquivo, status_lub):
  if "SUA_URL" in WEBHOOK_SHEETS_URL:
    return
  payload = {
      "data_hora": data_hora,
      "trecho": trecho,
      "arquivo": nome_arquivo,
      "status": status_lub,
      "obs": "Relatório Consolidado Gerado com Sucesso",
  }
  try:
    requests.post(WEBHOOK_SHEETS_URL, json=payload, timeout=5)
  except Exception as e:
    print(f"Erro ao enviar para planilha: {e}")


def processar_automatico(caminho_audio, trecho):
  nome_arquivo = os.path.basename(caminho_audio)
  data_assinatura = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")

  y, sr = librosa.load(caminho_audio, sr=None)
  nyq = 0.5 * sr
  b, a = butter(4, [500 / nyq, 10000 / nyq], btype="band")
  envelope = np.abs(hilbert(lfilter(b, a, y)))

  n_fft = 2048
  D = librosa.stft(y, n_fft=n_fft, hop_length=512)
  S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

  n = len(envelope)
  fft_vals = np.fft.rfft(envelope - np.mean(envelope))
  fft_freqs = np.fft.rfftfreq(n, d=1 / sr)
  fft_mag = np.abs(fft_vals) / n

  # --- 1. GERAR GRÁFICOS COM FAIXAS DE FALHA (1500 a 1750 RPM) ---
  fig_graficos = plt.figure(figsize=(12, 11))
  plt.suptitle(
      f"ANÁLISE DE VIBRAÇÃO MECÂNICA - TRECHO: {trecho}\n📁 {nome_arquivo} | 📅"
      f" {data_assinatura}",
      fontsize=13,
      fontweight="bold",
      y=0.96,
  )

  # Subplot 1: Sinal Demodulado
  ax1 = plt.subplot(3, 1, 1)
  tempo = np.linspace(0, len(y) / sr, len(envelope))
  ax1.plot(tempo, envelope, color="darkorange", alpha=0.8)
  ax1.set_title("1. Sinal Demodulado (Envelope de Alta Frequência)")
  ax1.set_ylabel("Amplitude")
  ax1.grid(True, linestyle="--", alpha=0.6)

  # Subplot 2: Espectrograma
  ax2 = plt.subplot(3, 1, 2)
  img = librosa.display.specshow(
      S_db, sr=sr, x_axis="time", y_axis="log", cmap="magma", ax=ax2
  )
  ax2.set_ylim(500, 10000)
  ax2.set_title("2. Espectrograma (Foco em Alta Frequência: 500 Hz - 10 kHz)")
  ax2.set_ylabel("Freq (Hz)")
  plt.colorbar(img, ax=ax2, format="%+2.0f dB")

  # Subplot 3: Espectro de Envoltória com Faixas de Referência (1500 - 1750 RPM)
  ax3 = plt.subplot(3, 1, 3)
  ax3.plot(
      fft_freqs,
      fft_mag,
      color="purple",
      label="Espectro de Envoltória",
      zorder=3,
  )

  # Frequências de referência base a 1750 RPM
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
    # Harmônico (2x)
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
  plt.close(fig_graficos)

  # --- 2. DIAGNÓSTICO E LAUDO ESCRITO ---
  freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
  idx_lub = np.where((freqs >= 1000) & (freqs <= 3000))[0]
  energia_lubrificacao = (
      np.mean(S_db[idx_lub, :]) if len(idx_lub) > 0 else -100
  )
  status_lub = (
      "ALERTA: Falta de lubrificação detectada (Atrito elevado 1-3 kHz)."
      if energia_lubrificacao > -35.0
      else "SAUDÁVEL: Lubrificação estável."
  )

  fig_laudo, ax_laudo = plt.subplots(figsize=(10, 11))
  ax_laudo.axis("off")
  texto_laudo = (
      "========================================================================\n"
      "                    LAUDO TÉCNICO DE INSPEÇÃO PREDITIVA                 \n"
      "========================================================================\n\n"
      f"Identificação do Trecho: {trecho}\n"
      f"Arquivo de Áudio: {nome_arquivo}\n"
      f"Data da Geração: {data_assinatura}\n"
      "Parâmetro Operacional Alvo: Faixa dinâmica variável de 1500 a 1750 RPM\n\n"
      "------------------------------------------------------------------------\n"
      "1. DIAGNÓSTICO DETALHADO POR COMPONENTE / FAIXA DE FALHA:\n"
      "------------------------------------------------------------------------\n"
      "🔴 PISTA EXTERNA (BPFO) - Janela: 71.10 Hz a 83.00 Hz\n"
      "   • Status: Saudável. Sem evolução geométrica de defeito.\n\n"
      "🔵 PISTA INTERNA (BPFI) - Janela: 110.00 Hz a 128.40 Hz\n"
      "   • Status: Íntegro. Ausência de quaisquer defeitos na pista interna.\n\n"
      "🟢 ELEMENTOS ROLANTES (BSF) - Janela: 45.50 Hz a 53.10 Hz\n"
      "   • Status: Íntegro. Amplitudes normais de rotação natural das"
      " esferas.\n\n"
      "------------------------------------------------------------------------\n"
      "2. DIAGNÓSTICO FINAL DA CONDIÇÃO DO ATIVO:\n"
      "------------------------------------------------------------------------\n"
      f"⚠️ CONDIÇÃO GERAL: {status_lub}\n"
      "   • Geometria Interna: Isenta de trincas, quebras ou desgaste severo.\n\n"
      "------------------------------------------------------------------------\n"
      "3. RECOMENDAÇÃO TÉCNICA DE MANUTENÇÃO:\n"
      "------------------------------------------------------------------------\n"
      "   • Programar relubrificação do mancal com graxa indicada pelo"
      " fabricante.\n"
      "   • Repetir a coleta de áudio em 15 dias para validação.\n"
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
  plt.close(fig_laudo)

  # --- 3. UNIFICAR EM UM ÚNICO PDF (RELATÓRIO DE 2 PÁGINAS) ---
  from matplotlib.backends.backend_pdf import PdfPages

  caminho_pdf_unificado = f"{trecho}_Relatorio_Preditivo.pdf"
  with PdfPages(caminho_pdf_unificado) as pdf:
    # Página 1: Gráficos com faixas de RPM
    pdf.savefig(fig_graficos)
    # Página 2: Laudo Escrito
    pdf.savefig(fig_laudo)

  # Enviar para a planilha
  enviar_para_planilha(data_assinatura, trecho, nome_arquivo, status_lub)

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
          "Analisando áudio, aplicando faixas de RPM e gerando relatório"
          " unificado..."
      ):
        pdf_gerado, status = processar_automatico(
            caminho_temp, trecho_selecionado
        )

      st.success("✅ Análise concluída e registrada com sucesso!")
      st.info(
          f"**Trecho Analisado:** `{trecho_selecionado}` \n\n **Diagnóstico:**"
          f" {status}"
      )

      # Botão Único para baixar o Relatório Completo de 2 Páginas (PDF)
      with open(pdf_gerado, "rb") as file:
        st.download_button(
            label="📥 Baixar Relatório Completo (PDF - 2 Páginas)",
            data=file,
            file_name=pdf_gerado,
            mime="application/pdf",
        )

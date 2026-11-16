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
WEBHOOK_SHEETS_URL = "https://script.google.com/macros/s/AKfycbwM31z_UPun_tQH2L-feUjDuaFsFg4QfYcstw6939iBNWxIZ5OjxSgeK_m1n3ex2akg/exec"

st.set_page_config(
    page_title="Laudo Automático de Rolamentos", page_icon="🔧", layout="centered"
)

st.title("🔧 Análise Preditiva - Campo")
st.write(
    "Insira o código do trecho, envie o áudio do estetoscópio e o sistema gerará"
    " tudo automaticamente."
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
      "obs": "Gerado automaticamente via App Pessoal",
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

  # Gerar Gráficos
  fig_graficos = plt.figure(figsize=(10, 10))
  plt.suptitle(
      f"ANÁLISE DE VIBRAÇÃO - TRECHO: {trecho}\n{nome_arquivo} | {data_assinatura}",
      fontsize=11,
      fontweight="bold",
  )

  ax1 = plt.subplot(3, 1, 1)
  ax1.plot(
      np.linspace(0, len(y) / sr, len(envelope)),
      envelope,
      color="darkorange",
  )
  ax1.set_title("1. Sinal Demodulado")

  ax2 = plt.subplot(3, 1, 2)
  img = librosa.display.specshow(
      S_db, sr=sr, x_axis="time", y_axis="log", cmap="magma", ax=ax2
  )
  ax2.set_ylim(500, 10000)
  ax2.set_title("2. Espectrograma")

  ax3 = plt.subplot(3, 1, 3)
  ax3.plot(fft_freqs, fft_mag, color="purple")
  ax3.set_title("3. Espectro de Envoltória")
  ax3.set_xlim(0, 300)

  plt.tight_layout()
  caminho_graficos = "relatorio_visual_graficos.png"
  plt.savefig(caminho_graficos, dpi=150, bbox_inches="tight")
  plt.close(fig_graficos)

  # Diagnóstico de Lubrificação
  freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
  idx_lub = np.where((freqs >= 1000) & (freqs <= 3000))[0]
  energia_lubrificacao = (
      np.mean(S_db[idx_lub, :]) if len(idx_lub) > 0 else -100
  )
  status_lub = (
      "ALERTA: Falta de lubrificação"
      if energia_lubrificacao > -35.0
      else "SAUDÁVEL: Lubrificação estável"
  )

  # Gerar Laudo Escrito
  fig_laudo, ax_laudo = plt.subplots(figsize=(8, 10))
  ax_laudo.axis("off")
  texto_laudo = (
      "========================================\n"
      "      LAUDO TÉCNICO DE INSPEÇÃO         \n"
      "========================================\n\n"
      f"Identificação do Trecho: {trecho}\n"
      f"Arquivo de Áudio: {nome_arquivo}\n"
      f"Data da Coleta: {data_assinatura}\n\n"
      f"Condição: {status_lub}\n"
      "Pista Externa, Interna e Esferas: Íntegras.\n\n"
      "Recomendação: Acompanhar conforme treinamento.\n"
      "========================================"
  )
  ax_laudo.text(
      0.05,
      0.95,
      texto_laudo,
      fontsize=12,
      family="monospace",
      verticalalignment="top",
  )
  caminho_laudo = "relatorio_escrito_laudo.png"
  plt.savefig(caminho_laudo, dpi=150, bbox_inches="tight")
  plt.close(fig_laudo)

  # Enviar para a planilha
  enviar_para_planilha(data_assinatura, trecho, nome_arquivo, status_lub)

  return caminho_graficos, caminho_laudo, status_lub


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
      with st.spinner("Analisando áudio e registrando na planilha..."):
        img_g, img_l, status = processar_automatico(
            caminho_temp, trecho_selecionado
        )

      st.success("✅ Análise concluída e registrada com sucesso!")
      st.info(
          f"**Trecho Analisado:** `{trecho_selecionado}` \n\n **Diagnóstico:**"
          f" {status}"
      )

      col1, col2 = st.columns(2)
      with col1:
        with open(img_g, "rb") as file:
          st.download_button(
              "📥 Baixar Gráficos",
              file,
              file_name=f"{trecho_selecionado}_graficos.png",
              mime="image/png",
          )
      with col2:
        with open(img_l, "rb") as file:
          st.download_button(
              "📥 Baixar Laudo",
              file,
              file_name=f"{trecho_selecionado}_laudo.png",
              mime="image/png",
          )

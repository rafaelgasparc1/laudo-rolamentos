from datetime import datetime
import os
from zoneinfo import ZoneInfo
import librosa
import librosa.display
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from scipy.signal import butter, hilbert, sosfiltfilt
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
    "BPFO": "PISTA EXTERNA (BPFO)",
    "BPFI": "PISTA INTERNA (BPFI)",
    "BSF": "ELEMENTOS ROLANTES (BSF)",
}
CORES_FALHA = {"BPFO": "red", "BPFI": "blue", "BSF": "green"}


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
          <div class="aero-sub">Leitura espectral de envoltória, rotação por tag e emissão de laudo técnico.</div>
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
          f"Ativo: {tipo} ({subtipo}) | {rpm_faixa} | Horário Manaus (AMT)"
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


def analisar_espectro_envoltoria(fft_freqs, fft_mag, rpm_min, rpm_max):
  """Lê o espectro de envoltória nas janelas 1x e 2x de BPFO, BPFI e BSF

  e avalia se cada pista está íntegra ou apresenta sinal de início de falha.
  """
  fator_min = rpm_min / RPM_BASE_REFERENCIA
  fator_max = rpm_max / RPM_BASE_REFERENCIA
  resultados = {}

  for codigo, freq_ref in FREQUENCIAS_FALHA_BASE.items():
    f_low = freq_ref * fator_min
    f_high = freq_ref * fator_max

    # Janela fundamental (1x)
    mask_1x = (fft_freqs >= f_low) & (fft_freqs <= f_high)
    # Vizinhança local (excluindo 1x e 2x) para medir ruído de fundo real
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

    # Janela harmônica (2x)
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

    # Classificação de integridade da pista/componente
    if snr_max >= 2.5:
      suspeita_falha = True
      status_pista = (
          f"SINAL DE INÍCIO DE FALHA / PICO EXTERNO (Pico em"
          f" {freq_dominante:.1f} Hz [{harmonico_dominante}], {snr_max:.1f}x"
          " acima do ruído de fundo)."
      )
      detalhe_pista = (
          "   • Leitura: Pico destacado detectado — pode ser início de falha"
          " na pista ou pico vindo de outra fonte.\n"
          "   • Ação obrigatória: Exige nova auscultação para confirmar se há"
          " realmente falha."
      )
    else:
      suspeita_falha = False
      status_pista = (
          f"ÍNTEGRA (Sem picos relevantes; máx. {snr_max:.1f}x do ruído de"
          f" fundo em {freq_dominante:.1f} Hz)."
      )
      detalhe_pista = (
          "   • Leitura: Pista íntegra, dentro do comportamento normal de"
          " ruído de fundo."
      )

    resultados[codigo] = {
        "f_low": f_low,
        "f_high": f_high,
        "pico_1x": pico_1x,
        "freq_pico_1x": freq_pico_1x,
        "snr_1x": snr_1x,
        "pico_2x": pico_2x,
        "freq_pico_2x": freq_pico_2x,
        "snr_2x": snr_2x,
        "snr_max": snr_max,
        "freq_dominante": freq_dominante,
        "pico_dominante": pico_dominante,
        "suspeita_falha": suspeita_falha,
        "status_pista": status_pista,
        "detalhe_pista": detalhe_pista,
    }

  return resultados


def aplicar_marca_vinci_figura(fig, logo_img, data_assinatura):
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

  env_centered = envelope - np.mean(envelope)
  janela = np.hanning(len(env_centered))
  fft_vals = np.fft.rfft(env_centered * janela)
  fft_freqs = np.fft.rfftfreq(len(env_centered), d=1 / sr)
  fft_mag = (np.abs(fft_vals) * 2.0) / np.sum(janela)

  # --- LEITURA REAL DO ESPECTRO DE ENVOLTÓRIA ---
  analise_pistas = analisar_espectro_envoltoria(
      fft_freqs, fft_mag, rpm_min, rpm_max
  )

  # --- 1. GERAR GRÁFICOS (PÁGINA 1) ---
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
      label="Espectro de Envoltória",
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

    # Destaca visualmente no gráfico se houve pico suspeito na faixa
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
      "3. Espectro de Envoltória (Leitura Automática — Faixa"
      f" {rpm_faixa_str})"
  )
  ax3.set_xlabel("Frequência (Hz)")
  ax3.set_ylabel("Amplitude")
  ax3.set_xlim(0, 350)
  ax3.grid(True, linestyle="--", alpha=0.6)
  ax3.legend(loc="upper right", framealpha=0.92, fontsize=8.5)

  plt.tight_layout(rect=[0, 0, 1, 0.89])
  aplicar_marca_vinci_figura(fig_graficos, logo_vinci, data_assinatura)
  caminho_graficos = "relatorio_visual_graficos.png"
  plt.savefig(caminho_graficos, dpi=150, bbox_inches="tight")

  # --- 2. DIAGNÓSTICO DE PISTAS + LUBRIFICAÇÃO E LAUDO ESCRITO (PÁGINA 2) ---
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

  # Parecer consolidado das pistas
  if pistas_com_alerta:
    lista_pistas = ", ".join(pistas_com_alerta)
    parecer_pistas = (
        f"ALERTA NAS PISTAS ({lista_pistas}): Detectado sinal de início de"
        " falha ou pico vindo de outra fonte externa.\n"
        "   -> EXIGÊNCIA TÉCNICA: O técnico DEVE realizar uma nova"
        " auscultação para confirmar se há realmente uma falha."
    )
  else:
    parecer_pistas = (
        "PISTAS ÍNTEGRAS: Nenhuma assinatura de início de falha identificada"
        " nas janelas BPFO, BPFI e BSF."
    )

  # Parecer consolidado de lubrificação
  if baixa_lubrificacao and aceita_lub:
    status_lub = (
        "Baixa lubrificação detectada — deve ser feita a lubrificação e"
        " realizar uma nova auscultação."
    )
    acao_lub_texto = (
        "• LUBRIFICAÇÃO: Baixa lubrificação detectada (rolamento aceita"
        " lubrificação).\n"
        "   -> Deve ser feita a lubrificação e realizar uma nova auscultação."
    )
  elif baixa_lubrificacao and not aceita_lub:
    status_lub = (
        "Impulsividade/atrito elevado (Rolamento não aceita lubrificação)."
    )
    acao_lub_texto = (
        "• LUBRIFICAÇÃO: Atrito/impulsividade elevada, porém este rolamento"
        " NÃO aceita lubrificação.\n"
        "   -> Realizar nova auscultação de confirmação e acompanhar"
        " evolução."
    )
  elif aceita_lub:
    status_lub = "Lubrificação estável (Aceita lubrificação)."
    acao_lub_texto = (
        "• LUBRIFICAÇÃO: Sem indicativo de baixa lubrificação no momento"
        " (rolamento aceita lubrificação)."
    )
  else:
    status_lub = "Lubrificação/atrito estável (Não aceita lubrificação)."
    acao_lub_texto = "• LUBRIFICAÇÃO: Rolamento selado / não aceita lubrificação."

  # Status resumido para tela e planilha
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

  bpfo = analise_pistas["BPFO"]
  bpfi = analise_pistas["BPFI"]
  bsf = analise_pistas["BSF"]

  fig_laudo, ax_laudo = plt.subplots(figsize=(10, 12.0))
  ax_laudo.axis("off")

  selo_texto = "" if logo_vinci is not None else "[ VINCI AIRPORTS ]\n"
  texto_laudo = (
      "========================================================================\n"
      f"{selo_texto}       LAUDO TÉCNICO DE INSPEÇÃO PREDITIVA — OPERAÇÃO"
      " AEROPORTUÁRIA     \n"
      "========================================================================\n\n"
      f"Unidade / Localidade: Aeroporto de Manaus (MAO) — Horário Local AMT\n"
      f"Tipo de Ativo: {tipo} — {subtipo}\n"
      f"Identificação do Trecho / Tag: {trecho}\n"
      f"Arquivo de Áudio: {nome_arquivo}\n"
      f"Data e Hora da Emissão: {data_assinatura}\n"
      f"Faixa de Rotação do Ativo: {rpm_faixa_str} | Aceita Lubrificação:"
      f" {'Sim' if aceita_lub else 'Não'}\n"
      f"Banda Isolada: {low_ef:.0f} Hz a {high_ef:.0f} Hz | Fator de Crista:"
      f" {fator_crista:.2f}\n\n"
      "------------------------------------------------------------------------\n"
      "1. LEITURA AUTOMÁTICA DO ESPECTRO DE ENVOLTÓRIA (INTEGRIDADE DAS PISTAS):\n"
      "------------------------------------------------------------------------\n"
      f"🔴 PISTA EXTERNA (BPFO) [{bpfo['f_low']:.2f} a {bpfo['f_high']:.2f} Hz]:\n"
      f"   • Status: {bpfo['status_pista']}\n"
      f"{bpfo['detalhe_pista']}\n\n"
      f"🔵 PISTA INTERNA (BPFI) [{bpfi['f_low']:.2f} a {bpfi['f_high']:.2f} Hz]:\n"
      f"   • Status: {bpfi['status_pista']}\n"
      f"{bpfi['detalhe_pista']}\n\n"
      f"🟢 ELEMENTOS ROLANTES (BSF) [{bsf['f_low']:.2f} a {bsf['f_high']:.2f} Hz]:\n"
      f"   • Status: {bsf['status_pista']}\n"
      f"{bsf['detalhe_pista']}\n\n"
      "------------------------------------------------------------------------\n"
      "2. DIAGNÓSTICO FINAL (PISTAS E LUBRIFICAÇÃO):\n"
      "------------------------------------------------------------------------\n"
      f"• INTEGRIDADE: {parecer_pistas}\n"
      f"{acao_lub_texto}\n\n"
      "------------------------------------------------------------------------\n"
      "3. EXIGÊNCIA / RECOMENDAÇÃO TÉCNICA DE CAMPO:\n"
      "------------------------------------------------------------------------\n"
      + (
          "   • EXIGÊNCIA: Realizar NOVA AUSCULTAÇÃO posicionando o sensor"
          " firme\n"
          "     no mancal para confirmar se o pico é falha real ou fonte"
          " externa.\n"
          if (pistas_com_alerta or baixa_lubrificacao)
          else "   • Manter rotina periódica de inspeção preditiva.\n"
      )
      + "   • Registrar resultado na planilha de controle operacional.\n"
      "========================================================================"
  )
  ax_laudo.text(
      0.01,
      0.93,
      texto_laudo,
      fontsize=9.3,
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
          f"Lendo espectro de envoltória de {trecho_selecionado}"
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

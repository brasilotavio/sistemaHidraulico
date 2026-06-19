import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, step, ss2tf

# Configuração da página do site
st.set_page_config(page_title="Simulador Hidráulico Interativo", layout="wide")

st.title("Simulador Interativo: Sistema de Dois Tanques")
st.markdown(
    "Interface interativa voltada para a análise dinâmica e validação local de modelos lineares por matrizes Jacobianas.")
st.markdown("---")

# =============================================================================
# 1. PARAMETRIZAÇÃO VIA MATRÍCULA (SID) DO GRUPO
# =============================================================================
g = 9.81
alpha = 0.62
S = 570387  # Semente baseada no maior SID do grupo

rng = np.random.default_rng(seed=S)
delta = rng.standard_normal()

# Parâmetros base gerados automaticamente pelo SID
A1_base = 0.25 * (1 + delta)
A2_base = 0.50 * (1 + delta)
a_base = 0.01 * (1 + delta)

# Barra lateral com os controles interativos
st.sidebar.header("🎛️ Painel de Controle")
st.sidebar.markdown("Altere os parâmetros para recalcular os modelos e métricas instantaneamente.")

# Sliders de Operação
u_bar = st.sidebar.slider("Vazão Nominal ($\overline{u}$)", 0.001, 0.010, 0.005, step=0.001, format="%.3f m³/s")
porcentagem_degrau = st.sidebar.slider("Magnitude do Degrau (%)", -50, 100, 10, step=5)

# Lógica com 3 cenários: Redução de Vazão, Vazão Controlada e Vazão Crítica
if porcentagem_degrau < 0:
    caminho_gif = "../videos/sem_agua.gif"
    texto_informativo = "📉 Redução de Vazão (Degrau Negativo: Níveis dos tanques vão descer!)"
elif 0 <= porcentagem_degrau <= 50:
    caminho_gif = "../videos/abaixo_50.gif"
    texto_informativo = "🎥 Fluxo Controlado (Pequena Amplitude: Aproximação Linear Confiável)"
else:
    caminho_gif = "../videos/acima_50.gif"
    texto_informativo = "🎥 Fluxo Turbulento (Larga Escala: Modelo Linear perde a validade)"

# Exibe o texto explicativo na barra lateral
st.sidebar.info(texto_informativo)

# Exibe o GIF em loop correspondente
st.sidebar.image(caminho_gif, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Modificação Física dos Tubos")

# NOVO SLIDER: Modificação em tempo real da área dos orifícios (a)
a = st.sidebar.slider(
    label="Área dos Orifícios ($a$)",
    min_value=0.001,
    max_value=a_base * 3,
    value=a_base,
    step=0.001,
    format="%.4f m²"
)

# Mantendo as áreas das seções transversais conforme o SID
A1 = A1_base
A2 = A2_base

# Mostrar parâmetros dinâmicos na barra lateral
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Parâmetros Atuais Calculados")
st.sidebar.text(f"Área Tanque 1: {A1:.4f} m²")
st.sidebar.text(f"Área Tanque 2: {A2:.4f} m²")

# =============================================================================
# 2. CÁLCULO DO PONTO DE EQUILÍBRIO E MATRIZES
# =============================================================================
x2_bar = (u_bar ** 2) / (2 * g * (alpha * a) ** 2)
x1_bar = 2 * x2_bar
x_bar = np.array([x1_bar, x2_bar])

st.sidebar.text(f"Nível Equilíbrio h1: {x1_bar:.3f} m")
st.sidebar.text(f"Nível Equilíbrio h2: {x2_bar:.3f} m")

R = u_bar / ((alpha * a) ** 2 * g)

A = np.array([[-1 / (A1 * R), 1 / (A1 * R)], [1 / (A2 * R), -2 / (A2 * R)]])
B = np.array([[1 / A1], [0]])
C = np.array([[0, 1]])
D = np.array([[0]])


# =============================================================================
# 3. DEFINIÇÃO DOS MODELOS DINÂMICOS (EDOs)
# =============================================================================
def modelo_nao_linear(t, x, u_atual):
    x1, x2 = x
    termo_interativo = alpha * a * np.sqrt(2 * g * np.abs(x1 - x2)) * np.sign(x1 - x2)
    termo_saida = alpha * a * np.sqrt(2 * g * np.abs(x2)) * np.sign(x2)
    dx1_dt = (1 / A1) * u_atual - (1 / A1) * termo_interativo
    dx2_dt = (1 / A2) * termo_interativo - (1 / A2) * termo_saida
    return [dx1_dt, dx2_dt]


def modelo_linearizado(t, delta_x, delta_u):
    return A @ delta_x + B.flatten() * delta_u


# =============================================================================
# 4. SIMULAÇÃO DO DEGRAU SELECIONADO
# =============================================================================
t_span = (0, 150)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

fator_degrau = 1 + (porcentagem_degrau / 100)
u_degrau = fator_degrau * u_bar
delta_u = u_degrau - u_bar

sol_nl = solve_ivp(modelo_nao_linear, t_span, x_bar, args=(u_degrau,), t_eval=t_eval)
sol_l = solve_ivp(modelo_linearizado, t_span, np.array([0.0, 0.0]), args=(delta_u,), t_eval=t_eval)

h2_linear = sol_l.y[1] + x2_bar
h2_nao_linear = sol_nl.y[1]

# =============================================================================
# 5. LAYOUT VISUAL DO SITE - PARTE 1 (TANQUES E COMPARAÇÃO NO TEMPO)
# =============================================================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("💧 Nível Atual dos Tanques (Regime)")
    nivel_final_h1 = sol_nl.y[0][-1]
    nivel_final_h2 = sol_nl.y[1][-1]

    fig_tanques, ax_t = plt.subplots(figsize=(4, 5))
    ax_t.bar(["Tanque 1 (Sup)", "Tanque 2 (Inf)"], [nivel_final_h1, nivel_final_h2], color='#1f77b4', edgecolor='black',
             width=0.6)
    ax_t.set_ylabel("Altura da Água (metros)")
    ax_t.set_ylim(0, max(x1_bar * 1.8, 2.0))
    ax_t.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig_tanques)

    st.metric(label="Altura Final do Tanque 2", value=f"{nivel_final_h2:.4f} m",
              delta=f"{nivel_final_h2 - x2_bar:.4f} m")

with col2:
    st.subheader("📈 Resposta Dinâmica: Linear vs Não Linear")
    fig_curva, ax_c = plt.subplots(figsize=(8, 4.8))
    ax_c.plot(sol_nl.t, h2_nao_linear, 'b-', label='Modelo Real Não Linear', linewidth=2.5)
    ax_c.plot(sol_l.t, h2_linear, 'r--', label='Modelo Linear (Aproximação)', linewidth=2)
    ax_c.axhline(x2_bar, color='g', linestyle=':', label='Equilíbrio de Origem')

    ax_c.set_xlabel('Tempo (segundos)')
    ax_c.set_ylabel('Nível do Tanque 2 (m)')
    ax_c.title.set_text(f"Análise de Erro de Linearização para Degrau de {porcentagem_degrau}%")
    ax_c.grid(True, linestyle='--', alpha=0.7)
    ax_c.legend()
    st.pyplot(fig_curva)



st.markdown("---")

# =============================================================================
# 6. ANÁLISE DA FUNÇÃO DE TRANSFERÊNCIA E MAPA DE POLOS/ZEROS
# =============================================================================
st.header("🔬 Análise de Estabilidade e Desempenho (Domínio de Frequência)")

num, den = ss2tf(A, B, C, D)
num = num[0]
G = TransferFunction(num, den)

polos = np.roots(den)
zeros = np.roots(num)
ganho_dc = np.polyval(num, 0) / np.polyval(den, 0)

# Cálculos de desempenho temporais a partir da FT
t_calc = np.linspace(0, 150, 2000)
t_step, y_step = step(G, T=t_calc)

# Aplicando a magnitude do desvio do degrau real da simulação
y_step = delta_u * y_step
valor_final = y_step[-1]

# Evitando divisões por zero ou cálculos inconsistentes com entradas nulas
if np.abs(valor_final) > 1e-6:
    pico = np.max(y_step) if delta_u >= 0 else np.min(y_step)
    Mp = ((np.abs(pico) - np.abs(valor_final)) / np.abs(valor_final)) * 100

    y10 = 0.1 * valor_final
    y90 = 0.9 * valor_final
    idx10 = np.where(np.abs(y_step) >= np.abs(y10))[0][0]
    idx90 = np.where(np.abs(y_step) >= np.abs(y90))[0][0]
    tr = t_step[idx90] - t_step[idx10]

    banda = 0.02 * valor_final
    indices = np.where(np.abs(y_step - valor_final) > np.abs(banda))[0]
    ts = t_step[indices[-1] + 1] if len(indices) > 0 else 0
else:
    Mp, tr, ts = 0.0, 0.0, 0.0

col_metrics, col_mapa = st.columns([1, 1])

with col_metrics:
    st.subheader("📋 Métricas Extras da FT Linearizada")
    st.metric(label="Ganho DC ($K$)", value=f"{ganho_dc:.4f}")
    st.metric(label="Sobressinal Máximo ($M_p$)", value=f"{Mp:.2f} %")
    st.metric(label="Tempo de Subida ($t_r$)", value=f"{tr:.2f} s")
    st.metric(label="Tempo de Assentamento ($t_s$ - 2%)", value=f"{ts:.2f} s")

    st.markdown("**Polos do Sistema ($s$):**")
    for idx, p in enumerate(polos):
        st.write(f"• Polo {idx + 1}: `{p:.4f}`")

with col_mapa:
    st.subheader("📌 Mapa de Polos e Zeros")
    fig_mapa, ax_m = plt.subplots(figsize=(6, 4.2))
    ax_m.scatter(np.real(polos), np.imag(polos), marker='x', color='red', s=120, label='Polos', linewidth=2.5)

    # Remover zeros fantasmas numéricos da conversão ss2tf se existirem
    zeros_validos = [z for z in zeros if np.abs(np.polyval(num, z)) < 1e-5]
    if len(zeros_validos) > 0:
        ax_m.scatter(np.real(zeros_validos), np.imag(zeros_validos), marker='o', s=120, facecolors='none',
                     edgecolors='blue', label='Zeros', linewidth=2)

    ax_m.axhline(0, color='black', linewidth=1)
    ax_m.axvline(0, color='black', linewidth=1)
    ax_m.set_xlabel('Parte Real ($\sigma$)')
    ax_m.set_ylabel('Parte Imaginária ($j\omega$)')
    ax_m.grid(True, linestyle='--', alpha=0.5)
    ax_m.legend()
    st.pyplot(fig_mapa)

st.markdown("---")

# =============================================================================
# 7. CONTROLABILIDADE E OBSERVABILIDADE
# =============================================================================
st.header("🎛️ Propriedades Estruturais")

# Matriz de Controlabilidade
Mc = np.hstack((B, A @ B))
rank_C = np.linalg.matrix_rank(Mc)

# Matriz de Observabilidade
Mo = np.vstack((C, C @ A))
rank_O = np.linalg.matrix_rank(Mo)

col_c, col_o = st.columns(2)

with col_c:
    st.subheader("Matriz de Controlabilidade ($\mathcal{C}$)")
    st.code(str(Mc))
    st.metric(label="Posto de $\mathcal{C}$", value=rank_C)
    if rank_C == A.shape[0]:
        st.success("✅ Sistema Completamente Controlável")
    else:
        st.error("❌ Sistema Não Controlável")

with col_o:
    st.subheader("Matriz de Observabilidade ($\mathcal{O}$)")
    st.code(str(Mo))
    st.metric(label="Posto de $\mathcal{O}$", value=rank_O)
    if rank_O == A.shape[0]:
        st.success("✅ Sistema Completamente Observável")
    else:
        st.error("❌ Sistema Não Observável")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import control as ct

# Configuração da página
st.set_page_config(page_title="Análise de Estabilidade - Dois Tanques", layout="wide")

st.title("💧 Análise de Estabilidade e Dinâmica de Sistemas Hidráulicos")
st.markdown("""
Esta interface interativa explora o conceito de **Estabilidade Local e Global** em um sistema de dois tanques acoplados, 
comparando a planta em malha aberta com o comportamento compensado por um controlador Proporcional-Derivativo (PD).
""")
st.markdown("---")

# ----------------------------------------------------------------------
# SIDEBAR - PARAMETRIZAÇÃO INTERATIVA
# ----------------------------------------------------------------------
st.sidebar.header("🎛️ Parâmetros de Operação e Controle")

# Parâmetros Físicos Fixos (Validados do seu script de análise)
A1 = 0.237029
A2 = 0.474062
beta = 0.02603661895897048

st.sidebar.subheader("📍 Ponto de Operação")
u_bar = st.sidebar.slider("Vazão Nominal ($\overline{u}$)", 0.002, 0.008, 0.005, step=0.001, format="%.3f m³/s")

st.sidebar.subheader("📈 Perturbação (Entrada Bounded)")
amp_percent = st.sidebar.slider("Amplitude do Degrau ($\Delta u$ %)", 5, 100, 10, step=5)

st.sidebar.subheader("🎮 Sintonia do Controlador PD")
Kp = st.sidebar.slider("Ganho Proporcional ($K_p$)", 0.1, 2.0, 0.65, step=0.05)
Kd = st.sidebar.slider("Ganho Derivativo ($K_d$)", 0.1, 2.0, 0.688, step=0.05)

# ----------------------------------------------------------------------
# CÁLCULOS MATEMÁTICOS DE EQUILÍBRIO E JACOBIANO
# ----------------------------------------------------------------------
h2_bar = (u_bar / beta) ** 2
h1_bar = 2 * h2_bar
xbar = np.array([h1_bar, h2_bar])

# Resistência hidráulica local e Matriz Jacobiana A do analysis3.py
Rres = np.sqrt(h2_bar) / beta
p_ = 1 / (2 * A1 * Rres)
q_ = 1 / (2 * A2 * Rres)

# Matriz exata do script analysis3.py
Amat = np.array([[-p_, p_], [q_, -2 * q_]])
Bmat = np.array([[1 / A1], [0]])
Cmat = np.array([[0., 1.]])
Dmat = np.array([[0.]])

sys_ss = ct.ss(Amat, Bmat, Cmat, Dmat)
G = ct.ss2tf(sys_ss)
G = ct.minreal(G, verbose=False)
poles_ol = ct.poles(G)

# Configuração do Controlador e Malha Fechada
zc = Kp / Kd
Cctrl = ct.tf([Kd, Kp], [1])
L = Cctrl * G
Tcl = ct.feedback(L, 1)
poles_cl = ct.poles(Tcl)


# ----------------------------------------------------------------------
# SIMULAÇÃO PRELIMINAR (Para alimentar as Abas 3 e 4)
# ----------------------------------------------------------------------
def Q1f(h1, h2):
    d = h1 - h2
    return beta * np.sign(d) * np.sqrt(abs(d))


def Q2f(h2):
    return beta * np.sqrt(max(h2, 0.0))


r_step = (amp_percent / 100) * h2_bar  # referência de desvio de 10% nominal
Tend = 40.0
t_eval = np.linspace(0, Tend, 1000)


# EDO Malha Fechada Não Linear (Igual ao analysis3.py)
def f_cl_nl(t, x):
    h1, h2 = x
    e = r_step - (h2 - h2_bar)
    q1 = Q1f(h1, h2)
    q2 = Q2f(h2)
    dh2dt = (q1 - q2) / A2
    edot = -dh2dt
    u = max(u_bar + Kp * e + Kd * edot, 0.0)
    dh1dt = (u - q1) / A1
    return [dh1dt, dh2dt]


sol_cl = solve_ivp(f_cl_nl, [0, Tend], xbar, t_eval=t_eval, rtol=1e-8)
t_l, y_l = ct.step_response(Tcl * r_step, T=t_eval)

# ----------------------------------------------------------------------
# LAYOUT EM TABS
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 1. Estabilidade em Malha Aberta",
    "🎯 2. Estabilização e Alocação de Polos",
    "🌊 3. Simulação Temporal (Linear vs NL)",
    "💧 4. Nível dos Tanques (Visual)"
])

# --- TAB 1: MALHA ABERTA ---
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Análise Local de Lyapunov")
        st.markdown("O sistema linearizado em torno do ponto de equilíbrio possui a seguinte matriz Jacobiana $A$:")
        st.code(f"A = {np.round(Amat, 5).tolist()}")

        st.markdown("### Autovalores da Matriz Jacobiana (Polos do Sistema):")
        for i, p in enumerate(poles_ol):
            st.markdown(f"**$\lambda_{i + 1}$:** `{p.real:.5f}` 1/s")

        if np.all(np.real(poles_ol) < 0):
            st.success("✅ **Sistema Localmente Assintoticamente Estável**")
            st.caption("Todos os autovalores possuem parte real estritamente negativa ($\Re(\lambda_i) < 0$).")

    with col2:
        st.subheader("Discussão Física da Estabilidade")
        st.info("""
        **Por que a planta é naturalmente estável?**
        O sistema de tanques é um sistema físico **passivo e dissipativo**. A força motriz que escoa o fluido é a gravidade, 
        e o arrasto nos orifícios atua como uma resistência (fricção). Matematicamente, isso se reflete em autovalores 
        reais e puramente negativos. 

        Não existem elementos armazenadores de energia capazes de gerar oscilações sustentadas em malha aberta (como indutores ou molas), 
        o que impede a existência de autovalores complexos conjugados nesta configuração.
        """)

# --- TAB 2: MALHA FECHADA ---
with tab2:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("O Lugar das Raízes Compensado (Root Locus)")

        fig_rl, ax_rl = plt.subplots(figsize=(6, 4.5))
        a1c, a0c = G.den[0][0][1], G.den[0][0][2]
        b0 = G.num[0][0][-1]

        Kvec = np.linspace(0, Kd * 3, 2000)
        locus = np.array([np.roots([1, a1c + K * b0, a0c + K * b0 * zc]) for K in Kvec])

        ax_rl.plot(locus[:, 0].real, locus[:, 0].imag, color='steelblue', label='Lugar das Raízes')
        ax_rl.plot(locus[:, 1].real, locus[:, 1].imag, color='steelblue')
        ax_rl.plot([p.real for p in poles_ol], [p.imag for p in poles_ol], 'x', color='crimson', ms=10, mew=2,
                   label='Polos MA')
        ax_rl.plot(-zc, 0, '^', color='darkorange', ms=10, label='Zero do PD ($s = -z_c$)')
        ax_rl.plot([p.real for p in poles_cl], [p.imag for p in poles_cl], 's', color='darkgreen', ms=8,
                   label='Polos MF Atuais')

        ax_rl.axhline(0, color='k', lw=0.5)
        ax_rl.axvline(0, color='k', lw=0.5)
        ax_rl.set_xlabel(r'$\Re(s)$')
        ax_rl.set_ylabel(r'$\Im(s)$')
        ax_rl.grid(True, alpha=0.3)
        ax_rl.legend(fontsize=9)
        st.pyplot(fig_rl)

    with col2:
        st.subheader("Análise de Desempenho e Robustez")
        st.markdown(f"**Zero do Controlador ($z_c$):** `{-zc:.4f}`")
        st.markdown("**Polos em Malha Fechada:**")
        for p in poles_cl:
            st.markdown(f"• `{p.real:.4f} + {p.imag:.4f}j` 1/s")

        if abs(poles_cl[0].imag) > 1e-6:
            zeta = -poles_cl[0].real / abs(poles_cl[0])
            st.metric("Fator de Amortecimento ($\zeta$)", f"{zeta:.3f}")
        else:
            st.metric("Fator de Amortecimento ($\zeta$)", "1.000 (Supercrítico)")

        st.warning("""
        **O Efeito Estabilizador do Termo Derivativo:**
        Um controlador Proporcional puro ($K_p$) empurra os polos verticalmente no plano complexo, tornando o sistema altamente oscilatório 
        para ganhos elevados. 

        Ao introduzir a ação **Derivativa ($K_d$)**, adicionamos um **zero** no semiplano esquerdo. Esse zero atrai as ramificações do Lugar das Raízes, 
        "puxando" os polos de malha fechada para mais longe do eixo imaginário. Isso aumenta a taxa de decaimento exponencial da resposta transitória, 
        garantindo estabilidade com amortecimento controlado.
        """)

# --- TAB 3: SIMULAÇÃO TEMPORAL ---
with tab3:
    st.subheader("Verificação Dinâmica: Linear vs Não Linear")

    fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
    ax_sim.plot(t_eval, r_step * np.ones_like(t_eval), 'k:', label='Referência ($\Delta r$)')
    ax_sim.plot(sol_cl.t, sol_cl.y[1] - h2_bar, 'r--', lw=2, label='Malha Fechada Não Linear')
    ax_sim.plot(t_l, y_l, 'b-', lw=1.5, label='Malha Fechada Linearizada')

    ax_sim.set_xlabel('Tempo [s]')
    ax_sim.set_ylabel('Desvio de Nível $\Delta h_2$ [m]')
    ax_sim.set_title(f'Resposta ao Degrau de Referência de +{amp_percent}%')
    ax_sim.grid(True, alpha=0.3)
    ax_sim.legend()
    st.pyplot(fig_sim)

    st.markdown("""
    ### 💡 Estabilidade Global e Região de Validade da Linearização
    Observe que para pequenos degraus (ex: 5% a 10%), as curvas linear e não linear quase se sobrepõem. Isso valida localmente o **Teorema de Hartman-Grobman**. 

    Contudo, à medida que você aumenta a amplitude do degrau na barra lateral para valores elevados (ex: 50% ou 100%), as duas curvas começam a divergir significativamente. 
    Isso ocorre porque o termo de raiz quadrada ($\sqrt{h}$) perde sua característica linear local à medida que nos afastamos do ponto de operação, provando que a análise de estabilidade por Jacobiano é estritamente **local**.
    """)

# --- TAB 4: VISUALIZAÇÃO FÍSICA DOS TANQUES (NOVA ABA) ---
with tab4:
    st.subheader("💧 Nível Absoluto Estacionário dos Tanques")
    st.markdown("""
    Este gráfico ilustra a altura real acumulada da água dentro de cada tanque ($\bar{h} + \Delta h$) ao final da 
    simulação não linear sob controle malha fechada.
    """)

    # Extração dos estados estáveis finais (fim do vetor de integração)
    nivel_final_h1 = sol_cl.y[0][-1]
    nivel_final_h2 = sol_cl.y[1][-1]

    col_graph, col_metrics = st.columns([1, 1])

    with col_graph:
        fig_tanques, ax_t = plt.subplots(figsize=(5, 5.5))
        bars = ax_t.bar(
            ["Tanque 1 (Superior)", "Tanque 2 (Inferior)"],
            [nivel_final_h1, nivel_final_h2],
            color='#1f77b4', edgecolor='black', width=0.5
        )
        ax_t.set_ylabel("Altura Absoluta da Água [metros]")
        ax_t.set_ylim(0, max(h1_bar * 1.5, 2.0))
        ax_t.grid(axis='y', linestyle='--', alpha=0.5)

        # Adiciona os valores numéricos no topo de cada barra
        for bar in bars:
            height = bar.get_height()
            ax_t.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.4f} m",
                ha='center', va='bottom', fontweight='bold', color='black'
            )
        st.pyplot(fig_tanques)

    with col_metrics:
        st.markdown("### Comparação com os Níveis Nominais de Equilíbrio")

        st.metric(
            label="Nível Final do Tanque 1 ($h_1$)",
            value=f"{nivel_final_h1:.4f} m",
            delta=f"{nivel_final_h1 - h1_bar:.4f} m (vs. equilíbrio inicial)"
        )

        st.metric(
            label="Nível Final do Tanque 2 ($h_2$)",
            value=f"{nivel_final_h2:.4f} m",
            delta=f"{nivel_final_h2 - h2_bar:.4f} m (vs. equilíbrio inicial)"
        )

        st.info(f"""
        **Configuração do Ponto de Operação Original ($\overline{u}$):**
        * Altura de Equilíbrio $\overline{{h}}_1$: `{h1_bar:.4f} m`
        * Altura de Equilíbrio $\overline{{h}}_2$: `{h2_bar:.4f} m`

        **Análise Física do Controle:**
        Como a referência solicitou uma elevação no nível do segundo tanque, o controlador PD detecta o erro transiente e abre a vazão de entrada $u(t)$ para além do valor nominal $\overline{{u}}$. Para elevar o Tanque 2, o Tanque 1 obrigatoriamente precisa acumular mais carga hidráulica primeiro para vencer a restrição do orifício de acoplamento, explicando o aumento concomitante em ambos os níveis.
        """)

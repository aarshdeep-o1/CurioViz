import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, t
import math

# Must be first
st.set_page_config(page_title="CurioViz - Distribution Visualizer", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=IBM+Plex+Mono:wght@300;400;600&display=swap');

/* Main Background */
.stApp {
    background-color: #050A14;
    color: #e0eaff;
    font-family: 'IBM Plex Mono', monospace;
}

/* Hide Default Elements */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Headings */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    color: #00F5FF !important;
}
.stMarkdown p {
    font-family: 'IBM Plex Mono', monospace;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(5, 10, 20, 0.9) !important;
    border-right: 1px solid rgba(0, 245, 255, 0.15);
}

/* Custom Cards */
.topic-card {
    background: #0b1223;
    border: 1px solid rgba(0, 245, 255, 0.15);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Data
topics = [
    {
        'id': 'normal', 'icon': '📊', 'badge': 'Distributions', 'title': 'Normal Distribution',
        'short': 'Symmetrical bell-shaped curve where most occurrences take place in the middle.',
        'definition': 'The Normal Distribution (also called Gaussian distribution) is a continuous probability distribution that is symmetric around its mean. It describes how values of a variable are distributed — most observations cluster around the central peak, and probabilities for values further away from the mean taper off equally in both directions.',
        'formula': 'f(x) = (1 / \\sigma\\sqrt{2\\pi}) \\cdot e^{-(x-\\mu)^2 / 2\\sigma^2}',
        'formulaNote': 'Where μ = mean, σ = standard deviation, e = Euler\'s number (~2.718)',
        'properties': [
            {'label': 'Symmetry', 'desc': 'Perfectly symmetric around the mean μ'},
            {'label': 'Mean = Median = Mode', 'desc': 'All three measures of center are equal'},
            {'label': '68-95-99.7 Rule', 'desc': '68% data within ±1σ, 95% within ±2σ, 99.7% within ±3σ'},
            {'label': 'Total Area = 1', 'desc': 'The area under the curve equals 1 (all probabilities)'},
        ],
        'example': 'Heights of adult humans, IQ scores, exam marks in a large class, and measurement errors in scientific instruments all follow approximately normal distributions.',
        'uses': ['Hypothesis testing', 'Confidence intervals', 'Machine learning assumptions', 'Natural phenomena modeling']
    },
    {
        'id': 'skewness', 'icon': '↔️', 'badge': 'Shape Measure', 'title': 'Skewness',
        'short': 'Measure of the asymmetry of the probability distribution of a real-valued random variable.',
        'definition': 'Skewness measures the asymmetry of a distribution around its mean. A skewness of 0 means the distribution is perfectly symmetric. Positive skewness means the tail is on the right (more extreme high values), while negative skewness means the tail is on the left (more extreme low values).',
        'formula': 'Skewness = E[(X - \\mu)^3] / \\sigma^3',
        'formulaNote': 'Pearson\'s moment coefficient of skewness — a dimensionless value',
        'properties': [
            {'label': 'Skew = 0', 'desc': 'Perfectly symmetric (Normal Distribution)'},
            {'label': 'Skew > 0', 'desc': 'Right-skewed (positive): long right tail, mean > median'},
            {'label': 'Skew < 0', 'desc': 'Left-skewed (negative): long left tail, mean < median'},
        ],
        'example': 'Income distribution is typically right-skewed.',
        'uses': ['Detecting outlier influence', 'Choosing appropriate statistical tests']
    },
    {
        'id': 'kurtosis', 'icon': '📐', 'badge': 'Tail Shape', 'title': 'Kurtosis',
        'short': 'Measure of the "tailedness" of the probability distribution, identifying outliers.',
        'definition': 'Kurtosis measures how heavily the tails of a distribution differ from those of a normal distribution. High kurtosis means more data in the tails. Excess kurtosis = kurtosis − 3.',
        'formula': 'Kurt(X) = E[(X - \\mu)^4] / \\sigma^4',
        'formulaNote': 'A normal distribution has kurtosis = 3, so excess kurtosis = 0',
        'properties': [
            {'label': 'Mesokurtic (K=3)', 'desc': 'Normal distribution — baseline reference'},
            {'label': 'Leptokurtic (K>3)', 'desc': 'Heavy tails, sharp peak — more outliers than normal'},
            {'label': 'Platykurtic (K<3)', 'desc': 'Light tails, flat peak — fewer outliers than normal'},
        ],
        'example': 'Stock market returns are leptokurtic.',
        'uses': ['Financial risk modeling', 'Outlier detection']
    },
    {
        'id': 'histogram', 'icon': '🧱', 'badge': 'Visualization', 'title': 'Histogram',
        'short': 'Approximate representation of the distribution of numerical data using bars.',
        'definition': 'A histogram is a graphical representation of the distribution of numerical data. It groups data into continuous intervals (bins).',
        'formula': 'Bin\\ Width = (Max - Min) / Number\\ of\\ Bins',
        'formulaNote': 'Sturges\' Rule for bins: k = ceil(log2(n) + 1)',
        'properties': [
            {'label': 'X-axis', 'desc': 'Continuous variable range divided into equal-width bins'},
            {'label': 'Y-axis', 'desc': 'Frequency or relative frequency'},
        ],
        'example': 'A teacher recording student scores.',
        'uses': ['Exploring data distribution shape', 'Detecting outliers and gaps']
    },
    {
        'id': 'bellcurve', 'icon': '🔔', 'badge': 'Normal Curve', 'title': 'Bell Curve',
        'short': 'The classic shape of the normal distribution representing natural phenomena.',
        'definition': 'The Bell Curve is the visual representation of the Normal Distribution. The Empirical Rule describes how data spreads across standard deviations.',
        'formula': 'P(\\mu-\\sigma < X < \\mu+\\sigma) \\approx 68.27\\%',
        'formulaNote': 'The Empirical Rule',
        'properties': [
            {'label': '68% Rule', 'desc': '68.27% of data lies within 1 standard deviation of mean'},
            {'label': '95% Rule', 'desc': '95.45% of data lies within 2 standard deviations of mean'},
        ],
        'example': 'Test scores in a large population.',
        'uses': ['Grading on a curve', 'Quality control (6-sigma)']
    }
]

questions = [
    {'topic':'Normal Distribution', 'q':'What shape does a normal distribution form when plotted?', 'opts':['Uniform rectangle','Symmetric bell curve','Right-skewed triangle','Bimodal twin peaks'], 'ans':1, 'exp':'A normal distribution always forms a symmetric bell-shaped curve.'},
    {'topic':'Normal Distribution', 'q':'In a normal distribution, which statement is always true?', 'opts':['Mean > Median','Mean = Median = Mode','Mode > Mean','Median < Mode'], 'ans':1, 'exp':'Mean, median, and mode are all equal.'},
    {'topic':'Skewness', 'q':'If a distribution has a long tail on the right side, it is:', 'opts':['Negatively skewed','Normally distributed','Positively skewed','Platykurtic'], 'ans':2, 'exp':'Long right tail = positive (right) skewness.'},
    {'topic':'Skewness', 'q':'For a perfectly symmetric distribution, skewness equals:', 'opts':['1','−1','0','3'], 'ans':2, 'exp':'Zero skewness means perfect symmetry.'},
    {'topic':'Kurtosis', 'q':'A leptokurtic distribution compared to a normal distribution has:', 'opts':['Lighter tails and flat peak','Heavier tails and sharper peak','Symmetric tails','Zero variance'], 'ans':1, 'exp':'Leptokurtic (K > 3) have heavier tails and a sharper peak.'},
    {'topic':'Kurtosis', 'q':'What is the kurtosis value of a standard normal distribution?', 'opts':['0','1','2','3'], 'ans':3, 'exp':'Normal distribution has kurtosis = 3.'},
    {'topic':'Histogram', 'q':'In a histogram, what does each bar\'s HEIGHT represent?', 'opts':['The exact data value','Frequency or count of data in that bin','The mean of that bin','Standard deviation'], 'ans':1, 'exp':'Height represents frequency (count) of data in that bin.'},
    {'topic':'Histogram', 'q':'Unlike a bar chart, histogram bars have:', 'opts':['Color gradients','No gaps between them','Alphabetical labels','Different widths always'], 'ans':1, 'exp':'Histogram bars touch each other.'},
    {'topic':'Bell Curve', 'q':'According to the Empirical Rule, approximately what % of data falls within 2 standard deviations of the mean?', 'opts':['68%','95%','99.7%','50%'], 'ans':1, 'exp':'The 68-95-99.7 rule: ~95% of data lies within ±2σ.'},
    {'topic':'Bell Curve', 'q':'At what points does a bell curve change from concave to convex?', 'opts':['At the mean','At the median','At exactly ±1 standard deviation','At the maximum'], 'ans':2, 'exp':'The inflection points occur exactly at μ±σ.'},
]

# State
if 'learned_topics' not in st.session_state:
    st.session_state.learned_topics = set()
if 'q_idx' not in st.session_state:
    st.session_state.q_idx = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = ""
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = None

# Plot configs
layout_config = dict(
    paper_bgcolor='#050A14',
    plot_bgcolor='#050A14',
    font=dict(color='#e0eaff', family='IBM Plex Mono'),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='rgba(0,245,255,0.1)', zeroline=False)
)

def plot_normal(mu, sigma):
    x = np.linspace(-6, 6, 500)
    y = norm.pdf(x, mu, sigma)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor='rgba(0, 245, 255, 0.15)', line=dict(color='#00F5FF', width=2)))
    fig.add_vline(x=mu, line_dash="dash", line_color="#9B5DE5", annotation_text=f"μ={mu}")
    fig.update_layout(**layout_config, title="Normal Distribution")
    return fig

def plot_skewness(skew):
    x = np.linspace(-4, 8, 500)
    alpha = skew * 3
    phi = norm.pdf(x)
    Phi = norm.cdf(alpha * x)
    y = 2 * phi * Phi
    color = '#9B5DE5' if skew > 0 else ('#00F5FF' if skew < 0 else '#e0eaff')
    fillcolor = color.replace(')', ', 0.15)').replace('rgb', 'rgba') if 'rgb' in color else color
    if color == '#9B5DE5': fillcolor = 'rgba(155, 93, 229, 0.15)'
    elif color == '#00F5FF': fillcolor = 'rgba(0, 245, 255, 0.15)'
    else: fillcolor = 'rgba(224, 234, 255, 0.15)'
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor=fillcolor, line=dict(color=color, width=2)))
    fig.update_layout(**layout_config, title=f"Skewness = {skew}")
    return fig

def plot_kurtosis(ex_kurt):
    x = np.linspace(-4, 4, 500)
    y_norm = norm.pdf(x)
    df = max(2.5, 10 - ex_kurt * 2) if ex_kurt > 0 else 100
    y_kurt = t.pdf(x, df) if ex_kurt > 0 else norm.pdf(x, scale=1 + abs(ex_kurt)*0.2)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_norm, line=dict(color='rgba(255,255,255,0.3)', dash='dash'), name='Normal'))
    color = '#ff6b6b' if ex_kurt > 0.5 else ('#9B5DE5' if ex_kurt < -0.5 else '#00F5FF')
    fig.add_trace(go.Scatter(x=x, y=y_kurt, fill='tozeroy', fillcolor='rgba(0, 245, 255, 0.1)', line=dict(color=color, width=2), name='Custom Kurtosis'))
    fig.update_layout(**layout_config, title=f"Excess Kurtosis = {ex_kurt}")
    return fig

def plot_histogram(bins, spread):
    data = np.random.normal(0, spread, 1000)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=bins, marker_color='#9B5DE5', opacity=0.7))
    fig.update_layout(**layout_config, title="Histogram")
    return fig

def plot_bellcurve(mu, sigma):
    x = np.linspace(-5, 5, 500)
    y = norm.pdf(x, mu, sigma)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='#00F5FF', width=2), showlegend=False))
    
    x_68 = np.linspace(mu-sigma, mu+sigma, 100)
    y_68 = norm.pdf(x_68, mu, sigma)
    fig.add_trace(go.Scatter(x=x_68, y=y_68, fill='tozeroy', fillcolor='rgba(0, 245, 255, 0.2)', mode='none', name='68% (1σ)'))
    
    x_95 = np.linspace(mu-2*sigma, mu+2*sigma, 100)
    y_95 = norm.pdf(x_95, mu, sigma)
    fig.add_trace(go.Scatter(x=x_95, y=y_95, fill='tozeroy', fillcolor='rgba(155, 93, 229, 0.1)', mode='none', name='95% (2σ)'))
    
    fig.update_layout(**layout_config, title="Bell Curve (Empirical Rule)")
    return fig

# Sidebar
st.sidebar.markdown("<h2 style='text-align: center; color: #00F5FF;'>CurioViz</h2>", unsafe_allow_html=True)
nav = st.sidebar.radio("Navigation", ["LEARN", "ABOUT", "ASSESS"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Progress:** {len(st.session_state.learned_topics)} / 5 Topics Learned")
for t_obj in topics:
    mark = "✅" if t_obj['id'] in st.session_state.learned_topics else "❌"
    st.sidebar.markdown(f"{mark} {t_obj['title']}")

if nav == "LEARN":
    if st.session_state.current_topic is None:
        st.markdown("<h1 style='text-align: center;'>Understand Data, <span style='color: #9B5DE5;'>Visually.</span></h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6a7fa8;'>Explore normal distributions, skewness, kurtosis, histograms and bell curves.</p>", unsafe_allow_html=True)
        
        cols = st.columns(3)
        for i, t_obj in enumerate(topics):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="topic-card">
                    <h3 style="margin-top: 0;">{t_obj['icon']} {t_obj['title']}</h3>
                    <p style="font-size: 0.85rem; color: #6a7fa8;">{t_obj['short']}</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Explore {t_obj['title']}", key=f"btn_{t_obj['id']}", use_container_width=True):
                    st.session_state.current_topic = t_obj['id']
                    st.rerun()
    else:
        if st.button("← Back to Topics"):
            st.session_state.current_topic = None
            st.rerun()
            
        t_obj = next(t for t in topics if t['id'] == st.session_state.current_topic)
        st.title(f"{t_obj['icon']} {t_obj['title']}")
        
        st.subheader("Definition")
        st.write(t_obj['definition'])
        
        st.subheader("Formula")
        st.latex(t_obj['formula'])
        st.caption(t_obj['formulaNote'])
        
        st.subheader("Interactive Simulation")
        if t_obj['id'] == 'normal':
            c1, c2 = st.columns(2)
            mu = c1.slider("Mean (μ)", -3.0, 3.0, 0.0, 0.1)
            sigma = c2.slider("Std Dev (σ)", 0.3, 3.0, 1.0, 0.1)
            st.plotly_chart(plot_normal(mu, sigma), use_container_width=True)
        elif t_obj['id'] == 'skewness':
            skew = st.slider("Skewness", -3.0, 3.0, 0.0, 0.1)
            st.plotly_chart(plot_skewness(skew), use_container_width=True)
        elif t_obj['id'] == 'kurtosis':
            ex_kurt = st.slider("Excess Kurtosis", -2.0, 5.0, 0.0, 0.1)
            st.plotly_chart(plot_kurtosis(ex_kurt), use_container_width=True)
        elif t_obj['id'] == 'histogram':
            c1, c2 = st.columns(2)
            bins = c1.slider("Bins", 4, 25, 10, 1)
            spread = c2.slider("Spread (σ)", 0.3, 2.5, 1.0, 0.1)
            st.plotly_chart(plot_histogram(bins, spread), use_container_width=True)
        elif t_obj['id'] == 'bellcurve':
            c1, c2 = st.columns(2)
            mu = c1.slider("Mean (μ)", -2.0, 2.0, 0.0, 0.1)
            sigma = c2.slider("Std Dev (σ)", 0.3, 2.5, 1.0, 0.1)
            st.plotly_chart(plot_bellcurve(mu, sigma), use_container_width=True)
            
        st.subheader("Key Properties")
        for p in t_obj['properties']:
            st.markdown(f"- **{p['label']}**: {p['desc']}")
            
        if t_obj['id'] not in st.session_state.learned_topics:
            if st.button("Mark as Learned"):
                st.session_state.learned_topics.add(t_obj['id'])
                st.rerun()
        else:
            st.success("✓ Marked as Learned")

elif nav == "ABOUT":
    st.title("About CurioViz")
    st.write("An interactive educational web app designed to make complex statistical concepts intuitive and visually engaging. Built for data science students, analysts, and curious minds who want to truly understand data distributions — not just memorize formulas.")
    st.write("### Tech Stack")
    st.write("- Python, Streamlit, Plotly, NumPy, SciPy")
    st.write("### Creator")
    st.write("Aarshdeep")

elif nav == "ASSESS":
    st.title("Final Assessment")
    
    if len(st.session_state.learned_topics) < 5:
        st.warning(f"🔒 You must mark all 5 topics as learned in the **LEARN** tab before taking the assessment. Currently: {len(st.session_state.learned_topics)}/5")
    else:
        st.success("Modules Complete! You are now officially ready to take the final examination.")
        
        if st.session_state.q_idx >= len(questions):
            pct = round(st.session_state.score / len(questions) * 100)
            st.header(f"Final Score: {pct}%")
            if pct >= 90:
                st.success("Outstanding! You're a distribution wizard. 🎯")
            elif pct >= 75:
                st.info("Great job! Strong understanding of all concepts. 🔥")
            else:
                st.warning("Keep learning! Head back to the Learn tab. 💪")
                
            if st.button("Retry Quiz"):
                st.session_state.q_idx = 0
                st.session_state.score = 0
                st.session_state.quiz_submitted = False
                st.rerun()
        else:
            q = questions[st.session_state.q_idx]
            st.progress((st.session_state.q_idx) / len(questions))
            st.write(f"**Question {st.session_state.q_idx + 1} of {len(questions)}** ({q['topic']})")
            st.subheader(q['q'])
            
            choice = st.radio("Select an answer:", q['opts'], key=f"radio_{st.session_state.q_idx}")
            
            if not st.session_state.quiz_submitted:
                if st.button("Submit Answer"):
                    st.session_state.quiz_submitted = True
                    selected_idx = q['opts'].index(choice)
                    if selected_idx == q['ans']:
                        st.session_state.score += 1
                        st.session_state.last_result = "Correct"
                    else:
                        st.session_state.last_result = "Incorrect"
                    st.rerun()
            else:
                if st.session_state.last_result == "Correct":
                    st.success(f"✓ Correct! {q['exp']}")
                else:
                    st.error(f"✗ Not quite. {q['exp']}")
                    
                if st.button("Next Question →"):
                    st.session_state.q_idx += 1
                    st.session_state.quiz_submitted = False
                    st.rerun()

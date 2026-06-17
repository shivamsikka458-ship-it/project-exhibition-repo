import streamlit as st
import cv2
import numpy as np
import os
import time
from collections import deque

# ── MediaPipe: compatible with both old and new API ──
try:
    # mediapipe >= 0.10 new API
    from mediapipe.python.solutions import hands as _mp_hands_mod
    from mediapipe.python.solutions import drawing_utils as mp_draw
    from mediapipe.python.solutions import drawing_styles as mp_style
    mp_hands = _mp_hands_mod
    _MP_NEW_API = True
except ImportError:
    # fallback old API (< 0.10)
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles
    _MP_NEW_API = False

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SignSense AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  — dark neon-cyberpunk theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;700&display=swap');

:root {
    --bg:        #050a14;
    --surface:   #0b1628;
    --border:    #1a3a6e;
    --neon:      #00f0ff;
    --neon2:     #7b2fff;
    --accent:    #ff3d7f;
    --green:     #00ff9d;
    --text:      #c8deff;
    --muted:     #4a6fa5;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Exo 2', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--neon) !important;
    font-size: 2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--neon) !important;
    color: var(--neon) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 10px 28px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--neon) !important;
    color: var(--bg) !important;
    box-shadow: 0 0 20px var(--neon) !important;
}

/* ── Select / Input / Slider ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}
.stSlider > div > div > div > div { background: var(--neon) !important; }

/* ── Progress bar ── */
.stProgress > div > div > div { background: var(--neon) !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { border-bottom: 1px solid var(--border) !important; }
[data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    color: var(--muted) !important;
}
[aria-selected="true"] { color: var(--neon) !important; border-bottom: 2px solid var(--neon) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Custom card ── */
.hud-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}
.hud-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--neon), var(--neon2));
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 40px 0 20px;
}
.hero h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 3.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 4px !important;
    background: linear-gradient(135deg, var(--neon), var(--neon2), var(--accent));
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    line-height: 1.1 !important;
    margin: 0 !important;
}
.hero p {
    color: var(--muted) !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 2px !important;
    font-size: 0.85rem !important;
    margin-top: 8px !important;
}

/* ── Prediction badge ── */
.pred-badge {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: var(--green);
    text-shadow: 0 0 24px var(--green);
    letter-spacing: 4px;
    text-align: center;
    padding: 20px;
    border: 1px solid var(--green);
    border-radius: 12px;
    background: rgba(0,255,157,0.05);
    margin: 10px 0;
}

/* ── Status dot ── */
.dot-live  { display:inline-block; width:10px; height:10px; border-radius:50%; background:var(--accent); box-shadow:0 0 10px var(--accent); animation:pulse 1s infinite; }
.dot-ready { display:inline-block; width:10px; height:10px; border-radius:50%; background:var(--green); box-shadow:0 0 10px var(--green); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Sentence display ── */
.sentence-box {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
    color: var(--neon);
    background: rgba(0,240,255,0.04);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 20px;
    min-height: 48px;
    letter-spacing: 1px;
    word-break: break-all;
}

/* ── Confidence bar ── */
.conf-row { display:flex; align-items:center; gap:10px; margin:4px 0; }
.conf-label { font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:var(--muted); width:80px; }
.conf-track { flex:1; height:8px; background:var(--surface); border-radius:4px; overflow:hidden; border:1px solid var(--border); }
.conf-fill  { height:100%; border-radius:4px; background:linear-gradient(90deg,var(--neon2),var(--neon)); transition:width 0.3s; }
.conf-pct   { font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:var(--neon); width:42px; text-align:right; }

/* ── Section label ── */
.sec-label {
    font-family:'Rajdhani',sans-serif;
    font-size:0.75rem;
    letter-spacing:3px;
    text-transform:uppercase;
    color:var(--muted);
    margin-bottom:8px;
}

/* Alert boxes */
.alert-ok   { background:rgba(0,255,157,0.08); border:1px solid var(--green); border-radius:8px; padding:12px 16px; color:var(--green); font-family:'Share Tech Mono',monospace; font-size:0.85rem; }
.alert-warn { background:rgba(255,61,127,0.08); border:1px solid var(--accent); border-radius:8px; padding:12px 16px; color:var(--accent); font-family:'Share Tech Mono',monospace; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MEDIAPIPE  setup
# ─────────────────────────────────────────────
# MediaPipe already imported above (new/old API compatible)

DATA_PATH = "sign_language_dataset/data"
MODEL_PATH = "action_model.h5"
FRAMES_PER_SAMPLE = 30

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        kp = []
        for lm in hand.landmark:
            kp.extend([lm.x, lm.y, lm.z])
        return np.array(kp)
    return np.zeros(63)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained Keras model (cached across reruns)."""
    try:
        from tensorflow.keras.models import load_model as km
        model = km(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)


def get_actions():
    """Scan dataset folder and return sorted unique action labels."""
    if not os.path.exists(DATA_PATH):
        return []
    actions = set()
    for f in os.listdir(DATA_PATH):
        if f.endswith(".npy"):
            actions.add(f.split("_")[0])
    return sorted(list(actions))


def draw_hand_landmarks(frame, results):
    """Draw stylised landmarks on frame (compatible with mediapipe 0.10.x)."""
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 240, 255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(123, 47, 255), thickness=2)
            )
    return frame


def make_hands_detector():
    """Create a MediaPipe Hands detector — works with both old and new API."""
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )


def overlay_hud(frame, text, conf=None, color=(0, 240, 255)):
    """Burn prediction text + confidence onto frame."""
    h, w = frame.shape[:2]
    # semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 54), (5, 10, 20), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.putText(frame, text, (14, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)
    if conf is not None:
        bar_w = int(conf * (w - 20))
        cv2.rectangle(frame, (10, h - 10), (w - 10, h - 4), (26, 45, 80), -1)
        cv2.rectangle(frame, (10, h - 10), (10 + bar_w, h - 4), color, -1)
    return frame

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🤟 SIGNSENSE AI</h1>
  <p>// REAL-TIME SIGN LANGUAGE RECOGNITION SYSTEM //</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec-label">System Status</div>', unsafe_allow_html=True)

    model, model_err = load_model()
    if model:
        st.markdown('<div class="alert-ok">✓ Model Loaded — action_model.h5</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-warn">✗ Model not found<br><small>{model_err}</small></div>', unsafe_allow_html=True)

    st.markdown("---")
    actions = get_actions()
    st.markdown('<div class="sec-label">Detected Labels</div>', unsafe_allow_html=True)
    if actions:
        st.markdown(f'<div class="alert-ok">✓ {len(actions)} classes: {", ".join(actions)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-warn">⚠ No dataset found at<br>sign_language_dataset/data/</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sec-label">Configuration</div>', unsafe_allow_html=True)
    threshold     = st.slider("Confidence Threshold", 0.1, 1.0, 0.6, 0.05)
    sentence_len  = st.slider("Max Sentence Words",    1,  20,   8)
    cam_index     = st.number_input("Camera Index", 0, 5, 0, 1)

    st.markdown("---")
    st.markdown('<div class="sec-label">Data Collection Settings</div>', unsafe_allow_html=True)
    col_word    = st.text_input("Sign Word", "hello")
    col_samples = st.number_input("Samples", 5, 100, 30)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Live Recognition",
    "📹  Collect Data",
    "🏋️  Train Model",
    "📊  Dataset Info",
])

# ══════════════════════════════════════════════
# TAB 1 — LIVE RECOGNITION
# ══════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="hud-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Live Camera Feed</div>', unsafe_allow_html=True)
        cam_placeholder = st.empty()
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            start_recog = st.button("▶  START RECOGNITION", key="start_r")
        with btn_c2:
            stop_recog  = st.button("⏹  STOP", key="stop_r")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="hud-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Prediction</div>', unsafe_allow_html=True)
        pred_placeholder = st.empty()

        st.markdown('<div class="sec-label" style="margin-top:16px">Confidence Scores</div>', unsafe_allow_html=True)
        conf_placeholder = st.empty()

        st.markdown('<div class="sec-label" style="margin-top:16px">Sentence</div>', unsafe_allow_html=True)
        sent_placeholder = st.empty()

        btn_clear = st.button("🗑  CLEAR SENTENCE")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Session state ──
    if "sentence"    not in st.session_state: st.session_state.sentence    = []
    if "running_r"   not in st.session_state: st.session_state.running_r   = False
    if "last_word"   not in st.session_state: st.session_state.last_word   = ""

    if btn_clear:
        st.session_state.sentence  = []
        st.session_state.last_word = ""

    if start_recog: st.session_state.running_r = True
    if stop_recog:  st.session_state.running_r = False

    # default idle display
    pred_placeholder.markdown('<div class="pred-badge">—</div>', unsafe_allow_html=True)
    sent_placeholder.markdown(
        f'<div class="sentence-box">{" ".join(st.session_state.sentence)}&nbsp;</div>',
        unsafe_allow_html=True
    )

    if st.session_state.running_r:
        if model is None:
            st.error("❌ Cannot start: model not loaded. Train the model first.")
            st.session_state.running_r = False
        elif not actions:
            st.error("❌ Cannot start: no dataset labels found.")
            st.session_state.running_r = False
        else:
            label_map = {i: a for i, a in enumerate(actions)}
            sequence  = deque(maxlen=FRAMES_PER_SAMPLE)

            cap = cv2.VideoCapture(int(cam_index))
            hands_det = make_hands_detector()

            while st.session_state.running_r:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠ Camera not accessible.")
                    break

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = hands_det.process(rgb)

                frame = draw_hand_landmarks(frame, res)
                kp    = extract_keypoints(res)
                sequence.append(kp)

                prediction_text = "Collecting frames…"
                conf_scores     = None
                detected_word   = None

                if len(sequence) == FRAMES_PER_SAMPLE:
                    seq_arr = np.expand_dims(np.array(sequence), axis=0)
                    probs   = model.predict(seq_arr, verbose=0)[0]
                    idx     = int(np.argmax(probs))
                    conf    = float(probs[idx])
                    word    = label_map.get(idx, "?")
                    conf_scores = probs

                    if conf >= threshold:
                        prediction_text = f"{word.upper()}  [{conf*100:.0f}%]"
                        detected_word   = word
                        frame = overlay_hud(frame, prediction_text, conf, (0, 255, 157))

                        # append to sentence (no repeat)
                        if word != st.session_state.last_word:
                            st.session_state.sentence.append(word)
                            if len(st.session_state.sentence) > sentence_len:
                                st.session_state.sentence.pop(0)
                            st.session_state.last_word = word
                    else:
                        prediction_text = f"{word}  [{conf*100:.0f}%] — low conf"
                        frame = overlay_hud(frame, prediction_text, conf, (255, 61, 127))

                # ── Update UI ──
                cam_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_container_width=True
                )

                if detected_word:
                    pred_placeholder.markdown(
                        f'<div class="pred-badge">{detected_word.upper()}</div>',
                        unsafe_allow_html=True
                    )

                if conf_scores is not None:
                    bars_html = ""
                    for i, score in enumerate(conf_scores):
                        lbl = label_map.get(i, str(i))
                        pct = int(score * 100)
                        bars_html += f"""
                        <div class="conf-row">
                          <span class="conf-label">{lbl[:8]}</span>
                          <div class="conf-track"><div class="conf-fill" style="width:{pct}%"></div></div>
                          <span class="conf-pct">{pct}%</span>
                        </div>"""
                    conf_placeholder.markdown(bars_html, unsafe_allow_html=True)

                sent_placeholder.markdown(
                    f'<div class="sentence-box">{" ".join(st.session_state.sentence)}&nbsp;</div>',
                    unsafe_allow_html=True
                )

            cap.release()
            hands_det.close()

# ══════════════════════════════════════════════
# TAB 2 — COLLECT DATA
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="hud-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Data Collection</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        feed_ph   = st.empty()
        status_ph = st.empty()
        prog_ph   = st.empty()

    with col_b:
        st.markdown(f"""
        <div style="margin-bottom:20px">
          <div class="sec-label">Target Sign</div>
          <div style="font-family:'Rajdhani',sans-serif;font-size:2rem;color:var(--neon);letter-spacing:3px;">{col_word.upper()}</div>
        </div>
        <div>
          <div class="sec-label">Samples to Collect</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:1.4rem;color:var(--text);">{col_samples} × {FRAMES_PER_SAMPLE} frames</div>
        </div>
        """, unsafe_allow_html=True)

        info_ph = st.empty()

    collect_btn = st.button("▶  START COLLECTION", key="collect")
    st.markdown('</div>', unsafe_allow_html=True)

    if collect_btn:
        save_path = DATA_PATH
        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(int(cam_index))
        hands_det = make_hands_detector()

        for sample_idx in range(int(col_samples)):
            # countdown
            status_ph.markdown(
                f'<div class="alert-warn">⏳ Get ready — Sample {sample_idx+1}/{int(col_samples)}</div>',
                unsafe_allow_html=True
            )

            for cd in range(3, 0, -1):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    frame = overlay_hud(frame, f"Get ready — starting in {cd}…", color=(255, 180, 0))
                    feed_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                  channels="RGB", use_container_width=True)
                time.sleep(1)

            # record frames
            sequence = []
            status_ph.markdown(
                f'<div class="alert-ok">🔴 Recording — Sample {sample_idx+1}/{int(col_samples)}</div>',
                unsafe_allow_html=True
            )

            for fn in range(FRAMES_PER_SAMPLE):
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = hands_det.process(rgb)
                frame = draw_hand_landmarks(frame, res)
                kp    = extract_keypoints(res)
                sequence.append(kp)

                frame = overlay_hud(
                    frame,
                    f"{col_word.upper()}  | frame {fn+1}/{FRAMES_PER_SAMPLE}",
                    fn / FRAMES_PER_SAMPLE,
                    (0, 255, 157)
                )
                feed_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                              channels="RGB", use_container_width=True)
                prog_ph.progress((sample_idx * FRAMES_PER_SAMPLE + fn + 1) /
                                 (int(col_samples) * FRAMES_PER_SAMPLE))

            fname = f"{col_word}_{sample_idx+1}.npy"
            np.save(os.path.join(save_path, fname), np.array(sequence))
            info_ph.markdown(
                f'<div class="alert-ok">✓ Saved {fname}</div>', unsafe_allow_html=True
            )

        cap.release()
        hands_det.close()
        status_ph.markdown(
            f'<div class="alert-ok">✅ Collection complete! {int(col_samples)} samples saved for "{col_word}"</div>',
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════
# TAB 3 — TRAIN MODEL
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="hud-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Model Training — CNN + BiLSTM Architecture</div>', unsafe_allow_html=True)

    arch_col, param_col = st.columns(2, gap="large")

    with arch_col:
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;color:var(--muted);line-height:2">
        INPUT (30 frames × 63 keypoints)<br>
        &nbsp;&nbsp;↓<br>
        Conv1D(64) → BatchNorm → MaxPool<br>
        &nbsp;&nbsp;↓<br>
        BiLSTM(64, return_seq=True)<br>
        &nbsp;&nbsp;↓<br>
        BiLSTM(128)<br>
        &nbsp;&nbsp;↓<br>
        Dense(64) → Dropout(0.3)<br>
        &nbsp;&nbsp;↓<br>
        Dense(N_classes, softmax)
        </div>
        """, unsafe_allow_html=True)

    with param_col:
        epochs     = st.slider("Epochs",      10, 300, 200, 10)
        batch_size = st.slider("Batch Size",   8, 128,  32,  8)
        test_split = st.slider("Test Split",  0.05, 0.3, 0.1, 0.05)
        patience   = st.slider("Early Stop Patience", 5, 50, 20, 5)

    train_btn = st.button("🏋️  START TRAINING", key="train_btn")
    train_out = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    if train_btn:
        if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
            train_out.markdown('<div class="alert-warn">⚠ No data found. Collect data first.</div>', unsafe_allow_html=True)
        else:
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import (
                    LSTM, Dense, Bidirectional, Dropout,
                    Conv1D, MaxPooling1D, BatchNormalization
                )
                from tensorflow.keras.utils import to_categorical
                from tensorflow.keras.callbacks import EarlyStopping
                from sklearn.model_selection import train_test_split

                train_out.markdown('<div class="alert-ok">⚙ Loading dataset…</div>', unsafe_allow_html=True)

                X, y = [], []
                act_set = set()
                for f in os.listdir(DATA_PATH):
                    if f.endswith(".npy"):
                        act_set.add(f.split("_")[0])
                act_list  = sorted(list(act_set))
                lmap      = {a: i for i, a in enumerate(act_list)}

                for f in os.listdir(DATA_PATH):
                    if f.endswith(".npy"):
                        seq  = np.load(os.path.join(DATA_PATH, f))
                        word = f.split("_")[0]
                        X.append(seq)
                        y.append(lmap[word])

                X = np.array(X)
                y = to_categorical(y).astype(int)

                idx = np.arange(len(X))
                np.random.shuffle(idx)
                X, y = X[idx], y[idx]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_split))

                train_out.markdown(
                    f'<div class="alert-ok">✓ Dataset: {X.shape[0]} samples, {len(act_list)} classes<br>'
                    f'Train: {len(X_train)} | Test: {len(X_test)}</div>',
                    unsafe_allow_html=True
                )

                model_t = Sequential([
                    Conv1D(64, kernel_size=3, activation='relu', input_shape=(30, 63)),
                    BatchNormalization(),
                    MaxPooling1D(pool_size=2),
                    Bidirectional(LSTM(64, return_sequences=True, activation='relu')),
                    Bidirectional(LSTM(128, return_sequences=False, activation='relu')),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(len(act_list), activation='softmax'),
                ])
                model_t.compile(
                    optimizer='Adam',
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy']
                )

                early_stop = EarlyStopping(monitor='val_loss', patience=int(patience), restore_best_weights=True)

                train_out.markdown('<div class="alert-ok">🚀 Training started — this may take a few minutes…</div>', unsafe_allow_html=True)

                history = model_t.fit(
                    X_train, y_train,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop],
                    verbose=0
                )

                model_t.save(MODEL_PATH)
                val_acc = max(history.history.get('val_categorical_accuracy', [0])) * 100

                train_out.markdown(
                    f'<div class="alert-ok">✅ Training complete!<br>'
                    f'Best Val Accuracy: {val_acc:.1f}%<br>'
                    f'Epochs run: {len(history.history["loss"])}<br>'
                    f'Saved → action_model.h5</div>',
                    unsafe_allow_html=True
                )
                # clear model cache so new model loads
                load_model.clear()

            except Exception as e:
                train_out.markdown(f'<div class="alert-warn">❌ Training failed:<br>{e}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — DATASET INFO
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="hud-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Dataset Overview</div>', unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        st.markdown('<div class="alert-warn">⚠ Dataset folder not found: sign_language_dataset/data/</div>', unsafe_allow_html=True)
    else:
        files = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
        if not files:
            st.markdown('<div class="alert-warn">⚠ No .npy files found. Start collecting data first.</div>', unsafe_allow_html=True)
        else:
            word_counts = {}
            total_size  = 0
            for f in files:
                word = f.split("_")[0]
                word_counts[word] = word_counts.get(word, 0) + 1
                total_size += os.path.getsize(os.path.join(DATA_PATH, f))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Samples", len(files))
            m2.metric("Unique Signs",  len(word_counts))
            m3.metric("Frames/Sample", FRAMES_PER_SAMPLE)
            m4.metric("Dataset Size",  f"{total_size/1024:.1f} KB")

            st.markdown("---")
            st.markdown('<div class="sec-label">Samples Per Sign</div>', unsafe_allow_html=True)

            bars_html = ""
            max_count = max(word_counts.values()) if word_counts else 1
            for word, count in sorted(word_counts.items()):
                pct = int(count / max_count * 100)
                bars_html += f"""
                <div class="conf-row" style="margin:8px 0">
                  <span class="conf-label" style="width:100px;font-size:0.9rem;color:var(--text)">{word}</span>
                  <div class="conf-track" style="height:14px">
                    <div class="conf-fill" style="width:{pct}%;height:100%"></div>
                  </div>
                  <span class="conf-pct" style="width:60px">{count} files</span>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # CSV info
    if os.path.exists("MP_Data.csv"):
        st.markdown('<div class="hud-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">MP_Data.csv Preview</div>', unsafe_allow_html=True)
        try:
            import pandas as pd
            df = pd.read_csv("MP_Data.csv")
            st.markdown(f'<div class="alert-ok">✓ {df.shape[0]} rows × {df.shape[1]} columns</div>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.markdown(f'<div class="alert-warn">Could not load CSV: {e}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:40px 0 20px;color:var(--muted);
            font-family:'Share Tech Mono',monospace;font-size:0.75rem;letter-spacing:2px;">
  SIGNSENSE AI  //  CNN + BiLSTM Architecture  //  MediaPipe Hands  //  Built with Streamlit
</div>
""", unsafe_allow_html=True)
import re
import io
import random
import pandas as pd
import streamlit as st

try:
    import pdfplumber
    PDF_OK = True
except Exception:
    PDF_OK = False

st.set_page_config(page_title="Gambare!", layout="wide")

# ---------------------------
# Utilities: parsing helpers
# ---------------------------

NOISE_PATTERNS = [
    r"(?im)^ *Quick *Answer:.*?$",
    r"(?im)^ *The Details:.*?$",
    r"(?im)^ *Practice Exam.*?$",
    r"(?im)^ *Answer:.*?$",
    r"(?im)^ *T\s*$",
    r"(?im)^ *\d+\s*$",
]

QUESTION_START_RE = re.compile(
    r"(?im)^\s*(?:Question\s+(\d+)\.|([A-Z])(\d+)\.)"
)

OPT_BLOCK_RE = re.compile(
    r"""(?imx)
    ^\s*(?:[❍\-\•\*]?\s*)?    # optional bullet
    \(?([A-Z])\)?[.)]\s*      # (A) / A. / A) / (A).
    (.+?)                     # option text, lazy
    (?=^\s*(?:[❍\-\•\*]?\s*)?\(?[A-Z]\)?[.)]\s+|^\s*(?:Question\s+\d+\.|[A-Z]\d+\.)|\Z)
    """,
    re.S
)

def clean_text(t: str) -> str:
    t = t.replace("\r\n", "\n")
    # --- Step 1: Normalize broken PDF words ---
    t = re.sub(r"Quick\s*\n\s*Answer", "Quick Answer", t, flags=re.I)
    t = re.sub(r"\bT\s*\n\s*he Details", "The Details", t, flags=re.I)
    # --- Step 2: Remove 'Quick Answer' / 'The Details' / 'Practice Exam' junk ---
    t = re.sub(r"Quick Answer:.*?(?=$|\n)", "", t, flags=re.I)
    t = re.sub(r"The Details:.*?(?=$|\n)", "", t, flags=re.I)
    t = re.sub(r"Practice Exam.*?- Questions", "", t, flags=re.I)
    # --- Step 3: Remove dangling leftovers ---
    t = re.sub(r"\bQuick\b", "", t, flags=re.I)
    t = re.sub(r"Answer:\s*\d+", "", t, flags=re.I)
    t = re.sub(r"The Details:\s*\d+", "", t, flags=re.I)
    # --- Step 4: Noise filters ---
    for pat in NOISE_PATTERNS:
        t = re.sub(pat, "", t)
    # --- Step 5: Collapse whitespace ---
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def split_into_question_blocks(text: str):
    blocks = []
    starts = [m for m in QUESTION_START_RE.finditer(text)]
    if not starts:
        return blocks
    for i, m in enumerate(starts):
        start = m.start()
        end = starts[i+1].start() if i + 1 < len(starts) else len(text)
        blocks.append(text[start:end].strip())
    return blocks

def parse_header(block: str):
    m = QUESTION_START_RE.match(block)
    if not m:
        return None, None, block
    if m.group(1):
        qid = f"Q{m.group(1)}"
    else:
        qid = f"{m.group(2)}{m.group(3)}"
    return qid, m.group(0), block[m.end():].strip()

def extract_options_and_question(rest: str):
    lines = [l.strip() for l in rest.splitlines() if l.strip()]
    matches = list(OPT_BLOCK_RE.finditer(rest))
    if matches:
        q_text = rest[:matches[0].start()].strip()
        opts_map = {}
        for m in matches:
            letter = m.group(1)
            body = m.group(2).strip()
            opts_map[letter] = body
        letters = sorted(opts_map.keys())
        options = [opts_map[ch] for ch in letters]
        return q_text, options, letters
    return rest.strip(), [], []

@st.cache_data
def parse_questions(big_text: str):
    text = clean_text(big_text)
    blocks = split_into_question_blocks(text)
    questions = []
    for block in blocks:
        qid, header, rest = parse_header(block)
        if not qid:
            continue
        q_text, options, letters = extract_options_and_question(rest)
        q_text = " ".join(q_text.split())
        options = [(" ".join(o.split())).strip() for o in options]
        questions.append({
            "qid": qid,
            "question": q_text,
            "options": options,
            "letters": letters or list("ABCD")
        })
    return questions

@st.cache_data
def load_pdf(file_bytes):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages)

# ---------------------------
# UI: Sidebar inputs
# ---------------------------

st.sidebar.title("Exam Setup")
source_mode = st.sidebar.radio(
    "Source of questions",
    ["Paste text", "Upload TXT", "Upload PDF" + (" (needs pdfplumber)" if not PDF_OK else "")]
)

big_text = ""
if source_mode == "Paste text":
    big_text = st.sidebar.text_area("Paste your questions here", height=300)
elif source_mode == "Upload TXT":
    f = st.sidebar.file_uploader("Upload .txt", type=["txt"])
    if f is not None:
        big_text = f.read().decode("utf-8", errors="ignore")
elif source_mode.startswith("Upload PDF"):
    f = st.sidebar.file_uploader("Upload .pdf", type=["pdf"])
    if f is not None and PDF_OK:
        big_text = load_pdf(f.read())
    elif f is not None and not PDF_OK:
        st.sidebar.error("Install pdfplumber: pip install pdfplumber")

shuffle_questions = st.sidebar.checkbox("Shuffle questions", value=False)
shuffle_options = st.sidebar.checkbox("Shuffle options", value=False)

answer_key_file = st.sidebar.file_uploader("Upload answer_key.csv (optional)", type=["csv"])
answer_key = {}
if answer_key_file is not None:
    df_key = pd.read_csv(answer_key_file)
    for _, row in df_key.iterrows():
        k = str(row["qid"]).strip()
        v = str(row["correct"]).strip().upper()
        answer_key[k] = {c.strip() for c in v.split(",") if c.strip()}

if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}

# ---------------------------
# Parse button
# ---------------------------

col_l, col_r = st.columns([1, 2])
with col_l:
    if st.button("Parse Questions", type="primary"):
        qs = parse_questions(big_text or "")
        if not qs:
            st.error("No questions detected.")
        else:
            if shuffle_questions:
                random.shuffle(qs)
            if shuffle_options:
                for q in qs:
                    opts = q["options"]
                    letters = q["letters"]
                    zipped = list(zip(letters, opts))
                    random.shuffle(zipped)
                    new_letters, new_opts = zip(*zipped)
                    q["options"] = list(new_opts)
                    q["letter_map"] = dict(zip(letters, new_letters))
                else:
                    for q in qs:
                        q["letter_map"] = {ch: ch for ch in q["letters"]}
            st.session_state.questions = qs
            st.session_state.current_idx = 0
            st.session_state.answers = {}
            st.success(f"Parsed {len(qs)} questions.")

with col_r:
    st.markdown("### You Shall Pass!")

# ---------------------------
# Exam view / navigation
# ---------------------------

qs = st.session_state.questions
if qs:
    # Navigation bar
    nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 3])
    if nav1.button("⟵ Previous", disabled=st.session_state.current_idx == 0):
        st.session_state.current_idx -= 1
    if nav2.button("Next ⟶", disabled=st.session_state.current_idx >= len(qs) - 1):
        qid = qs[st.session_state.current_idx]["qid"]
        if not st.session_state.answers.get(qid):
            st.warning("⚠️ Please select an answer before moving to the next question.")
        else:
            st.session_state.current_idx += 1
    with nav3:
        if st.button("Submit Exam", type="secondary"):
            st.session_state["submitted"] = True

    # Jump to Question (compact input box)
    with st.expander("Jump to Question"):
        col1, col2 = st.columns([1, 4])  # small input, big button
        with col1:
            jump_to = st.text_input("QID", placeholder="e.g. A33, B69, Q2", label_visibility="collapsed")
        with col2:
            if st.button("Go"):
                for i, q in enumerate(qs):
                    if q["qid"].lower() == jump_to.lower().strip():
                        st.session_state.current_idx = i
                        break

    # Display current question
    idx = st.session_state.current_idx
    q = qs[idx]
    st.markdown(f"**{q['qid']}**  \n{q['question']}")

    labels = q["letters"]
       # Options
    opts = q["options"]

    # Expand labels dynamically up to number of options
    labels = list("ABCDEFG")[:len(opts)]

    # Pad options if needed
    while len(opts) < len(labels):
        opts.append("")

    # --- Single vs Multiple answer handling ---
    if len(labels) > 4:  # Assume multi-answer (E, F, G present)
        choices = [f"{labels[i]}. {opts[i]}" if opts[i] else f"{labels[i]}." for i in range(len(labels))]

        # ✅ FIX: no default=, Streamlit will keep selection by key
        chosen = st.multiselect(
            "Choose all that apply:",
            choices,
            key=f"multi_{q['qid']}"
        )

        st.session_state.answers[q["qid"]] = [c.split(".")[0] for c in chosen]

    else:  # Normal single-answer
        labels = labels[:4]
        while len(opts) < 4:
            opts.append("")

        choices = ["-- Select an answer --"] + [
            f"{labels[i]}. {opts[i]}" if opts[i] else f"{labels[i]}." for i in range(4)
        ]

        current_selection = st.session_state.answers.get(q["qid"], None)
        if current_selection and current_selection in labels:
            index = labels.index(current_selection) + 1
        else:
            index = 0

        chosen = st.radio(
            "Choose one:",
            options=choices,
            index=index,
            key=f"radio_{q['qid']}"
        )

        if chosen != "-- Select an answer --":
            st.session_state.answers[q["qid"]] = chosen.split(".")[0].strip()
        else:
            st.session_state.answers[q["qid"]] = ""


    st.caption(f"Question {idx+1} of {len(qs)}")

    if st.session_state.get("submitted"):
        rows = []
        correct_count = 0
        for item in qs:
            qid = item["qid"]
            picked = st.session_state.answers.get(qid, [])
            if isinstance(picked, str):
                picked = [picked] if picked else []
            correct = answer_key.get(qid, set())
            if correct and "letter_map" in item:
                inv_map = {v: k for k, v in item["letter_map"].items()}
                displayed_correct = {inv_map.get(c, c) for c in correct}
            else:
                displayed_correct = correct
            is_correct = set(picked) == displayed_correct if displayed_correct else None
            if is_correct is True:
                correct_count += 1
            rows.append({
                "qid": str(qid),
                "your_choice": ",".join(picked),
                "correct_choice": ",".join(displayed_correct) if displayed_correct else "",
                "result": "✔️" if is_correct is True else ("❌" if is_correct is False else "—")
            })
        df = pd.DataFrame(rows, columns=["qid", "your_choice", "correct_choice", "result"])
        st.subheader("Results")
        if any(r["correct_choice"] for r in rows):
            total_count = sum(1 for r in rows if r["correct_choice"])
            score_percent = round((correct_count / total_count) * 100, 2)
            st.write(f"Score: **{correct_count} / {total_count}** ({score_percent}%)")
        else:
            st.info("No answer key uploaded.")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results (CSV)", data=csv, file_name="exam_results.csv", mime="text/csv")
else:
    st.info("Paste/upload questions on the left, then click **Parse Questions**.")

# ---------------------------
# Tips test
# ---------------------------
# with st.expander("tips for best parsing results"):
#    st.markdown("""
#- Supported question headers: **`Question N.`** or **`A##.` / `B##.`**
#- Supported options: `(A) Something...`, `A. Something...`, etc.
#- Misaligned case from PDFs is handled.
#- Upload an **answer_key.csv** with columns: `qid,correct`  
#  - Single answer: `Q12,A`  
#  - Multi-answer: `Q13,E,F,G`
#- Shuffle option remaps answers correctly.
#""")

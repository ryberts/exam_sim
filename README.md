# Exam Simulator

A Streamlit-based exam simulator designed to help professional certification exam takers practice multiple-choice and multi-answer questions with automated scoring.

🔗 **Live App**: [https://ryanexamsim.streamlit.app/](https://ryanexamsim.streamlit.app/)

---

## Features

- **Multiple Input Formats**: Paste text, upload TXT files, or upload PDFs (with `pdfplumber`)
- **Question Parsing**: Automatically detects questions and options from structured text
- **Shuffle Support**: Randomize question order and option order for varied practice
- **Answer Key Support**: Upload a CSV with correct answers to get scored results
- **Multi-Answer Questions**: Supports both single-answer and multiple-answer questions
- **Navigation**: Easy navigation between questions with jump-to functionality
- **Results Export**: Download your exam results as a CSV file

---

## How to Use

### 1. Input Your Questions

You can provide questions in three ways:
- **Paste Text**: Copy and paste your questions directly into the text area
- **Upload TXT**: Upload a `.txt` file containing your questions
- **Upload PDF**: Upload a `.pdf` file (requires `pdfplumber` library)

### 2. Parse Questions

Click the **"Parse Questions"** button to process your input. The app will:
- Detect individual questions
- Extract question text and options
- Display them in the exam interface

### 3. Take the Exam

- Navigate through questions using **Previous** and **Next** buttons
- Select your answers (single or multiple as appropriate)
- Use the **Jump to Question** feature to quickly navigate to specific questions

### 4. Submit and Score

- Click **"Submit Exam"** when finished
- If you uploaded an answer key, you'll see your score and which questions you got right/wrong
- Download your results as a CSV file for review

---

## Answer Key Format

Create a CSV file with columns: `qid,correct`

**Examples:**
- Single answer: `Q12,A`
- Multiple answers: `Q13,A,B,C`

Upload this file in the sidebar before submitting your exam.

---

## Question Format Tips

For best parsing results, format your questions like:

---
Question 1.
What is the capital of France?
(A) London
(B) Berlin
(C) Paris
(D) Madrid

A2.
Which colors are in the French flag?
A. Red
B. Blue
C. White
D. Green

---

Supported formats:
- Question headers: `Question N.` or `A##./B##.`
- Options: `(A) Text`, `A. Text`, `A) Text`

---

## Installation (For Local Development)

```bash
git clone <repository-url>
cd exam-simulator
pip install -r requirements.txt
streamlit run exam_simulator.py
---

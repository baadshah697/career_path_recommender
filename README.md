# 🎯 AI Career Path Recommender

> **IBM SkillsBuild · AI & Machine Learning Internship**  
> Aligned with **UN SDG 8 — Decent Work and Economic Growth**

An intelligent career guidance tool that takes a user's current skills and suggests the best-fit tech job roles — along with a personalised list of skills to learn next. Built specifically for job seekers in **Tier-2 and Tier-3 cities** in India who lack access to structured career counselling.

---

## 🌐 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

> Replace the link above with your Streamlit Cloud URL after deployment.

---

## 📸 App Overview

The app is organised into **6 tabs**:

| Tab | Description |
|-----|-------------|
| 🎯 **Recommender** | Enter your skills → get ranked career matches with match %, skills you have, and skills to learn |
| 📋 **Dataset Explorer** | Browse and filter all 20 job roles, domain distribution chart, skills-per-role stats |
| 🔬 **Skills Analysis** | Skill frequency chart, skill lookup (which roles need a skill?), TF-IDF weight heatmap |
| 🧠 **Model Insights** | Role-to-role cosine similarity matrix, pre-built sample user predictions, IDF weight table |
| ✅ **Sanity Check** | Self-match accuracy test — verifies each role predicts itself as #1 |
| 🤖 **AI Career Coach** | Local LLM chatbot (Ollama · phi3) for conversational career guidance and summaries |

---

## 🧠 How It Works

```
User Skills (text input)
        │
        ▼
  Alias Expansion  (e.g. "ml" → "machine learning", "aws" → "cloud computing")
        │
        ▼
  TF-IDF Vectorisation  (rare/specialised skills weighted higher)
        │
        ▼
  Cosine Similarity  (user vector vs. each role vector)
        │
        ▼
  Composite Score = 60% Cosine Similarity + 40% Skill Coverage [+ 5% Domain Bonus]
        │
        ▼
  Top N Ranked Roles  →  Skills You Have  +  Skills To Learn Next
```

### Scoring Formula

```
Match % = 0.6 × cosine_similarity + 0.4 × skill_coverage  [+ 0.05 if domain matches]
```

- **Cosine Similarity** — TF-IDF weighted directional overlap between the user's skills and a role's skills.  
- **Skill Coverage** — fraction of a role's required skills the user already holds.  
- **Domain Bonus** — optional +5% if the role belongs to the user's chosen interest area.

---

## 🗂️ Project Structure

```
Job_prediction/
├── app.py                          # Streamlit application (all 6 tabs)
├── career_path_recommender.ipynb   # Original Jupyter notebook
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Running Locally

### Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com) installed and running (for the AI Career Coach tab)

### 1 — Clone the repository

```bash
git clone https://github.com/your-username/Job_prediction.git
cd Job_prediction
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Pull the Ollama model (for the chatbot tab)

```bash
ollama pull phi3
```

> The AI Career Coach tab requires Ollama to be running locally.  
> Start it with: `ollama serve`  
> If Ollama is not available, all other 5 tabs work without it.

### 4 — Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploying to Streamlit Cloud

> **Note:** The **AI Career Coach** tab requires a locally running Ollama instance and is not available on Streamlit Cloud. All other 5 tabs deploy and work fully in the cloud.

1. Fork or push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app** → select your repository → set **Main file path** to `app.py`.
4. Click **Deploy**.

Streamlit Cloud will automatically install packages from `requirements.txt`.

> If you want to deploy **without** the chatbot tab (to avoid the `ollama` import error on Cloud), remove the `ollama` import and the `🤖 AI Career Coach` tab block from `app.py` before deploying, or wrap the import in a `try/except`.

---

## 💬 AI Career Coach (Ollama)

The chatbot tab runs a local **phi3** model via [Ollama](https://ollama.com) — no data leaves your machine.

| Feature | Detail |
|---------|--------|
| **Model** | `phi3:latest` (2.2 GB, runs on CPU/GPU) |
| **Streaming** | Token-by-token live output |
| **Context injection** | Current recommender results are automatically injected into the system prompt |
| **Quick prompts** | One-click buttons: Analyse my skills · What to learn next · Career summary · Job market tips |
| **Chat history** | Full conversation context passed on every turn |
| **Privacy** | 100% local — no API keys, no external calls |

---

## 📊 Dataset

20 hand-curated entry-level tech roles across 12 domains:

| Domain | Roles |
|--------|-------|
| Data Science & AI | Data Scientist, ML Engineer, Computer Vision Engineer |
| Software Development | Python Backend, Java Developer, Mobile App Developer |
| Web Development | Frontend Developer, Full Stack Developer |
| Data Analytics | Data Analyst |
| Cloud & DevOps | DevOps Engineer, Cloud Engineer |
| Data & Databases | Data Engineer, Database Administrator |
| Quality Assurance | Software Tester (QA) |
| Security | Cybersecurity Analyst |
| Business | Business Analyst |
| Design | UI/UX Designer |
| Marketing | Digital Marketing Executive, Content Writer |
| IT Support | Technical Support Engineer |

---

## 🔤 Skill Shortcuts

The recommender understands common shorthand:

| You type | Expands to |
|----------|------------|
| `ml` | machine learning |
| `dl` | deep learning |
| `js` | javascript |
| `aws`, `azure`, `cloud` | cloud computing |
| `tf` | tensorflow |
| `spark` | apache spark |
| `powerbi`, `power-bi` | power bi |
| `node`, `nodejs` | node.js |
| `rest`, `restapi` | rest api |
| `cv` | opencv |
| `ci`, `cicd` | ci/cd |
| `spring` | spring boot |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | [Streamlit](https://streamlit.io) 1.62 |
| ML | scikit-learn (TF-IDF + Cosine Similarity) |
| Data | pandas, numpy |
| Local LLM | [Ollama](https://ollama.com) + phi3 |
| Language | Python 3.10+ |

---

## ⚠️ Limitations

- Dataset is small (20 roles) and hand-written — prototype quality.
- Skills must be typed; spelling variants outside the alias list are ignored.
- Does not consider location, salary, years of experience, or job availability.
- The AI Career Coach tab requires a locally running Ollama instance.

---

## 🔭 Future Scope

- Expand the dataset using real job postings scraped from Naukri / LinkedIn.
- Add **regional language support** (Hindi, Tamil, Telugu, etc.).
- Integrate **IBM watsonx** to generate fully personalised career summaries.
- Link each missing skill to free courses on **IBM SkillsBuild** / Coursera.
- Add experience-level and salary range filters.

---

## 📜 License

This project was built as part of the **IBM SkillsBuild AI & Machine Learning Internship**.  
Released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- **IBM SkillsBuild** for the internship programme and learning resources.
- **United Nations SDG 8** — Decent Work and Economic Growth.
- [Ollama](https://ollama.com) for making local LLMs accessible.
- [Streamlit](https://streamlit.io) for the open-source app framework.

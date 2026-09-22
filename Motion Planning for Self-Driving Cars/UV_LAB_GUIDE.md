# Student Guide: Setting Up Python Lab Environments with `uv`

A complete guide for students starting from a **blank installation** (no Python, no virtual environments, no pre-installed dependencies).

---

## ⚡ Step 0: Install `uv` (First-Time Only)

`uv` automatically manages Python versions and virtual environments. **Students do NOT need to install Python beforehand**—`uv` will download and install the exact required Python version automatically.

### macOS & Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*(After running, restart your terminal or run `source $HOME/.cargo/env` / `source $HOME/.local/bin/env`)*

### Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Or via Homebrew / Winget:
* **macOS (Homebrew)**: `brew install uv`
* **Windows (Winget)**: `winget install --id=astral-sh.uv`

---

## 🚀 Scenario 1: Starting `planning_p2` from Scratch

If you just cloned or downloaded the [planning_p2](file:///Users/yasuomaidana/Projects/classes/Self-Driving-Cars/Motion%20Planning%20for%20Self-Driving%20Cars/planning_p2) directory:

```bash
# 1. Open your terminal and enter the lab folder
cd "planning_p2"

# 2. Automatically install the required Python version and all lab libraries
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

> [!TIP]
> `uv sync` reads `pyproject.toml` and `uv.lock`, automatically installs Python 3.14/3.12 (if not already on your system), creates `.venv`, and installs all dependencies (`numpy`, `polars`, `scipy`, `plotly`, `ipykernel`, etc.) in seconds.

---

## 📁 Scenario 2: Starting with Only a `pyproject.toml` File

If the instructor only shared a single `pyproject.toml` file (no repo, no `.venv`, no other files):

1. **Create an empty folder and put `pyproject.toml` inside it**:
   ```bash
   mkdir my_lab
   cd my_lab
   # (Place the instructor's pyproject.toml into this folder)
   ```

2. **Generate the environment**:
   ```bash
   uv sync
   ```
   *`uv` will generate `uv.lock`, fetch Python, create `.venv`, and install all packages declared in `pyproject.toml`.*

3. **Activate & Run**:
   ```bash
   # macOS / Linux
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate
   ```

---

## 🛠️ How to Execute Code & Notebooks

### 1. Running Python Scripts
You can run scripts inside the virtual environment without even activating it:
```bash
uv run python my_script.py
```

### 2. Running Jupyter Notebooks (`.ipynb`)
```bash
uv run jupyter lab
# or
uv run jupyter notebook
```

### 3. Using VS Code / Cursor / PyCharm:
1. Open the project folder in VS Code / Cursor.
2. Open any `.ipynb` file.
3. In the top-right corner, click **Select Kernel** $\rightarrow$ **Python Environments...** $\rightarrow$ select `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows).

---

## 💡 Alternatives: Which Approach Should You Use?

| Method | What the Instructor Shares | What the Student Runs | Best For |
| :--- | :--- | :--- | :--- |
| **`pyproject.toml` + `uv.lock`** *(Recommended)* | The whole lab folder or just the two config files | `uv sync` | Complete projects, deterministic dependencies, team labs |
| **Inline PEP 723 Script** | A single `.py` file with `# /// script` header | `uv run script.py` | Single-file homework scripts without managing folders |
| **`requirements.txt`** | A classic `requirements.txt` file | `uv venv` <br> `uv pip install -r requirements.txt` | Legacy course material |

---

## 📦 For Instructors: Creating a New Lab Environment from Scratch

If you want to start a brand new lab for your students:

```bash
# 1. Initialize as an application/environment (not an installable package)
uv init --app lab_name
cd lab_name

# 2. Pin desired Python version
uv python pin 3.12

# 3. Add lab libraries
uv add numpy scipy polars plotly ipykernel

# 4. Distribute pyproject.toml (and uv.lock) to students
```

> [!NOTE]
> Setting `[tool.uv]` with `package = false` in `pyproject.toml` (or initializing with `uv init --app`) ensures that `uv sync` will not fail if students only have `pyproject.toml` without the `src/` directory.


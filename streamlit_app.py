import streamlit as st
import sys
import os
import importlib.util

# -------------------------------------------------------------------
# ✅ 1. Setup paths
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # root of this file
LAB_DIR = os.path.join(BASE_DIR, "labs_app")
HW_DIR = os.path.join(BASE_DIR, "HWs")  # optional if HWs exist

# Add LAB_DIR and HW_DIR to sys.path for dynamic imports
sys.path.extend([LAB_DIR, HW_DIR])

# -------------------------------------------------------------------
# ✅ 2. Custom Lab Titles Mapping
# -------------------------------------------------------------------
custom_lab_titles = {
    "lab1": "Lab 1 - FileBot Q&A Assistant 📄",
    "lab2": "Lab 2 — PDF Summarizer 🤖",
    "lab3": "Lab 3 — Interactive Kid-Friendly Streaming Chatbot",
    "lab4": "Lab 4 — PDF📄 Q&A with ChromaDB + OpenAI",
    "lab5": "Lab 5 — What to Wear Bot 👕",
    "lab6": "Lab 6 - AI Fact-Checker ✅ + Citation Builder 🔗",
}

# -------------------------------------------------------------------
# ✅ 3. Scan labs_app folder for lab files
# -------------------------------------------------------------------
def get_lab_modules():
    labs = []
    if os.path.exists(LAB_DIR):
        for file in sorted(os.listdir(LAB_DIR)):
            if file.startswith("lab") and file.endswith(".py"):
                lab_name = os.path.splitext(file)[0]
                display_name = custom_lab_titles.get(
                    lab_name, lab_name.replace("lab", "Lab ").title()
                )
                labs.append((display_name, lab_name))
    return labs

lab_modules = get_lab_modules()

# -------------------------------------------------------------------
# ✅ 4. Scan HWs folder for HW files dynamically
# -------------------------------------------------------------------
def get_hw_modules():
    hws = []
    if os.path.exists(HW_DIR):
        for file in sorted(os.listdir(HW_DIR)):
            if file.startswith("HW") and file.endswith(".py"):
                hw_name = os.path.splitext(file)[0]
                display_name = hw_name.replace("HW", "HW ").title()
                hws.append((display_name, hw_name))
    return hws

hw_modules = get_hw_modules()

# -------------------------------------------------------------------
# ✅ 5. Sidebar Navigation
# -------------------------------------------------------------------
st.set_page_config(page_title="Labs and HWs App", page_icon="📘")

page_groups = {
    "🧪 Labs": [name for name, _ in lab_modules],
    "📚 Homeworks": [name for name, _ in hw_modules],
}

with st.sidebar:
    st.markdown("## 📘 Labs & HWs Navigator")
    section = st.radio("Select Section", list(page_groups.keys()))
    page = st.radio("Select Page", page_groups[section])

# -------------------------------------------------------------------
# ✅ 6. Map display name to module name safely
# -------------------------------------------------------------------
page_module_map = {name: mod for name, mod in (lab_modules + hw_modules)}
module_name = page_module_map.get(page)
if not module_name:
    st.error(f"Selected page '{page}' does not map to a module.")
    st.stop()

# -------------------------------------------------------------------
# ✅ 7. Load and Run Module Dynamically
# -------------------------------------------------------------------
def load_module_from_path(module_name, folder_path):
    module_file = os.path.join(folder_path, f"{module_name}.py")
    if not os.path.exists(module_file):
        raise FileNotFoundError(f"{module_file} not found")
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    # Determine folder dynamically
    folder_path = LAB_DIR if module_name.startswith("lab") else HW_DIR
    module = load_module_from_path(module_name, folder_path)

    if hasattr(module, "app"):
        module.app()
    elif hasattr(module, "main"):
        module.main()
    else:
        st.warning(f"⚠️ `{module_name}` has no `app()` or `main()` function.")

except Exception as e:
    st.error(f"🚨 Error loading `{module_name}`: {e}")

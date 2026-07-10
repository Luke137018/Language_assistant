import streamlit as st
from openai import OpenAI
from dotenv import dotenv_values
import time
from deep_translator import GoogleTranslator

env = dotenv_values('.env')

if 'openai_api_key' not in st.session_state:
    st.session_state['openai_api_key'] = ''

with st.sidebar:
    st.header("Ustawienia API")
    st.session_state['openai_api_key'] = st.text_input(
        "Podaj swój OpenAI API Key:",
        type="password",
        value=st.session_state['openai_api_key']
    )

# Try to use user-provided key first, fallback to .env
openai_api_key = st.session_state['openai_api_key'] or dotenv_values('.env').get('OPENAI_API_KEY')
if openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)
else:
    openai_client = None




try:
    openai_client = OpenAI(api_key=env['OPENAI_API_KEY'])
except KeyError:
    openai_client = None

# Lista obsługiwanych języków
LANGUAGES = {
    "Angielski": "en",
    "Hiszpański": "es",
    "Niemiecki": "de",
    "Francuski": "fr",
    "Włoski": "it",
    "Portugalski (brazylijski)": "pt",
    "Rosyjski": "ru",
    "Chiński (mandaryński)": "zh",
    "Japoński": "ja",
    "Arabski": "ar",
    "Polski": "pl",
    "Turecki": "tr",
    "Koreański": "ko",
    "Hindi": "hi",
    "Indonezyjski": "id",
    "Niderlandzki": "nl",
    "Herbajski": "he",
    "Grecki": "el",
    "Aramejski": "arc",
    "Czeski": "cs",
    "Szwedzki": "sv",
    "Norweski": "no",
    "Duński": "da",
    "Ukraiński": "uk"
}


# ====================================================MAIN APP=====================================================================
# =================================================================================================================================

st.set_page_config(page_title="Language assistant", layout="wide")

st.title('Pomocnik językowy   |   Language assistant')
st.markdown('Tłumaczenie tekstów na 22 popularne języki + hebrajski, aramejski, grecki')

# ==================== TAB 1: TRANSLATOR (GoogleTranslator) ====================
def translate_with_google(text: str, target_lang_code: str) -> str:
    """Translate Polish text to target language using GoogleTranslator."""
    try:
        return GoogleTranslator(source="pl", target=target_lang_code).translate(text)
    except Exception as e:
        return f"❌ Błąd tłumaczenia: {str(e)}"


# ====================== tab3  tlumaczenie

def correct_foreign_text_with_gpt(text: str, source_lang: str) -> str:
    """Correct spelling and grammar mistakes in a foreign-language text."""
    if not openai_client:
        return "❌ Błąd: Brak klucza API OpenAI. Sprawdź plik .env"

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Jesteś korektorem tekstów napisanych po {source_lang}. Popraw błędy ortograficzne, gramatyczne i interpunkcyjne. Zwróć tylko poprawioną wersję tekstu, bez dodatkowych komentarzy."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Błąd API: {str(e)}"


def translate_to_polish_with_google(text: str, source_lang_code: str) -> str:
    """Translate text from a foreign language to Polish using GoogleTranslator."""
    try:
        return GoogleTranslator(source=source_lang_code, target="pl").translate(text)
    except Exception as e:
        return f"❌ Błąd tłumaczenia: {str(e)}"

# ==================== GOOGLE TRANSLATOR ====================
st.header("📝 Tłumaczenie translatorem")
st.write("Szybkie tłumaczenie z polskiego na wybrany język")

if "tab1_translations" not in st.session_state:
    st.session_state.tab1_translations = []

col1, col2 = st.columns([3, 1])
with col1:
    target_language_tab1 = st.selectbox(
        "Wybierz język docelowy:",
        options=list(LANGUAGES.keys()),
        key="tab1_language",
        index=None
    )
with col2:
    st.write("")
    st.write("")
    st.info(f"🎯 {target_language_tab1}")

col1, col2 = st.columns([4, 1])
with col1:
    polish_text_tab1 = st.text_input(
        "Wpisz tekst po polsku:",
        placeholder="np. Wpisz tekst w języku do przetłumaczenia.",
        key="tab1_input"
    )
with col2:
    st.write("")
    st.write("")
    translate_btn_tab1 = st.button("Tłumacz", key="tab1_button", use_container_width=True)

if translate_btn_tab1:
    if not polish_text_tab1.strip():
        st.warning("⚠️ Proszę wpisać tekst po polsku")
    else:
        target_lang_code = LANGUAGES[target_language_tab1]
        translated = translate_with_google(polish_text_tab1, target_lang_code)
        st.session_state.tab1_translations.append({
            "source": polish_text_tab1,
            "target": translated,
            "language": target_language_tab1
        })
        st.success("✅ Tekst przetłumaczony!")

if st.session_state.tab1_translations:
    st.subheader("📚 Historia tłumaczeń")
    for translation in reversed(st.session_state.tab1_translations):
        with st.container(border=True):
            st.markdown(f"**PL:** {translation['source']}")
            st.markdown(f"**{translation['language']}:** {translation['target']}")

st.divider()

# ==================== ERROR CORRECTION ====================
st.header("🛠️ Poprawianie błędów w tekście obcym")
st.write("Wpisz tekst w wybranym języku obcym, popraw błędy i zobacz tłumaczenie na polski")

if "tab3_corrected_text" not in st.session_state:
    st.session_state.tab3_corrected_text = ""
if "tab3_translation" not in st.session_state:
    st.session_state.tab3_translation = ""

col1, col2 = st.columns([3, 1])
with col1:
    source_language_tab3 = st.selectbox(
        "Wybierz język tekstu:",
        options=list(LANGUAGES.keys()),
        key="tab3_language",
        index=None
    )
with col2:
    st.write("")
    st.write("")
    st.info(f"🎯 {source_language_tab3}")

foreign_text_tab3 = st.text_area(
    "Wpisz tekst z błędami:",
    placeholder="np. Hallo, wie geit es dir?",
    key="tab3_input",
    height=220
)

if st.button("Popraw błędy", key="tab3_correct_button", use_container_width=True):
    if not foreign_text_tab3.strip():
        st.warning("⚠️ Proszę wpisać tekst do poprawy")
    elif not source_language_tab3:
        st.warning("⚠️ Proszę wybrać język tekstu")
    else:
        corrected_text = correct_foreign_text_with_gpt(foreign_text_tab3, source_language_tab3)
        st.session_state.tab3_corrected_text = corrected_text
        st.session_state.tab3_translation = ""
        st.success("✅ Tekst poprawiony!")

if st.session_state.tab3_corrected_text:
    st.subheader("✍️ Poprawiona wersja")
    st.write(st.session_state.tab3_corrected_text)

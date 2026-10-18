import csv
import io
from typing import List

import instructor
import streamlit as st
from dotenv import dotenv_values
from openai import OpenAI
from pydantic import BaseModel, Field


MODEL = "gpt-4o"
LANGUAGES = {
    "Angielski": "en", "Hiszpański": "es", "Niemiecki": "de",
    "Francuski": "fr", "Włoski": "it", "Portugalski (brazylijski)": "pt",
    "Rosyjski": "ru", "Chiński (mandaryński)": "zh", "Japoński": "ja",
    "Arabski": "ar", "Polski": "pl", "Turecki": "tr", "Koreański": "ko",
    "Hindi": "hi", "Indonezyjski": "id", "Niderlandzki": "nl",
    "Hebrajski": "he", "Grecki": "el", "Aramejski": "arc",
    "Czeski": "cs", "Szwedzki": "sv", "Norweski": "no", "Duński": "da",
    "Ukraiński": "uk",
}


class VocabularyItem(BaseModel):
    word: str = Field(description="Pojedyncze słowo lub krótki, często używany zwrot w języku docelowym")
    meaning_pl: str = Field(description="Polskie znaczenie")
    example: str = Field(description="Krótkie, naturalne zdanie z tym słowem w języku docelowym")
    example_pl: str = Field(description="Polskie tłumaczenie zdania przykładowego")


class GrammarTip(BaseModel):
    title: str = Field(description="Krótka nazwa konstrukcji gramatycznej")
    explanation_pl: str = Field(description="Przystępne wyjaśnienie po polsku")
    example: str = Field(description="Przykład użycia w języku docelowym")
    example_pl: str = Field(description="Polskie tłumaczenie przykładu")


class LearningPack(BaseModel):
    vocabulary: List[VocabularyItem] = Field(description="Dokładnie 5 przydatnych i interesujących haseł")
    grammar_tips: List[GrammarTip] = Field(description="Kilka praktycznych wskazówek gramatycznych")


def get_clients(api_key: str):
    if not api_key:
        raise ValueError("Podaj klucz API OpenAI w panelu bocznym lub w pliku .env.")
    client = OpenAI(api_key=api_key)
    return client, instructor.from_openai(client)


def generate_learning_pack(structured_client, language: str) -> LearningPack:
    return structured_client.chat.completions.create(
        model=MODEL,
        response_model=LearningPack,
        temperature=0.8,
        messages=[
            {
                "role": "system",
                "content": (
                    "Jesteś nauczycielem języka. Przygotuj materiał dla polskiego ucznia. "
                    "Zwróć dokładnie 5 ciekawych, praktycznych słów lub krótkich zwrotów "
                    "wskazanego języka wraz ze znaczeniem po polsku i naturalnym przykładem. "
                    "Dodaj 3 krótkie, przydatne wskazówki gramatyczne. Wszystkie objaśnienia "
                    "i polskie tłumaczenia napisz po polsku; przykłady podaj w języku docelowym."
                ),
            },
            {"role": "user", "content": f"Język docelowy: {language}."},
        ],
    )


def translate_with_gpt(client, text: str, target_language: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": f"Przetłumacz tekst na język: {target_language}. Zachowaj sens i styl. Zwróć wyłącznie tłumaczenie."},
            {"role": "user", "content": text},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def correct_with_gpt(client, text: str, source_language: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": f"Popraw błędy ortograficzne, gramatyczne i interpunkcyjne w tekście po {source_language}. Zwróć wyłącznie poprawiony tekst."},
            {"role": "user", "content": text},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def translations_csv(rows) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["Język", "Tekst źródłowy", "Tłumaczenie"])
    writer.writerows((row["language"], row["source"], row["translation"]) for row in rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


st.set_page_config(page_title="Language assistant", layout="wide")
st.title("Pomocnik językowy | Language assistant")
st.write("Tłumacz teksty, poprawiaj błędy i poznawaj nowe słownictwo oraz gramatykę.")

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "translations" not in st.session_state:
    st.session_state.translations = []

with st.sidebar:
    st.header("Ustawienia API")
    st.session_state.openai_api_key = st.text_input(
        "Podaj swój OpenAI API Key:", type="password", value=st.session_state.openai_api_key
    )

api_key = st.session_state.openai_api_key or dotenv_values(".env").get("OPENAI_API_KEY", "")
languages = list(LANGUAGES)
tab_learn, tab_translate, tab_correct = st.tabs(["Słówka i gramatyka", "Tłumacz", "Popraw tekst"])

with tab_learn:
    language = st.selectbox("Wybierz język docelowy:", languages, key="learning_language")
    if st.button("Wygeneruj wyjaśnienia słów i gramatyki", key="generate_learning"):
        try:
            _, structured_client = get_clients(api_key)
            with st.spinner("Przygotowuję słówka i wskazówki gramatyczne..."):
                pack = generate_learning_pack(structured_client, language)
            if len(pack.vocabulary) != 5:
                raise ValueError("Model nie zwrócił dokładnie 5 haseł. Spróbuj ponownie.")
            st.session_state.learning_pack = pack
            st.session_state.learning_language = language
        except Exception as exc:
            st.error(f"Nie udało się wygenerować materiału: {exc}")

    pack = st.session_state.get("learning_pack")
    if pack:
        st.subheader(f"Słówka: {st.session_state.learning_language}")
        for item in pack.vocabulary:
            with st.container(border=True):
                st.markdown(f"**{item.word}** — {item.meaning_pl}")
                st.markdown(f"*{item.example}*  \n{item.example_pl}")
        st.subheader("Wskazówki gramatyczne")
        for tip in pack.grammar_tips:
            with st.container(border=True):
                st.markdown(f"**{tip.title}**")
                st.write(tip.explanation_pl)
                st.markdown(f"*{tip.example}*  \n{tip.example_pl}")

with tab_translate:
    target_language = st.selectbox("Tłumacz na:", languages, key="translation_language")
    source_text = st.text_area("Wpisz tekst do tłumaczenia:", key="translation_input", height=160)
    if st.button("Tłumacz", key="translate_button"):
        if not source_text.strip():
            st.warning("Wpisz tekst do tłumaczenia.")
        else:
            try:
                client, _ = get_clients(api_key)
                with st.spinner("Tłumaczę..."):
                    translated = translate_with_gpt(client, source_text.strip(), target_language)
                st.session_state.translations.append({
                    "source": source_text.strip(), "translation": translated, "language": target_language
                })
                st.success("Tłumaczenie gotowe.")
            except Exception as exc:
                st.error(f"Nie udało się przetłumaczyć tekstu: {exc}")

    if st.session_state.translations:
        st.subheader("Historia tłumaczeń")
        for row in reversed(st.session_state.translations):
            with st.container(border=True):
                st.markdown(f"**Tekst:** {row['source']}")
                st.markdown(f"**{row['language']}:** {row['translation']}")
        st.download_button(
            "Pobierz tłumaczenia (CSV)",
            data=translations_csv(st.session_state.translations),
            file_name="tlumaczenia.csv",
            mime="text/csv; charset=utf-8",
        )

with tab_correct:
    source_language = st.selectbox("Język tekstu:", languages, key="correction_language")
    text_to_correct = st.text_area("Wpisz tekst do poprawy:", key="correction_input", height=220)
    if st.button("Popraw błędy i przetłumacz na polski", key="correct_button"):
        if not text_to_correct.strip():
            st.warning("Wpisz tekst do poprawy.")
        else:
            try:
                client, _ = get_clients(api_key)
                with st.spinner("Poprawiam tekst..."):
                    corrected = correct_with_gpt(client, text_to_correct.strip(), source_language)
                st.session_state.corrected_text = corrected
                with st.spinner("Tłumaczę poprawiony tekst na polski..."):
                    st.session_state.corrected_translation = translate_with_gpt(
                        client, corrected, "polski"
                    )
            except Exception as exc:
                st.error(f"Nie udało się poprawić tekstu: {exc}")
    if st.session_state.get("corrected_text"):
        st.subheader("Poprawiony tekst")
        st.write(st.session_state.corrected_text)
        st.subheader("Tłumaczenie na polski")
        st.write(st.session_state.get("corrected_translation", ""))

import csv
import io
import re
from typing import List

import instructor
import streamlit as st
from dotenv import dotenv_values
from openai import OpenAI
from pydantic import BaseModel, Field


MODEL = "gpt-4o"
TTS_MODEL = "tts-1"
TTS_VOICE = "alloy"
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


class GrammarConstruction(BaseModel):
    fragment: str = Field(description="Dokładny fragment tłumaczenia ilustrujący konstrukcję")
    name: str = Field(description="Krótka nazwa konstrukcji gramatycznej")
    explanation_pl: str = Field(description="Jasne wyjaśnienie po polsku, jak działa konstrukcja w tym zdaniu")


class TranslationGrammarExplanation(BaseModel):
    overview_pl: str = Field(description="Krótki opis najważniejszych cech gramatycznych tłumaczenia")
    constructions: List[GrammarConstruction] = Field(description="Konstrukcje rzeczywiście użyte w tłumaczeniu")


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


def explain_translation_grammar(structured_client, source_text: str, translated_text: str, target_language: str) -> TranslationGrammarExplanation:
    return structured_client.chat.completions.create(
        model=MODEL,
        response_model=TranslationGrammarExplanation,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Jesteś nauczycielem gramatyki języka obcego. Wyjaśnij po polsku konstrukcje "
                    "gramatyczne faktycznie użyte w podanym tłumaczeniu. Przeanalizuj każde zdanie "
                    "osobno. Wskaż dokładne fragmenty z tłumaczenia, nazwij konstrukcję i opisz "
                    "jej funkcję w kontekście zdania. "
                    "Nie dopisuj konstrukcji, których nie ma w tekście. Jeśli zdanie jest proste, "
                    "wyjaśnij jego podstawowy szyk lub formę czasownika."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Język tłumaczenia: {target_language}.\n"
                    f"Tekst źródłowy: {source_text}\n"
                    f"Tłumaczenie do analizy: {translated_text}"
                ),
            },
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


def generate_speech(client, text: str) -> bytes:
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="mp3",
    )
    return response.content


def render_audio_button(api_key: str, text: str, key: str, label: str = "Generuj audio") -> None:
    cache_key = f"audio_{key}"
    if st.button(label, key=f"generate_{key}", disabled=not text.strip()):
        try:
            with st.spinner("Generuję audio..."):
                client, _ = get_clients(api_key)
                st.session_state[cache_key] = generate_speech(client, text)
        except Exception as exc:
            st.error(f"Nie udało się wygenerować audio: {exc}")
    if st.session_state.get(cache_key):
        st.audio(st.session_state[cache_key], format="audio/mp3")


def all_work_csv(translations, learning_packs, corrections) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["Rodzaj", "Język", "Tekst / hasło", "Znaczenie / wyjaśnienie", "Przykład", "Tłumaczenie przykładu"])
    for row in translations:
        writer.writerow(["Tłumaczenie", row["language"], row["source"], row["translation"], "", ""])
    for row in corrections:
        writer.writerow(["Korekta i tłumaczenie", row["language"], row["source"], row["corrected"], row["corrected"], row["translation_pl"]])
    for entry in learning_packs:
        language = entry["language"]
        pack = entry["pack"]
        for item in pack.vocabulary:
            writer.writerow(["Słówko", language, item.word, item.meaning_pl, item.example, item.example_pl])
        for tip in pack.grammar_tips:
            writer.writerow(["Gramatyka", language, tip.title, tip.explanation_pl, tip.example, tip.example_pl])
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def grammar_tip_text(language: str, tip: GrammarTip) -> str:
    return (
        f"Wyjaśnienie gramatyczne — {language}\n\n"
        f"{tip.title}\n\n{tip.explanation_pl}\n\n"
        f"Przykład ({language}): {tip.example}\n"
        f"Tłumaczenie: {tip.example_pl}\n"
    )


def translation_grammar_text(language: str, source: str, translation: str, explanation: dict) -> str:
    lines = [
        f"Wyjaśnienie gramatyczne — {language}",
        f"Tekst źródłowy: {source}",
        f"Tłumaczenie: {translation}",
        "",
        explanation["overview_pl"],
    ]
    for construction in explanation["constructions"]:
        lines.extend([
            "",
            f"{construction['name']} — {construction['fragment']}",
            construction["explanation_pl"],
        ])
    return "\n".join(lines) + "\n"


def safe_filename_part(value: str) -> str:
    return re.sub(r"[^\w-]+", "_", value, flags=re.UNICODE).strip("_") or "jezyk"


st.set_page_config(page_title="Language assistant", layout="wide")
st.markdown("""
<style>
  .stApp { background: #fffaf5; }
  [data-testid="stHeader"] { background: rgba(255,250,245,.92); }
  .hero { padding: 1.1rem 1.45rem; border: 1px solid #f2c6c6; border-radius: 18px;
    background: linear-gradient(110deg,#ffc9c9 0%,#ffec99 55%,#b2f2bb 100%); margin-bottom: 1rem; }
  .hero h1 { margin: 0; color: #342b2b; font-size: 2rem; }
  .hero p { margin: .3rem 0 0; color: #554747; }
  [data-testid="stVerticalBlockBorderWrapper"] { border-color: #f0d7c5; background: rgba(255,255,255,.78); }
  [data-testid="stBaseButton-primary"] { background: #f3aaa9; border-color: #e99a99; color: #302525; }
  [data-testid="stBaseButton-primary"]:hover { background: #ed9292; border-color: #e48989; color: #302525; }
  div[data-testid="stTabs"] button[role="tab"] { background: #e8f7eb; border-radius: 12px 12px 0 0; }
  div[data-testid="stTabs"] button[aria-selected="true"] { background: #b2f2bb; color: #24362a; }
  .section-kicker { color: #815f50; font-size: .85rem; text-transform: uppercase; letter-spacing: .08em; }
</style>
<div class="hero"><h1>Language assistant</h1><p>Tłumacz, poprawiaj i ucz się języków w jednym miejscu.</p></div>
""", unsafe_allow_html=True)

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "translations" not in st.session_state:
    st.session_state.translations = []
if "learning_packs" not in st.session_state:
    st.session_state.learning_packs = []
if "corrections" not in st.session_state:
    st.session_state.corrections = []

with st.sidebar:
    st.header("Ustawienia API")
    st.session_state.openai_api_key = st.text_input(
        "Podaj swój OpenAI API Key:", type="password", value=st.session_state.openai_api_key
    )

api_key = st.session_state.openai_api_key or dotenv_values(".env").get("OPENAI_API_KEY", "")
languages = list(LANGUAGES)
tab_translate, tab_exercises = st.tabs(["Tłumaczenie", "Ćwiczenia"])

with tab_translate:
    target_language = st.selectbox("Wybierz język tłumaczenia:", languages, key="translation_language")
    input_col, history_col = st.columns([1.55, 1], gap="large")
    with input_col:
        st.markdown('<div class="section-kicker">Tłumaczenie tekstu</div>', unsafe_allow_html=True)
        source_text = st.text_area("Wpisz tekst po polsku", key="translation_input", height=180, placeholder="Wpisz tekst do przetłumaczenia na wybrany język...")
        if st.button("Tłumacz", key="translate_button", type="primary", use_container_width=True):
            if not source_text.strip():
                st.warning("Wpisz tekst do tłumaczenia.")
            else:
                try:
                    client, _ = get_clients(api_key)
                    with st.spinner("Tłumaczę..."):
                        translated = translate_with_gpt(client, source_text.strip(), target_language)
                    st.session_state.translations.append({"source": source_text.strip(), "translation": translated, "language": target_language})
                    st.success("Tłumaczenie gotowe.")
                except Exception as exc:
                    st.error(f"Nie udało się przetłumaczyć tekstu: {exc}")


        with st.container(border=True):
            st.markdown("**Tłumacz i popraw błędy**")
            correction_language = st.selectbox("Język tekstu obcego:", languages, key="inline_correction_language")
            foreign_text = st.text_area("Wpisz tekst w języku obcym", key="inline_correction_input", height=120, placeholder="Tekst do korekty i tłumaczenia na polski...")
            if st.button("Popraw błędy i przetłumacz", key="inline_correct_button", use_container_width=True):
                if not foreign_text.strip():
                    st.warning("Wpisz tekst do poprawy.")
                else:
                    try:
                        client, _ = get_clients(api_key)
                        with st.spinner("Poprawiam tekst i tłumaczę na polski..."):
                            corrected = correct_with_gpt(client, foreign_text.strip(), correction_language)
                            translation_pl = translate_with_gpt(client, corrected, "polski")
                        correction = {"source": foreign_text.strip(), "corrected": corrected, "translation_pl": translation_pl, "language": correction_language}
                        st.session_state.corrections.append(correction)
                        st.session_state.corrected_text = corrected
                        st.session_state.corrected_translation = translation_pl
                    except Exception as exc:
                        st.error(f"Nie udało się poprawić tekstu: {exc}")
            if st.session_state.get("corrected_text"):
                st.markdown(f"**Poprawiona wersja:** {st.session_state.corrected_text}")
                st.markdown(f"**Po polsku:** {st.session_state.get('corrected_translation', '')}")

    with history_col:
        st.markdown('<div class="section-kicker">Historia tłumaczeń</div>', unsafe_allow_html=True)
        if st.session_state.translations:
            for index, row in enumerate(reversed(st.session_state.translations), start=1):
                with st.container(border=True):
                    st.caption(f"{index}. {row['language']}")
                    st.markdown(f"**PL:** {row['source']}")
                    st.markdown(f"**{row['language']}:** {row['translation']}")
                    row_index = len(st.session_state.translations) - index
                    render_audio_button(api_key, row["translation"], f"translation_{row_index}")
                    if st.button("Wyjaśnij gramatykę", key=f"explain_grammar_{row_index}"):
                        try:
                            _, structured_client = get_clients(api_key)
                            with st.spinner("Analizuję konstrukcje gramatyczne..."):
                                explanation = explain_translation_grammar(
                                    structured_client,
                                    row["source"],
                                    row["translation"],
                                    row["language"],
                                )
                            row["grammar_explanation"] = explanation.model_dump()
                        except Exception as exc:
                            st.error(f"Nie udało się wyjaśnić gramatyki: {exc}")
                    if row.get("grammar_explanation"):
                        explanation = row["grammar_explanation"]
                        st.markdown("**Wyjaśnienie gramatyczne**")
                        st.write(explanation["overview_pl"])
                        for construction in explanation["constructions"]:
                            st.markdown(f"**{construction['name']}** — *{construction['fragment']}*")
                            st.write(construction["explanation_pl"])
                        st.download_button(
                            "Pobierz wyjaśnienie gramatyczne",
                            data=translation_grammar_text(row["language"], row["source"], row["translation"], explanation),
                            file_name=f"gramatyka_{safe_filename_part(row['language'])}_{row_index + 1}.txt",
                            mime="text/plain; charset=utf-8",
                            key=f"download_translation_grammar_{row_index}",
                        )
        else:
            with st.container(border=True):
                st.write("Twoje tłumaczenia pojawią się tutaj.")
        st.download_button(
            "Pobierz całą pracę i tłumaczenia CSV",
            data=all_work_csv(st.session_state.translations, st.session_state.learning_packs, st.session_state.corrections),
            file_name="language_assistant.csv",
            mime="text/csv; charset=utf-8",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-kicker">Losowe słownictwo i gramatyka</div>', unsafe_allow_html=True)
    language = st.selectbox("Wybierz język do nauki:", languages, key="learning_language")
    if st.button("Wygeneruj losowe wyjaśnienia słów i gramatyki", key="generate_learning", type="primary"):
        try:
            _, structured_client = get_clients(api_key)
            with st.spinner("Przygotowuję słówka i wskazówki gramatyczne..."):
                pack = generate_learning_pack(structured_client, language)
            if len(pack.vocabulary) != 5:
                raise ValueError("Model nie zwrócił dokładnie 5 haseł. Spróbuj ponownie.")
            st.session_state.learning_packs.append({"language": language, "pack": pack})
            st.session_state.learning_pack = pack
            st.session_state.learning_language = language
        except Exception as exc:
            st.error(f"Nie udało się wygenerować materiału: {exc}")

    pack = st.session_state.get("learning_pack")
    if pack:
        pack_index = len(st.session_state.learning_packs) - 1
        st.subheader(f"Słówka: {st.session_state.learning_language}")
        words_col, grammar_col = st.columns(2, gap="large")
        with words_col:
            st.markdown("#### 5 słówek i przykłady użycia")
            for item in pack.vocabulary:
                with st.container(border=True):
                    st.markdown(f"**{item.word}** — {item.meaning_pl}")
                    st.markdown(f"*{item.example}*  \n{item.example_pl}")
                    render_audio_button(api_key, item.example, f"vocabulary_{pack_index}_{pack.vocabulary.index(item)}")
        with grammar_col:
            st.markdown("#### Wskazówki gramatyczne")
            for tip_index, tip in enumerate(pack.grammar_tips):
                with st.container(border=True):
                    st.markdown(f"**{tip.title}**")
                    st.write(tip.explanation_pl)
                    st.markdown(f"*{tip.example}*  \n{tip.example_pl}")
                    st.download_button(
                        "Pobierz wyjaśnienie",
                        data=grammar_tip_text(st.session_state.learning_language, tip),
                        file_name=f"gramatyka_{safe_filename_part(st.session_state.learning_language)}_{pack_index + 1}_{tip_index + 1}.txt",
                        mime="text/plain; charset=utf-8",
                        key=f"download_grammar_tip_{pack_index}_{tip_index}",
                    )
                    render_audio_button(api_key, tip.example, f"grammar_{pack_index}_{pack.grammar_tips.index(tip)}")


with tab_exercises:
    pass

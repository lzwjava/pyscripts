import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.llm.openrouter_client import call_openrouter_api
from scripts.translation.translate_utils import validate_translated_languages, detect_languages_with_langdetect

LANGUAGE_MAP = {
    "ja": "Japanese",
    "es": "Spanish",
    "hi": "Hindi",
    "fr": "French",
    "zh": "Simplified Chinese",
    "hant": "Traditional Chinese (Hong Kong)",
    "en": "English",
    "de": "German",
    "ar": "Arabic"
}

def build_prompt_template(target_language, type_, front_matter):
    lang_name = LANGUAGE_MAP.get(target_language, target_language)
    if type_ == "title":
        tpl = "Translate the following title into {lang}. Return only the translated title without any extra notes, explanations, or repetition of the input text. If the title is already in {lang}, return it as is. If the target language is English, ensure the title is in Title Case.\n"
    else:
        head = "Translate the following markdown text into {lang}. Return only the translated content without any additional notes or explanations. If the text is already in {lang}, return it unchanged.\n"
        if front_matter:
            head += f"{front_matter}\n"
        tpl = head
    return tpl.format(lang=lang_name)

def validate_length(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    length = len(text)
    if not 1 <= length <= 30000:
        raise ValueError(f"text length {length} outside allowed range [1, 30000]")

def clean_response(text):
    if not isinstance(text, str):
        raise RuntimeError("Model returned non-text response")
    return text.strip()

def check_echo(original, translated):
    if original.lower().strip() in translated.lower():
        raise RuntimeError("Model returned input or echoed text")

def check_commentary(translated):
    indicators = [
        "the output is", "output is", "translated title",
        "the translation is", "translation is", "translated:",
        "result:", "output:", "return only", "return only the translated"
    ]
    low = translated.lower()
    for ind in indicators:
        if ind in low:
            raise RuntimeError("Model included commentary")

def check_title_strict(title, target_lang):
    """Strict title validation - removes quotes and special chars based on language"""
    if "\n" in title.strip():
        raise RuntimeError("Model returned multi-line title")
    
    # Define forbidden characters by language
    forbidden_chars = {
        'ja': ['"', "'", '"', '"', '「', '」', '『', '』', '《', '》'],
        'zh': ['"', "'", '"', '"', '「', '」', '『', '』', '《', '》', '〈', '〉'],
        'hant': ['"', "'", '"', '"', '「', '」', '『', '』', '《', '》', '〈', '〉'],
        'hi': ['"', "'", '"', '"', '«', '»'],
        'ar': ['"', "'", '"', '"', '«', '»'],
        'es': ['"', "'", '"', '"', '«', '»'],
        'fr': ['"', "'", '"', '"', '«', '»'],
        'de': ['"', "'", '"', '"', '„', '"', '»', '«'],
        'en': ['"', "'", '"', '"']
    }
    
    # Get forbidden chars for target language, default to common quotes
    chars_to_remove = forbidden_chars.get(target_lang, ['"', "'", '"', '"'])
    
    # Check if title contains any forbidden characters
    for char in chars_to_remove:
        if char in title:
            raise RuntimeError(f"Title contains forbidden character: {char}")
    
    return title.strip()

def run_translate(text, target, kind, model, front_matter, orig_lang, need_en, source_file=None):
    validate_length(text)
    if target == orig_lang:
        return text

    prompt = build_prompt_template(target, kind, front_matter) + "\n\n" + text
    translated = clean_response(call_openrouter_api(prompt, model))
    check_echo(text, translated)
    check_commentary(translated)
    if kind == "title":
        check_title_strict(translated, target)

    try:
        detected = detect_languages_with_langdetect(translated)
    except Exception as e:
        detected = []
    validate_translated_languages(translated, target, require_english=need_en, source_file=source_file)
    return translated

def translate_text(text, target_language, type="content", model="deepseek-v3", front_matter_prompt=None, original_lang=None, front_matter=None, source_file=None):
    """Wrapper function for markdown_translate_client.py compatibility"""
    kind = type  # Map 'type' parameter to 'kind' parameter used in run_translate
    orig_lang = original_lang if original_lang else "en"
    return run_translate(
        text=text,
        target=target_language,
        kind=kind,
        model=model,
        front_matter=front_matter_prompt or front_matter,
        orig_lang=orig_lang,
        need_en=False,
        source_file=source_file
    )

def cli_translate():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--target", required=True)
    parser.add_argument("--type", default="content")
    parser.add_argument("--model", default="deepseek-v3")
    parser.add_argument("--front-matter")
    parser.add_argument("--original-lang")
    parser.add_argument("--require-english", action="store_true")
    args = parser.parse_args()

    translated = translate_text(
        text=args.text,
        target_language=args.target,
        type=args.type,
        model=args.model,
        front_matter_prompt=args.front_matter,
        original_lang=args.original_lang,
        front_matter=args.original_lang
    )
    print(translated)

if __name__ == "__main__":
    cli_translate()
# Demo runs
# python scripts/translation/translate_client.py "Hello world" --target zh --model mistral-medium --original-lang en
# python scripts/translation/translate_client.py "Nice Day" --target ja --type title

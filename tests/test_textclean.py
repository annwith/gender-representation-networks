from gender_networks.textclean import (
    clean_article,
    cut_sections,
    has_excluded_infobox,
    is_disambiguation,
    keep_paragraph,
    split_sentences,
    wikitext_to_text,
)

LONG = " ".join(["palavra"] * 45)


def lines(text: str) -> list[str]:
    return [line for line in text.split("\n") if line]


def test_detects_biography_infobox_and_disambiguation() -> None:
    bio = "{{Info/Biografia\n|nome = Fulano\n}}\nFulano foi um físico."
    assert has_excluded_infobox(bio, ["Info/Biografia", "Info/Futebolista"])
    assert not has_excluded_infobox("{{Info/Elemento químico}} Texto.", ["Info/Biografia"])
    assert is_disambiguation("{{Desambiguação}}\n'''Banco''' pode referir-se a:")
    assert not is_disambiguation("{{Referências}}")
    assert is_disambiguation("Texto.", title="Nota (desambiguação)")


def test_dablink_hatnote_is_not_disambiguation() -> None:
    text = "{{Dablink|Para outros significados, veja [[Nota (desambiguação)]].}}\nA nota é..."

    assert not is_disambiguation(text)


def test_cut_sections_removes_listed_sections() -> None:
    text = "Intro.\n== História ==\nCorpo.\n== Referências ==\nRef.\n== Ver também ==\nX"

    assert cut_sections(text, ["Referências", "Ver também"]) == "Intro.\n== História ==\nCorpo.\n"
    assert cut_sections(text, ["Bibliografia"]) == text


def test_cut_sections_keeps_later_sections_that_are_not_listed() -> None:
    text = (
        "Intro.\n== Notas ==\nA nota dó.\n=== Sub ===\nSub.\n== História ==\nProsa.\n"
        "=== Origem ===\nOrigem.\n== Referências ==\nRef.\n"
    )

    assert cut_sections(text, ["Notas", "Referências"]) == (
        "Intro.\n== História ==\nProsa.\n=== Origem ===\nOrigem.\n"
    )


def test_keep_paragraph_filters() -> None:
    assert keep_paragraph(LONG + ".", 40, 0.08)
    assert keep_paragraph(LONG + ".)", 40, 0.08)
    assert not keep_paragraph(LONG, 40, 0.08)  # does not end like a sentence
    assert not keep_paragraph("Curto demais.", 40, 0.08)
    numbers = " ".join(["1990 2000 3000"] * 20) + " texto."
    assert not keep_paragraph(numbers, 10, 0.08)


def test_split_sentences_respects_abbreviations_and_initials() -> None:
    text = (
        "O Sr. Silva estudou no séc. XIX a física. Depois, J. Smith chegou! "
        "Em 1990 veio a crise. «Citação» encerra."
    )

    spans = split_sentences(text)
    sentences = [text[s:e] for s, e in spans]

    assert sentences == [
        "O Sr. Silva estudou no séc. XIX a física.",
        "Depois, J. Smith chegou!",
        "Em 1990 veio a crise.",
        "«Citação» encerra.",
    ]


def test_split_sentences_after_unit_symbols() -> None:
    for text in (
        "O rio tem 300 km. Sua nascente fica na serra.",
        "A massa é de 5 kg. Outro valor foi medido.",
        "O comprimento é 3 m. Depois veio a medida.",
    ):
        assert len(split_sentences(text)) == 2, text
    assert len(split_sentences("Ver a p. 3 do livro.")) == 1


def test_clean_article_keeps_prose_and_drops_markup() -> None:
    body = (
        "'''Física''' é a [[ciência]] que estuda a natureza<ref>Fonte.</ref> e "
        '[[Ficheiro:Atom.png|thumb|Legenda da figura]]<ref name="x" /> a velocidade '
        "<math>v</math> "
        + LONG
        + ".\n\nA energia <math>E=mc^2</math> "
        + LONG
        + '.\n\n{| class="wikitable"\n| 1 || 2\n|}\n\n== Referências ==\n'
        + LONG
        + "."
    )

    paragraphs = clean_article(body, ["Referências"], 40, 0.08)

    assert len(paragraphs) == 1  # the paragraph with a non-trivial formula is dropped
    assert paragraphs[0].startswith("Física é a ciência que estuda a natureza e a velocidade v ")
    assert "Fonte" not in paragraphs[0]
    assert "Legenda" not in paragraphs[0] and "mc^2" not in paragraphs[0]


def test_tables_are_removed_entirely() -> None:
    body = (
        "Texto antes.\n"
        '{| class="wikitable"\n|+ Legenda\n! Nome !! Descrição\n|-\n'
        "| Próton || Partícula com carga positiva que forma o núcleo dos átomos.\n|-\n"
        "| Nêutron\n| Partícula sem carga, com efeitos.\n|}\n"
        "<table><tr><td>Célula HTML com texto.</td></tr></table>\n"
        "Texto depois."
    )

    text = wikitext_to_text(body)

    assert lines(text) == ["Texto antes.", "Texto depois."]


def test_line_breaks_do_not_glue_words() -> None:
    assert wikitext_to_text("Primeira linha<br>segunda linha<br />terceira.") == (
        "Primeira linha segunda linha terceira."
    )


def test_text_bearing_templates_keep_their_text_and_residue_is_tidied() -> None:
    cases = {
        "A cidade tem {{formatnum:12345}} habitantes e {{converter|10|km|mi}} de extensão.": (
            "A cidade tem 12345 habitantes e 10 km de extensão."
        ),
        "'''Física''' ({{lang-grc|φύσις}}; {{IPA|ˈfi.zi.kɐ}}) é uma ciência.": (
            "Física (φύσις) é uma ciência."
        ),
        "Um termo ({{IPA|x}}) pequeno <ref>x</ref>.": "Um termo pequeno.",
        "O termo {{lang|en|software}} designa {{nowrap|{{fmtn|1234}} km}} de código.": (
            "O termo software designa 1234 km de código."
        ),
        "Isto é verdade{{carece de fontes}}, e aquilo também.": (
            "Isto é verdade, e aquilo também."
        ),
    }
    for wikitext, expected in cases.items():
        assert wikitext_to_text(wikitext) == expected


def test_list_items_and_inline_chemistry_are_dropped() -> None:
    body = f"* Item de lista {LONG}.\n# Outro {LONG}.\nA água <chem>H2O</chem> {LONG}.\nFim."

    assert lines(wikitext_to_text(body)) == ["Fim."]

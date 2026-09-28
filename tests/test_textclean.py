from gender_networks.textclean import (
    clean_article,
    cut_sections,
    has_excluded_infobox,
    is_disambiguation,
    keep_paragraph,
    split_sentences,
)

LONG = " ".join(["palavra"] * 45)


def test_detects_biography_infobox_and_disambiguation() -> None:
    bio = "{{Info/Biografia\n|nome = Fulano\n}}\nFulano foi um físico."
    assert has_excluded_infobox(bio, ["Info/Biografia", "Info/Futebolista"])
    assert not has_excluded_infobox("{{Info/Elemento químico}} Texto.", ["Info/Biografia"])
    assert is_disambiguation("{{Desambiguação}}\n'''Banco''' pode referir-se a:")
    assert not is_disambiguation("{{Referências}}")


def test_cut_sections_stops_at_first_listed_heading() -> None:
    text = "Intro.\n== História ==\nCorpo.\n== Referências ==\nRef.\n== Ver também ==\nX"

    assert cut_sections(text, ["Referências", "Ver também"]) == "Intro.\n== História ==\nCorpo.\n"
    assert cut_sections(text, ["Bibliografia"]) == text


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


def test_clean_article_keeps_prose_and_drops_markup() -> None:
    body = (
        "'''Física''' é a [[ciência]] que estuda a natureza<ref>Fonte.</ref> e "
        "[[Ficheiro:Atom.png|thumb|Legenda da figura]]<ref name=\"x\" /> <math>E=mc^2</math> "
        + LONG
        + ".\n\n{| class=\"wikitable\"\n| 1 || 2\n|}\n\n== Referências ==\n"
        + LONG
        + "."
    )

    paragraphs = clean_article(body, ["Referências"], 40, 0.08)

    assert len(paragraphs) == 1
    assert paragraphs[0].startswith("Física é a ciência que estuda a natureza e")
    assert "Fonte" not in paragraphs[0]
    assert "Legenda" not in paragraphs[0] and "mc^2" not in paragraphs[0]

"""Clean up routines for unsed tag in WikiMarkup"""

import mwparserfromhell as psr


def remove_file_link(wikicode):
    for link in wikicode.ifilter_wikilinks():
        if 'File:' not in link.title:
            continue
        wikicode.replace(link, '')


def normalize_inter_lang_text(wikicode):
    for t in wikicode.ifilter_templates():
        if t.name != 'lang':
            continue
        _, text = t.params[0:2]
        text_value = text.value.strip_code()
        wikicode.replace(t, text_value)


def clean_up_ref_tag(wikicode):
    for tag in wikicode.ifilter_tags():
        if tag.tag != 'ref':
            continue
        wikicode.replace(tag, '')


def strip(src):
    """Returns stripped source text."""
    wikicode = psr.parse(src)

    # Keep a shallow copy of inline references for References section
    refs = [str(tag) for tag in wikicode.ifilter_tags() if tag.tag == 'ref']

    remove_file_link(wikicode)  # Remove all file link
    normalize_inter_lang_text(wikicode)  # Fix inter-lang templates
    clean_up_ref_tag(wikicode)  # Clean up ref tags in main article

    # Finally, strip unprintable code and clean up duplicated quotes.
    s = (wikicode
         .strip_code()          # Strip code using default mwpsr function
         .replace('""', '"')    # Remove duplicated quotes after
                                # stripped out everything else
         )

    # Allocate new object, for safe
    return str(s)

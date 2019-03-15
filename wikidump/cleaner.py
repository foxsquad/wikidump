"""Clean up routines for unsed tag in WikiMarkup"""

import mwparserfromhell as psr


def remove_file_link(wikicode):
    for link in wikicode.ifilter_wikilinks():
        if 'File:' not in link.title:
            continue

        # If link content not present in wikicode anymore (due to
        # previous duplicated replacement), a ValueError will be
        # raised. We should only replace if this link content only
        # if it is presented in this context.
        if link not in wikicode:
            continue
        wikicode.replace(link, '')


def normalize_inter_lang_text(wikicode):
    for t in wikicode.ifilter_templates():
        if t.name != 'lang':
            continue
        if t not in wikicode:
            continue

        # We only care about the second required param here
        text = t.params[1]
        text_value = text.value.strip_code()
        wikicode.replace(t, text_value)


def clean_up_ref_tag(wikicode):
    for tag in wikicode.ifilter_tags():
        if tag.tag != 'ref':
            continue

        # There is a weird situation that the Tag object presents in
        # wikicode, but deep_search method could not find it.
        # We must failback to string representation of tag object to
        # replace it in wikicode.
        if wikicode.contains(tag):
            wikicode.replace(tag, '')
        elif wikicode.contains(str(tag)):
            wikicode.replace(str(tag), '')


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

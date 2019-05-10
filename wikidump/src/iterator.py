import bz2

from lxml import etree

from wikidump import BZ_PATH, NSMAP


def strip_tag_name(elem):
    tag = etree.QName(elem)
    return tag.localname


def iter_read():
    with bz2.open(BZ_PATH, 'rb') as bz_file:
        for event, elm in etree.iterparse(source=bz_file):
            tname = strip_tag_name(elm)

            # skip condition, keep the rest of code block tidy
            if tname != 'page':
                continue

            # prune related element information when yield result
            # to parent context
            yield etree.tostring(elm)

            # it's safe to clear this element from here
            elm.clear()
            # eliminate now-empty references from the root node
            while elm.getprevious() is not None:
                del elm.getparent()[0]

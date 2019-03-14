"""Simple template expander moudle.

This module contains utilities function for expands some
well-known templates, i.e. citing, which is usually used
inside sources and references sections.

Known citation templates and alias(es):
  - sfn ; harvnb
  - sfnp
  - sfnm
  - cite book
  - cite journal
  - cite enclopedia
  - cite web
  - harv            ; Harvard citation
  - harvnb          ; Harvard citation no brackets
  - harvtxt         ; Harvard citation text
  - harvcol
  - harvcolnb
  - harvcoltxt
  - harvp
  - harvs           ; Harvard citations
  - LSJ
"""


def _get_named_params(template, param, default=None):
    if isinstance(param, str):
        if template.has(param):
            return template.get(param)
    elif isinstance(param, (list, tuple, set)):
        # Get first available named param:
        available = None
        for p in param:
            if template.has(p):
                available = p
                break
        if available is not None:
            return template.get(available)
    return default


def render_sfn(sfn):
    """Process sfn (shortened footnote) template into plain text.

    The reference list marked by this template usually apears in
    the {{reflist}} and generally references to other citation
    appear else where in the document.

    Ref:
        https://en.wikipedia.org/wiki/Template:Sfn
    """
    # TODO: Unifiy this function with other template processors.

    optional_idx = set()

    # optional param postscript
    ps = _get_named_params(sfn, {'ps', 'postscript'})
    if ps is None:
        ps = 'ps=.'
    else:
        optional_idx.add(sfn.params.index(ps))
        if 'none' in ps:
            ps = 'ps='

    # reference, usually not available
    ref = _get_named_params(sfn, {'ref', 'Ref'})

    p = _get_named_params(sfn, {'p', 'page'})  # single page
    pp = _get_named_params(sfn, {'pp', 'pages'})  # page range
    loc = _get_named_params(sfn, 'loc')  # source location

    optional_idx.update(
        set(sfn.params.index(i)
            for i in (ref, p, pp, loc)
            if i is not None)
    )

    authors = sfn.params[:min(optional_idx) - 1]
    pub_yr = sfn.params[min(optional_idx) - 1]

    # Up to 04 authors could be specified in the author list.
    # Each case is rendered a little different
    l = len(authors)
    if l == 4:
        text = '%s et al. %s' % (authors[0], pub_yr)
    elif l == 1:
        text = '%s %s' % (authors[0], pub_yr)
    else:
        last = authors[-1]
        remained = authors[:-1]
        text = '%s & %s %s' % (', '.join(remained), last, pub_yr)

    # p, pp and loc are mutual exclusive. Only one of this
    # should be rendered.
    if p is not None:
        text += ', p. %s' % p.value
    elif pp is not None:
        text += ', pp. %s' % pp.value
    elif loc is not None:
        text += ', %s' % loc.value

    # add postscript value
    text += ps.split('=')[1]

    return text

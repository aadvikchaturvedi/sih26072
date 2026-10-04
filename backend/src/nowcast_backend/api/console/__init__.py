"""The operator console's view of the backend.

The web console (``web/``) defines the response shapes it needs in
``web/src/data/types.ts``. ``schemas`` mirrors those shapes, ``mapping`` translates
the backend's domain models into them, and ``routes/console.py`` serves them.
Nothing here computes anything new about the weather.
"""

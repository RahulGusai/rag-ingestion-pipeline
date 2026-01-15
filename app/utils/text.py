import unicodedata
import re

_control_chars_re = re.compile(
    r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]'
)  # exclude \t(09), \n(0A), \r(0D)


def sanitize_text_for_db(s: str, max_len: int = None) -> str:
    if s is None:
        return ""

    # Normalize
    s = unicodedata.normalize("NFC", s)

    # Remove NULs and other problematic control chars (keeps newline/tab/carriage return)
    s = s.replace("\x00", "")
    s = _control_chars_re.sub("", s)

    # Force valid UTF-8: replace invalid bytes/sequences with replacement char
    # This also drops any lone surrogates.
    s = s.encode("utf-8", "replace").decode("utf-8", "replace")

    if max_len is not None and isinstance(max_len, int) and max_len > 0:
        if len(s) > max_len:
            s = s[:max_len]

    return s

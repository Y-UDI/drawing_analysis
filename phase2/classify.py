import re

from .models import DimensionValue, TextToken, TitleBlock


DIM_PATTERNS = [
    (r"[ØøφΦ⌀](\d+(?:\.\d+)?)", "diameter"),
    (r"[Rr](\d+(?:\.\d+)?)", "radius"),
    (r"(\d+(?:\.\d+)?)°", "angle"),
    (r"^(\d+(?:\.\d+)?)$", "linear"),
]


TITLE_KEYWORDS = {
    "material": ["material", "material:"],
    "mat_number": ["mat.no.", "mat no", "mat.no"],
    "part_name": ["part", "part name", "title,", "title"],
    "scale": ["scale", "scale:"],
    "tolerance": ["tolerance", "tolerances", "tolerances:", "iso 2768", "2768"],
    "creator": ["created by", "created by:", "drawn"],
    "approver": ["approved by", "approved by:"],
    "drawing_no": ["drawing number", "drawing number:", "dn"],
    "doc_status": ["document status", "in preparation", "released"],
    "issue_date": ["issue date", "date:"],
}


def is_in_titleblock(token: TextToken, page_width_mm: float, page_height_mm: float) -> bool:
    return token.y > page_height_mm * 0.70 and token.x > page_width_mm * 0.30


def classify_tokens(tokens: list, page_width_mm: float, page_height_mm: float) -> tuple:
    dimensions = []

    for index, token in enumerate(tokens):
        text = token.text.strip()
        matched = False

        for pattern, dim_type in DIM_PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if not match:
                continue
            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue
            if value <= 0:
                continue

            dimensions.append(DimensionValue(
                value=value,
                raw_text=text,
                dim_type=dim_type,
                x=token.x,
                y=token.y,
                token_idx=index,
            ))
            token.category = "dimension"
            matched = True
            break

        if matched:
            continue

        token.category = (
            "titleblock"
            if is_in_titleblock(token, page_width_mm, page_height_mm)
            else "note"
        )

    title_block = build_titleblock(tokens, page_width_mm, page_height_mm)
    return tokens, dimensions, title_block


def build_titleblock(tokens: list, page_width_mm: float, page_height_mm: float) -> TitleBlock:
    title_block = TitleBlock()
    tb_tokens = [
        (index, token)
        for index, token in enumerate(tokens)
        if is_in_titleblock(token, page_width_mm, page_height_mm)
    ]
    tb_tokens.sort(key=lambda item: (round(item[1].y, 1), item[1].x))

    def next_value(position: int, gap: int = 3) -> str:
        for next_index in range(position + 1, min(position + gap + 1, len(tb_tokens))):
            text = tb_tokens[next_index][1].text.strip()
            if text and not any(
                text.lower().startswith(keyword)
                for keywords in TITLE_KEYWORDS.values()
                for keyword in keywords
            ):
                return text
        return ""

    for position, (_, token) in enumerate(tb_tokens):
        text_lower = token.text.lower()

        if any(text_lower.startswith(keyword) for keyword in TITLE_KEYWORDS["material"]):
            value = next_value(position)
            title_block.material = value or title_block.material
        elif any(text_lower.startswith(keyword) for keyword in TITLE_KEYWORDS["mat_number"]):
            value = next_value(position)
            title_block.mat_number = value or title_block.mat_number
        elif any(text_lower.startswith(keyword) for keyword in TITLE_KEYWORDS["scale"]):
            match = re.search(r"(\d+:\d+)", token.text)
            title_block.scale = match.group(1) if match else (next_value(position) or title_block.scale)
        elif any(text_lower.startswith(keyword) for keyword in TITLE_KEYWORDS["tolerance"]):
            value = next_value(position)
            title_block.tolerance = (token.text + " " + value).strip()
        elif any(text_lower.startswith(keyword) for keyword in TITLE_KEYWORDS["creator"]):
            value = next_value(position)
            title_block.creator = value or title_block.creator
        elif any(text_lower.startswith(keyword) for keyword in TITLE_KEYWORDS["approver"]):
            value = next_value(position)
            title_block.approver = value or title_block.approver
        elif any(text_lower.startswith(keyword) for keyword in TITLE_KEYWORDS["drawing_no"]):
            value = next_value(position)
            title_block.drawing_no = value or title_block.drawing_no
        elif any(text_lower.startswith(keyword) for keyword in TITLE_KEYWORDS["doc_status"]):
            title_block.doc_status = token.text
        elif re.match(r"\d{4}[./]\d{2}[./]\d{2}", token.text):
            title_block.issue_date = token.text
        elif re.match(r"\d+\.\d{4}$", token.text):
            title_block.mat_number = token.text

    candidates = [
        (token.font_size, token.text)
        for _, token in tb_tokens
        if token.category == "titleblock"
        and len(token.text) > 2
        and not re.match(r"^[\d\./:]+$", token.text)
        and not any(
            token.text.lower().startswith(keyword)
            for keywords in TITLE_KEYWORDS.values()
            for keyword in keywords
        )
    ]
    if candidates:
        title_block.part_name = max(candidates, key=lambda item: item[0])[1]

    return title_block
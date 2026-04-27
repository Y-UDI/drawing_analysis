from dataclasses import dataclass, field


PT2MM = 0.3528


@dataclass
class TextToken:
    text: str
    x: float
    y: float
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    confidence: float
    source: str
    category: str = ""


@dataclass
class DimensionValue:
    value: float
    raw_text: str
    dim_type: str
    x: float
    y: float
    token_idx: int


@dataclass
class TitleBlock:
    part_name: str = ""
    material: str = ""
    mat_number: str = ""
    scale: str = ""
    tolerance: str = ""
    drawing_no: str = ""
    creator: str = ""
    approver: str = ""
    doc_status: str = ""
    issue_date: str = ""
    raw_fields: dict = field(default_factory=dict)


@dataclass
class OCRResult:
    tokens: list
    dimensions: list
    title_block: TitleBlock
    engine: str
    elapsed_sec: float
    input_path: str
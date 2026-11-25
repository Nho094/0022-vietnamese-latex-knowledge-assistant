"""
Module sinh câu huấn luyện tổng hợp (Synthetic Training Sentence Generator) cho các ý định (intents).

Tuân thủ nghiêm ngặt các quy tắc kiến trúc và tính trung thực trong AGENTS.md:
- Honesty Rule #1: Hệ thống kịch bản đơn giản hóa lấy cảm hứng từ Retrieve-and-Fill (RAF - arXiv:2202.00901),
  điền slot bằng biểu thức chính quy regex.
- Honesty Rule #3: Tuyệt đối không chỉnh sửa, tạo mới hay làm giàu thêm cho data/test.csv.
- Honesty Rule #4: Các phân vùng train/val/test không được chia sẻ bất kỳ câu trùng lặp nào
  (so khớp không phân biệt hoa thường và khoảng trắng).
- Thiết lập hạt giống tất định (deterministic reproducibility) qua hàm set_seed tập trung (core.config).
- Khử trùng lặp (deduplication) áp dụng độc quyền cho phân vùng huấn luyện (train split).
- Định nghĩa SENTENCE_FRAMES gồm 8-10 mẫu câu tiếng Việt cho mỗi intent với slot placeholder và khung câu thông dụng không chứa số.
- Điền slot số nguyên với các số nhỏ từ -9 đến 9 không có hệ số 0 đứng đầu (leading zero coefficients).
- Điền slot văn bản bằng các cụm từ tiếng Việt có hạt giống và slot tệp bằng tên tệp tiêu chuẩn.
- Biến thiên tiền tố câu hỏi (prompt starters) và chọn lọc chèn từ khóa 'latex'.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from core.config import (
    DEFAULT_INTEGER_MAX,
    DEFAULT_INTEGER_MIN,
    DEFAULT_INTENTS_PATH,
    DEFAULT_SEED,
    MAX_FRAMES_PER_INTENT,
    MIN_FRAMES_PER_INTENT,
    SYNTH_TRAIN_SAMPLES_PER_INTENT,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    TRAIN_METADATA_PATH,
    VAL_CSV_PATH,
    set_seed,
)

# Danh sách các tiền tố câu hỏi tiếng Việt đa dạng (Vietnamese Prompt Starters)
VIETNAMESE_PROMPT_STARTERS: Final[list[str]] = [
    "viết giúp tôi",
    "tạo giúp tôi",
    "cho tôi cú pháp",
    "hướng dẫn tôi tạo",
    "làm mẫu",
    "làm sao để tạo",
    "hãy tạo",
    "chèn vào tài liệu",
    "tôi muốn tạo",
    "cần đoạn mã",
    "cách viết",
    "giúp mình tạo",
    "xin cú pháp",
    "chỉ cho tôi cách viết",
]

# Danh sách các cụm từ tiếng Việt có hạt giống dùng để điền slot văn bản (Seeded Vietnamese Phrases)
SEEDED_VIETNAMESE_PHRASES: Final[list[str]] = [
    "Lý thuyết đồ thị và ứng dụng",
    "Giải tích toán học cao cấp",
    "Phương trình vi phân đạo hàm riêng",
    "Đại số tuyến tính và ma trận",
    "Cấu trúc dữ liệu và giải thuật",
    "Kiến trúc máy tính hiện đại",
    "Xử lý ngôn ngữ tự nhiên tiếng Việt",
    "Mạng nơ-ron và học sâu",
    "Hệ thống cơ sở dữ liệu phân tán",
    "Vật lý lượng tử ứng dụng",
    "Báo cáo tiến độ đề tài nghiên cứu",
    "Tài liệu hướng dẫn thực hành",
    "Giáo trình Toán rời rạc",
    "Khai phá dữ liệu lớn",
    "Nguyên lý hệ điều hành",
]

# Danh sách tên tệp tiêu chuẩn cho slot tệp hình ảnh (Standard Filenames)
STANDARD_FILENAMES: Final[list[str]] = [
    "hinh1.png",
    "figure.jpg",
    "bieu_do.pdf",
    "mo_hinh.png",
    "architecture.png",
    "data_plot.pdf",
    "so_do.png",
    "sample.jpg",
    "chart.pdf",
    "diagram.png",
]

# Danh sách nhãn tham chiếu tiêu chuẩn cho slot label (Standard Labels)
STANDARD_LABELS: Final[list[str]] = [
    "fig:hinh1",
    "fig:bieu_do",
    "sec:tong_quan",
    "eq:cong_thuc_1",
    "fig:so_do",
    "tab:bang1",
    "sec:phuong_phap",
    "eq:pt_bac_hai",
]

# Danh sách tên package LaTeX tiêu chuẩn cho slot package_name
STANDARD_PACKAGES: Final[list[str]] = [
    "amsmath",
    "amssymb",
    "graphicx",
    "geometry",
    "hyperref",
    "tikz",
    "booktabs",
    "microtype",
    "xcolor",
    "babel",
]

# Danh sách tùy chọn package tiêu chuẩn cho slot options
STANDARD_PACKAGE_OPTIONS: Final[list[str]] = [
    "utf8",
    "margin=1in",
    "vietnamese",
    "hidelinks",
    "dvipsnames",
    "a4paper",
]

# Danh sách tên tác giả mẫu cho slot author
STANDARD_AUTHORS: Final[list[str]] = [
    "Nguyễn Văn An",
    "Trần Thị Bình",
    "Lê Văn Cường",
    "Phạm Minh Đức",
    "Hoàng Gia Bảo",
]

# Danh sách tiêu đề bài viết cho slot title
STANDARD_TITLES: Final[list[str]] = [
    "Nghiên cứu khoa học sinh viên",
    "Báo cáo đề tài tốt nghiệp",
    "Giáo trình Toán cao cấp",
    "Tài liệu hướng dẫn LaTeX",
    "Tổng quan về Trí tuệ Nhân tạo",
]

# Danh sách lời cảm ơn cho slot thanks
STANDARD_THANKS: Final[list[str]] = [
    "Nghiên cứu được tài trợ bởi quỹ khoa học",
    "Cảm ơn sự hỗ trợ của phòng thí nghiệm AI",
    "Thành viên nhóm nghiên cứu xử lý ngôn ngữ",
]

# Danh sách chú thích cho slot caption
STANDARD_CAPTIONS: Final[list[str]] = [
    "Biểu đồ phân bố dữ liệu thực nghiệm",
    "Sơ đồ kiến trúc hệ thống xử lý",
    "Kết quả mô phỏng trên tập kiểm thử",
    "Hình ảnh minh họa cấu trúc mô hình",
]

# Danh sách bề rộng cho slot width
STANDARD_WIDTHS: Final[list[str]] = [
    "0.5",
    "0.6",
    "0.7",
    "0.75",
    "0.8",
    "0.85",
    "0.9",
    "1.0",
]

# Danh sách URL cho slot url
STANDARD_URLS: Final[list[str]] = [
    "https://ctan.org",
    "https://overleaf.com",
    "https://example.com/tailieu",
    "https://vi.wikipedia.org/wiki/LaTeX",
]

# Danh sách ngôn ngữ lập trình cho slot language
STANDARD_LANGUAGES: Final[list[str]] = [
    "python",
    "cpp",
    "latex",
    "bash",
    "java",
]

# Danh sách đoạn mã cho slot code
STANDARD_CODE_SNIPPETS: Final[list[str]] = [
    "print('Hello, LaTeX!')",
    "int x = 10;",
    "SELECT * FROM users;",
    "\\documentclass{article}",
]

# Danh sách ma trận mẫu cho slot matrix / matrix_content
STANDARD_MATRICES: Final[list[str]] = [
    "1 2; 3 4",
    "a b; c d",
    "1 0; 0 1",
    "2 -1; 4 3",
    "x y; z w",
]

# Danh sách biểu thức toán tiêu chuẩn cho slot kiểu math
STANDARD_MATH_EXPRS: Final[list[str]] = [
    "x+1",
    "a^2+b^2",
    "2x-3",
    "x^2",
    "n+1",
    "f(x)",
    "sin(x)",
    "\\alpha + \\beta",
]

# Nhãn tiếng Việt tự nhiên cho từng tên slot khi ghép khung tự động
SLOT_PROMPT_LABELS: Final[dict[str, str]] = {
    "text": "nội dung",
    "expr": "biểu thức",
    "matrix_content": "nội dung ma trận",
    "condition": "điều kiện",
    "num": "tử số",
    "den": "mẫu số",
    "base": "cơ số",
    "exp": "số mũ",
    "sub": "chỉ số dưới",
    "from": "cận dưới",
    "to": "cận trên",
    "caption": "chú thích",
    "label": "nhãn",
    "filename": "tệp",
    "file": "tệp",
    "width": "độ rộng",
    "package_name": "gói",
    "options": "tùy chọn",
    "title": "tiêu đề",
    "author": "tác giả",
    "thanks": "lời cảm ơn",
}

# Khung mẫu được thiết kế riêng thủ công cho các ý định quan trọng chứa slot đặc thù
# Đảm bảo mỗi intent có 8-10 câu mẫu, kết hợp giữa slot placeholders và casual number-free frames
HANDCRAFTED_INTENT_FRAMES: Final[dict[str, list[str]]] = {
    "QUADRATIC_EQUATION": [
        "{starter} phương trình bậc hai {a}x^2 + {b}x + {c} = 0",
        "{starter} giải phương trình bậc hai {a}x^2 + {b}x + {c} = 0 trong LaTeX",
        "{starter} công thức bậc hai {a}x^2 + {b}x + {c} = 0",
        "{starter} phương trình {a}x^2 + {b}x + {c} = 0",
        "{starter} phương trình bậc hai {a}x^2 + {b}x + {c} = 0 bằng LaTeX",
        "{starter} chèn phương trình bậc 2 {a}x^2 + {b}x + {c} = 0 vào tài liệu",
        "{starter} phương trình bậc hai dạng chuẩn",
        "{starter} cú pháp phương trình bậc 2 trong LaTeX",
        "{starter} hướng dẫn tạo phương trình bậc hai",
        "{starter} làm mẫu phương trình bậc hai tổng quát bằng LaTeX",
    ],
    "LATEX_NAP_PACKAGE": [
        "{starter} nạp gói {package_name} trong LaTeX",
        "{starter} sử dụng package {package_name} trong phần preamble",
        "{starter} khai báo gói {package_name} cho tài liệu",
        "{starter} thêm package {package_name} vào văn bản",
        "{starter} hướng dẫn dùng package {package_name} bằng LaTeX",
        "{starter} cú pháp nạp package {package_name}",
        "{starter} nạp gói package cho tài liệu LaTeX",
        "{starter} cách nạp thêm gói vào preamble",
        "{starter} khai báo sử dụng package trong LaTeX",
        "{starter} làm mẫu nạp package mới",
    ],
    "LATEX_NAP_PACKAGE_CO_TUY_CHON": [
        "{starter} nạp gói {package_name} với tùy chọn {options}",
        "{starter} khai báo package {package_name} có tùy chọn {options} trong LaTeX",
        "{starter} sử dụng gói {package_name} kèm options {options}",
        "{starter} thêm package {package_name} tùy chọn {options} vào tài liệu",
        "{starter} cú pháp nạp gói {package_name} có tham số {options}",
        "{starter} hướng dẫn nạp package {package_name} tùy chọn {options} bằng LaTeX",
        "{starter} nạp package có tùy chọn trong preamble",
        "{starter} cách nạp gói LaTeX với tham số tùy chọn",
        "{starter} khai báo gói có tùy chọn trong LaTeX",
        "{starter} làm mẫu lệnh usepackage có ngoặc vuông",
    ],
    "LATEX_CHEN_HINH_ANH": [
        "{starter} chèn hình ảnh {filename} với độ rộng {width}",
        "{starter} thêm tệp ảnh {filename} độ rộng {width} trong LaTeX",
        "{starter} chèn ảnh {filename} vào bài viết",
        "{starter} đưa hình {filename} vào văn bản",
        "{starter} cú pháp chèn ảnh {filename} rộng {width} trong LaTeX",
        "{starter} chèn file hình ảnh {filename}",
        "{starter} chèn hình ảnh vào văn bản LaTeX",
        "{starter} cách đưa tệp hình ảnh vào tài liệu",
        "{starter} hướng dẫn chèn ảnh trong LaTeX",
        "{starter} tạo lệnh chèn hình ảnh bằng LaTeX",
    ],
    "LATEX_FIGURE": [
        "{starter} tạo figure cho ảnh {file} chú thích {caption} nhãn {label} rộng {width}",
        "{starter} chèn hình {file} có nhãn {label} và caption {caption} trong LaTeX",
        "{starter} môi trường figure với tệp {file} và chú thích {caption}",
        "{starter} tạo môi trường figure ảnh {file} nhãn {label}",
        "{starter} thêm figure cho ảnh {file} rộng {width} trong LaTeX",
        "{starter} chèn ảnh {file} kèm chú thích {caption}",
        "{starter} tạo môi trường figure hoàn chỉnh trong LaTeX",
        "{starter} chèn hình figure có caption và nhãn tham chiếu",
        "{starter} hướng dẫn tạo khối figure chứa hình ảnh",
        "{starter} cú pháp môi trường figure chuẩn bằng LaTeX",
    ],
    "LATEX_THAM_CHIEU_NHAN": [
        "{starter} tham chiếu đến nhãn {label}",
        "{starter} tạo tham chiếu nhãn {label} trong LaTeX",
        "{starter} liên kết tới nhãn {label}",
        "{starter} chèn lệnh ref cho nhãn {label}",
        "{starter} trỏ đến nhãn {label} bằng LaTeX",
        "{starter} tham chiếu nhãn trong tài liệu LaTeX",
        "{starter} cách dùng lệnh ref để liên kết chéo",
        "{starter} hướng dẫn tạo tham chiếu chéo",
        "{starter} cú pháp tham chiếu đến nhãn bằng LaTeX",
        "{starter} làm mẫu tham chiếu nhãn",
    ],
    "LATEX_TIEU_E_TAC_GIA_NGAY_THANG": [
        "{starter} khai báo tiêu đề {title} tác giả {author}",
        "{starter} tạo phần tiêu đề {title} và tác giả {author} trong LaTeX",
        "{starter} thiết lập title {title} author {author}",
        "{starter} khai báo bài viết {title} bởi {author} bằng LaTeX",
        "{starter} chèn tiêu đề {title} cùng tác giả {author}",
        "{starter} tạo khối tiêu đề tác giả và ngày tháng",
        "{starter} cú pháp maketitle trong preamble LaTeX",
        "{starter} hướng dẫn hiển thị tiêu đề và tác giả",
        "{starter} cách tạo tiêu đề bài viết bằng LaTeX",
        "{starter} làm mẫu phần thông tin đầu tài liệu",
    ],
    "LATEX_LOI_CAM_ON_TAC_GIA": [
        "{starter} thêm lời cảm ơn {thanks} cho tác giả {author}",
        "{starter} tạo ghi chú cảm ơn {thanks} của {author} trong LaTeX",
        "{starter} lệnh thanks {thanks} dưới tên {author}",
        "{starter} khai báo author {author} kèm thanks {thanks}",
        "{starter} chèn lời cảm ơn tác giả {author} là {thanks} bằng LaTeX",
        "{starter} thêm lời cảm ơn tác giả bằng lệnh thanks",
        "{starter} cú pháp cảm ơn tác giả trong LaTeX",
        "{starter} cách viết lời cảm ơn trong author",
        "{starter} hướng dẫn chèn ghi chú cảm ơn bằng LaTeX",
        "{starter} làm mẫu ghi chú chân trang cho tác giả",
    ],
    "LATEX_CHU_THICH_HINH": [
        "{starter} thêm chú thích {caption} cho hình ảnh",
        "{starter} tạo caption {caption} trong LaTeX",
        "{starter} chèn dòng chú thích {caption} dưới hình",
        "{starter} đặt chú thích hình ảnh là {caption} bằng LaTeX",
        "{starter} viết caption {caption} cho ảnh",
        "{starter} tạo dòng chú thích hình ảnh",
        "{starter} cú pháp caption cho hình trong LaTeX",
        "{starter} cách viết chú thích dưới ảnh",
        "{starter} hướng dẫn chèn caption bằng LaTeX",
        "{starter} làm mẫu đặt chú thích hình",
    ],
    "MATH_MATH_BASIC_PHAN_SO": [
        "{starter} phân số tử số là {num} mẫu là {den}",
        "{starter} phân số tử số {num} và mẫu số {den}",
        "{starter} phân số {num} trên {den} trong LaTeX",
        "{starter} biểu diễn phân số {num} phần {den} với MathJax",
        "{starter} công thức phân số {num}/{den}",
        "{starter} phân số có tử {num} mẫu {den}",
        "{starter} phân số toán học",
        "{starter} cách tạo phân số trong LaTeX",
        "{starter} cú pháp biểu diễn phân số với MathJax",
        "{starter} công thức phân số tổng quát bằng frac",
    ],
    "MATH_MATH_BASIC_CAN_BAC_N": [
        "{starter} căn bậc {n} của biểu thức {expr}",
        "{starter} tính căn bậc {n} của {expr} trong LaTeX",
        "{starter} biểu diễn căn bậc {n} của {expr} với MathJax",
        "{starter} công thức căn bậc {n} cho {expr}",
        "{starter} chèn căn bậc {n} của {expr}",
        "{starter} viết căn bậc n tổng quát trong LaTeX",
        "{starter} cách tạo căn bậc n",
        "{starter} cú pháp căn thức bậc n với MathJax",
        "{starter} hướng dẫn biểu diễn căn bậc n",
        "{starter} công thức căn bậc n trong toán học",
    ],
    "MATH_MATRIX_MA_TRAN_2X2": [
        "{starter} ma trận 2x2 gồm {a}, {b}, {c}, {d}",
        "{starter} tạo ma trận 2 nhân 2 hàng 1 là {a} {b} hàng 2 là {c} {d}",
        "{starter} viết ma trận vuông 2x2 với a là {a}, b là {b}, c là {c}, d là {d}",
        "{starter} biểu diễn ma trận hai hàng hai cột gồm {a}, {b}, {c}, {d}",
        "{starter} ma trận 2x2 a={a} b={b} c={c} d={d} với MathJax",
        "{starter} ma trận hai hàng hai cột trong LaTeX",
        "{starter} cú pháp viết ma trận 2x2 với MathJax",
        "{starter} cách viết ma trận vuông bậc hai",
        "{starter} mẫu ma trận hai nhân hai chuẩn",
        "{starter} hướng dẫn tạo ma trận cỡ hai nhân hai",
    ],
    "MATH_MATRIX_CHUAN_MA_TRAN": [
        "{starter} chuẩn ma trận {matrix_content}",
        "{starter} tạo chuẩn ma trận Vmatrix nội dung là {matrix_content}",
        "{starter} viết chuẩn ma trận 2x2 gồm {matrix_content}",
        "{starter} biểu diễn chuẩn của ma trận với các hàng {matrix_content}",
        "{starter} chuẩn ma trận với MathJax nội dung ma trận là {matrix_content}",
        "{starter} ký hiệu chuẩn ma trận hai gạch đứng Vmatrix",
        "{starter} cách viết chuẩn ma trận với MathJax",
        "{starter} cú pháp môi trường Vmatrix cho chuẩn ma trận",
        "{starter} hướng dẫn viết chuẩn ma trận trong LaTeX",
        "{starter} tạo định mức chuẩn ma trận hai sọc đứng",
    ],
    "MATH_MATRIX_MA_TRAN": [
        "{starter} ma trận nội dung là {matrix_content}",
        "{starter} tạo ma trận trơn gồm {matrix_content} với MathJax",
        "{starter} biểu diễn ma trận có các hàng {matrix_content} trong LaTeX",
        "{starter} viết môi trường matrix nội dung ma trận là {matrix_content}",
        "{starter} tạo ma trận trơn trong toán học",
        "{starter} cú pháp môi trường matrix cơ bản",
        "{starter} cách viết ma trận nhiều hàng với MathJax",
        "{starter} hướng dẫn tạo ma trận không dấu ngoặc",
    ],
    "MATH_MATRIX_MA_TRAN_NGOAC_TRON": [
        "{starter} ma trận ngoặc tròn nội dung là {matrix_content}",
        "{starter} tạo ma trận pmatrix gồm {matrix_content}",
        "{starter} viết ma trận đóng ngoặc tròn các hàng {matrix_content} với MathJax",
        "{starter} ma trận dấu ngoặc đơn pmatrix trong LaTeX",
        "{starter} cách viết ma trận ngoặc tròn trong toán học",
        "{starter} cú pháp môi trường pmatrix với MathJax",
        "{starter} hướng dẫn tạo ma trận ngoặc tròn",
        "{starter} mẫu ma trận pmatrix chuẩn",
    ],
    "MATH_MATRIX_MA_TRAN_NGOAC_VUONG": [
        "{starter} ma trận ngoặc vuông nội dung là {matrix_content}",
        "{starter} tạo ma trận bmatrix gồm {matrix_content}",
        "{starter} viết ma trận đóng ngoặc vuông các hàng {matrix_content} với MathJax",
        "{starter} ma trận dấu ngoặc vuông bmatrix trong LaTeX",
        "{starter} cách viết ma trận ngoặc vuông trong toán học",
        "{starter} cú pháp môi trường bmatrix với MathJax",
        "{starter} hướng dẫn tạo ma trận ngoặc vuông",
        "{starter} mẫu ma trận bmatrix chuẩn",
    ],
    "MATH_MATRIX_INH_THUC_MA_TRAN": [
        "{starter} định thức ma trận nội dung là {matrix_content}",
        "{starter} tạo định thức ma trận vmatrix gồm {matrix_content}",
        "{starter} viết định thức dấu trị tuyệt đối các hàng {matrix_content} với MathJax",
        "{starter} ký hiệu định thức ma trận một gạch đứng vmatrix",
        "{starter} cách viết định thức ma trận trong LaTeX",
        "{starter} cú pháp môi trường vmatrix với MathJax",
        "{starter} hướng dẫn biểu diễn định thức của ma trận",
        "{starter} mẫu định thức ma trận det chuẩn",
    ],
    "OUT_OF_SCOPE": [
        "hôm nay thời tiết thế nào",
        "bây giờ là mấy giờ rồi bạn ơi",
        "gợi ý quán ăn trưa gần đây",
        "đặt báo thức lúc sáu giờ ba mươi sáng mai",
        "giá vàng hôm nay bao nhiêu",
        "lịch thi đấu bóng đá tối nay",
        "chỉ đường đi sân bay Nội Bài",
        "bạn tên là gì và do nhóm nào phát triển",
    ],
}


def normalize_sentence(sentence: str) -> str:
    """
    Chuẩn hóa câu văn để so khớp không phân biệt chữ hoa thường và khoảng trắng thừa.
    Tuân thủ Honesty Rule #4 (AGENTS.md): Train/val/test must not share duplicate sentences.

    Tham số:
        sentence: Chuỗi câu văn cần chuẩn hóa.

    Trả về:
        Chuỗi câu văn viết thường với khoảng trắng đã được thu gọn thành một ký tự đơn.
    """
    cleaned = re.sub(r"\s+", " ", sentence.strip().lower())
    return cleaned


def extract_action_name(describe: str, aliases: Sequence[str]) -> str:
    """
    Trích xuất tên hành động hoặc tên tác vụ ngắn gọn từ trường describe hoặc danh sách aliases.
    Tương thích với hàm extract_name_part theo kiến trúc Retrieve-and-Fill (RAF).

    Tham số:
        describe: Chuỗi mô tả kịch bản trong intents.json.
        aliases: Danh sách các tên gọi khác (aliases) của intent.

    Trả về:
        Tên hành động ngắn gọn bằng tiếng Việt.
    """
    desc = describe.strip()
    if "|" in desc and "dấu |" not in desc:
        parts = desc.split("|", 1)
        name = parts[0].strip()
        if name:
            return name

    # Nếu có alias ngắn gọn, ưu tiên dùng alias đầu tiên
    if aliases and len(aliases) > 0:
        alias_candidate = aliases[0].strip()
        if 2 <= len(alias_candidate) <= 35:
            return alias_candidate

    # Cắt ngắn mô tả nếu kết thúc bằng dấu chấm
    if desc.endswith("."):
        desc = desc[:-1].strip()
    return desc


def build_frames_for_intent(intent_info: dict[str, Any]) -> list[str]:
    """
    Tạo danh sách 8-10 mẫu khung câu (frames) cho một intent cụ thể.
    Kết hợp giữa các khung chứa slot placeholder và khung câu thông dụng (casual number-free frames).

    Tham số:
        intent_info: Bản ghi thông tin intent từ intents.json.

    Trả về:
        Danh sách chứa đúng từ 8 đến 10 chuỗi mẫu khung câu.
    """
    intent_name = intent_info.get("intent", "")

    # 1. Nếu intent đã được định nghĩa khung thủ công chi tiết
    if intent_name in HANDCRAFTED_INTENT_FRAMES:
        frames = list(HANDCRAFTED_INTENT_FRAMES[intent_name])
        # Đảm bảo số lượng nằm trong ngưỡng cấu hình
        if len(frames) > MAX_FRAMES_PER_INTENT:
            return frames[:MAX_FRAMES_PER_INTENT]
        if len(frames) >= MIN_FRAMES_PER_INTENT:
            return frames

    # 2. Xây dựng khung tự động dựa trên thông tin intent, domain và slots
    describe = intent_info.get("describe", "")
    aliases = intent_info.get("aliases", [])
    slots = intent_info.get("slots", {})
    category = intent_info.get("category", "tex_misc")
    action = extract_action_name(describe, aliases)

    if category == "obsidian":
        ctx_in = "trong Obsidian"
        ctx_with = "theo cú pháp Markdown"
        ctx_place = "vào ghi chú Obsidian"
    elif category == "math_shared":
        ctx_in = "trong toán học"
        ctx_with = "với MathJax"
        ctx_place = "trong LaTeX"
    else:
        ctx_in = "trong LaTeX"
        ctx_with = "bằng LaTeX"
        ctx_place = "vào tài liệu LaTeX"

    frames: list[str] = []

    if slots:
        slot_names = list(slots.keys())
        slot_parts = [f"{SLOT_PROMPT_LABELS.get(s, s)} là {{{s}}}" for s in slot_names]
        # Khung 1-2: chứa nhãn slot tiếng Việt tự nhiên + từ khóa đúng domain
        frames.append(f"{{starter}} {action} với {', '.join(slot_parts)} {ctx_in}")
        frames.append(f"{{starter}} {action} có {', '.join(slot_parts)}")

        # Khung 3-5: lồng slot hợp lý với nhãn ngắn hoặc trực tiếp
        direct_slots = " ".join([f"{SLOT_PROMPT_LABELS.get(s, s)} {{{s}}}" for s in slot_names])
        frames.append(f"{{starter}} {action} {direct_slots} {ctx_with}")
        frames.append(f"{{starter}} cú pháp {action} cho {', '.join(slot_parts)}")
        frames.append(f"{{starter}} chèn {action} với {', '.join(slot_parts)} {ctx_place}")

        # Khung 6-10: câu tự nhiên không chứa số hay slot placeholder
        frames.append(f"{{starter}} {action}")
        frames.append(f"{{starter}} {action} {ctx_in}")
        frames.append(f"{{starter}} cú pháp {action} {ctx_with}")
        frames.append(f"{{starter}} hướng dẫn tạo {action} {ctx_in}")
        frames.append(f"{{starter}} làm mẫu {action} chuẩn")
    else:
        # Đối với intent không có slots: tạo 10 biến thể đúng domain
        frames.append(f"{{starter}} {action}")
        frames.append(f"{{starter}} {action} {ctx_in}")
        frames.append(f"{{starter}} cú pháp {action}")
        frames.append(f"{{starter}} hướng dẫn tạo {action} {ctx_with}")
        frames.append(f"{{starter}} cách viết {action} {ctx_in}")
        frames.append(f"{{starter}} biểu diễn {action} {ctx_with}")
        frames.append(f"{{starter}} tạo nhanh {action}")
        frames.append(f"{{starter}} làm mẫu {action} {ctx_in}")
        frames.append(f"{{starter}} chèn {action} {ctx_place}")
        frames.append(f"{{starter}} cấu trúc {action} chuẩn")

    # Giới hạn số khung theo đúng ngưỡng cấu hình (8-10 mẫu)
    return frames[:MAX_FRAMES_PER_INTENT]


def build_sentence_frames(intents_path: Path = DEFAULT_INTENTS_PATH) -> dict[str, list[str]]:
    """
    Khởi tạo từ điển toàn bộ các khung câu (SENTENCE_FRAMES) cho mọi ý định trong intents.json.
    Mỗi intent đảm bảo có từ 8 đến 10 mẫu câu tiếng Việt.

    Tham số:
        intents_path: Đường dẫn tệp intents.json (mặc định lấy từ core.config).

    Trả về:
        Từ điển ánh xạ từ tên intent sang danh sách 8-10 khung câu mẫu.
    """
    if not intents_path.exists():
        # Trả về ít nhất các khung thủ công nếu tệp chưa tồn tại
        return dict(HANDCRAFTED_INTENT_FRAMES)

    with open(intents_path, encoding="utf-8") as f:
        intents_data = json.load(f)

    frames_dict: dict[str, list[str]] = {}
    for item in intents_data:
        intent_name = item.get("intent")
        if not intent_name:
            continue
        frames = build_frames_for_intent(item)
        frames_dict[intent_name] = frames

    return frames_dict


# Hằng số toàn cục SENTENCE_FRAMES theo yêu cầu bài toán
SENTENCE_FRAMES: Final[dict[str, list[str]]] = build_sentence_frames()


def generate_slot_value(slot_name: str, slot_def: dict[str, Any], rng: random.Random) -> str:
    """
    Sinh giá trị ngẫu nhiên có kiểm soát cho một slot dựa vào kiểu và tên của slot đó.
    - Slot số nguyên: sinh số ngẫu nhiên từ -9 đến 9, không có hệ số 0 đứng đầu.
    - Slot văn bản: chọn từ danh sách các cụm từ tiếng Việt có hạt giống.
    - Slot tệp: chọn từ danh sách tên tệp tiêu chuẩn.
    - Slot nhãn: chọn từ danh sách nhãn tiêu chuẩn.

    Tham số:
        slot_name: Tên tham số (slot).
        slot_def: Định nghĩa cấu trúc slot từ intents.json.
        rng: Bộ sinh số ngẫu nhiên độc lập (random.Random).

    Trả về:
        Chuỗi giá trị đã điền cho slot.
    """
    slot_type = slot_def.get("type", "text")

    # 1. Slot số nguyên / hệ số số học
    if slot_type == "number" or slot_name in ("a", "b", "c", "d", "n"):
        if slot_name == "a":
            # Hệ số bậc hai trong phương trình bậc hai hoặc hệ số đầu không được bằng 0
            non_zero_choices = [
                i for i in range(DEFAULT_INTEGER_MIN, DEFAULT_INTEGER_MAX + 1) if i != 0
            ]
            val = rng.choice(non_zero_choices)
            return str(val)
        if slot_name == "n":
            # Bậc của căn thức (căn bậc 2 đến căn bậc 9)
            val = rng.randint(2, DEFAULT_INTEGER_MAX)
            return str(val)
        if slot_name in ("b", "c", "d"):
            val = rng.randint(DEFAULT_INTEGER_MIN, DEFAULT_INTEGER_MAX)
            return str(val)
        if slot_name == "width":
            return rng.choice(STANDARD_WIDTHS)

        # Số nguyên chung từ -9 đến 9 không có số 0 dẫn đầu
        choices = [i for i in range(DEFAULT_INTEGER_MIN, DEFAULT_INTEGER_MAX + 1) if i != 0]
        return str(rng.choice(choices))

    # 1b. Slot ma trận
    if slot_type == "matrix" or slot_name == "matrix_content":
        return rng.choice(STANDARD_MATRICES)

    # 1c. Slot biểu thức toán
    if slot_type == "math":
        if slot_name == "condition":
            return rng.choice(["x -> 0", "n -> \\infty", "x -> 1", "x -> a"])
        if slot_name == "from":
            return rng.choice(["i=1", "k=0", "n=1", "k=1"])
        if slot_name == "to":
            return rng.choice(["n", "N", "\\infty", "10"])
        if slot_name in ("num", "den", "exp", "sub"):
            return rng.choice(["2", "3", "5", "x", "x+1", "n", "k"])
        return rng.choice(STANDARD_MATH_EXPRS)

    # 2. Slot tệp tin (File slots: standard filenames)
    if slot_type == "filename" or slot_name in ("filename", "file"):
        return rng.choice(STANDARD_FILENAMES)

    # 3. Slot nhãn tham chiếu (Label slots: standard labels)
    if slot_type == "label" or slot_name in ("label", "id", "tag"):
        return rng.choice(STANDARD_LABELS)

    # 4. Slot văn bản chuyên biệt
    if slot_name == "package_name":
        return rng.choice(STANDARD_PACKAGES)
    if slot_name == "options":
        return rng.choice(STANDARD_PACKAGE_OPTIONS)
    if slot_name == "author":
        return rng.choice(STANDARD_AUTHORS)
    if slot_name == "title":
        return rng.choice(STANDARD_TITLES)
    if slot_name == "thanks":
        return rng.choice(STANDARD_THANKS)
    if slot_name == "caption":
        return rng.choice(STANDARD_CAPTIONS)
    if slot_name == "width":
        return rng.choice(STANDARD_WIDTHS)
    if slot_name in ("url", "link"):
        return rng.choice(STANDARD_URLS)
    if slot_name in ("language", "lang"):
        return rng.choice(STANDARD_LANGUAGES)
    if slot_name == "code":
        return rng.choice(STANDARD_CODE_SNIPPETS)

    # 5. Slot văn bản chung: Chọn từ danh sách cụm từ tiếng Việt có hạt giống
    return rng.choice(SEEDED_VIETNAMESE_PHRASES)


def populate_frame(
    frame: str,
    slots_def: dict[str, Any],
    rng: random.Random,
    starters: Sequence[str] = VIETNAMESE_PROMPT_STARTERS,
) -> str:
    """
    Điền các slot placeholders và tiền tố prompt vào khung mẫu câu.

    Tham số:
        frame: Chuỗi khung câu chứa các placeholder dạng {starter}, {slot_name}.
        slots_def: Định nghĩa các slots của intent.
        rng: Bộ sinh số ngẫu nhiên độc lập.
        starters: Danh sách các tiền tố câu hỏi tiếng Việt.

    Trả về:
        Câu văn hoàn chỉnh đã điền giá trị.
    """
    # Điền tiền tố starter nếu có
    starter = rng.choice(starters)
    rendered = frame.replace("{starter}", starter)

    # Tìm các placeholder dạng {slot_name}
    placeholders = re.findall(r"\{([A-Za-z0-9_]+)\}", rendered)
    for p in placeholders:
        slot_def = slots_def.get(p, {})
        val = generate_slot_value(p, slot_def, rng)
        rendered = rendered.replace(f"{{{p}}}", val, 1)

    # Chuẩn hóa khoảng trắng
    rendered = re.sub(r"\s+", " ", rendered).strip()
    return rendered


def load_existing_sentences(*csv_paths: Path) -> set[str]:
    """
    Đọc và chuẩn hóa toàn bộ các câu trong các tệp phân vùng đã có (ví dụ val.csv, test.csv).
    Phục vụ kiểm tra chống trùng lặp theo Honesty Rule #4 (AGENTS.md).

    Tham số:
        csv_paths: Danh sách các đường dẫn tệp CSV.

    Trả về:
        Tập hợp (set) các câu văn đã được chuẩn hóa chữ thường và thu gọn khoảng trắng.
    """
    sentences: set[str] = set()
    for path in csv_paths:
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", "")
                    if text:
                        sentences.add(normalize_sentence(text))
        except (OSError, csv.Error):
            pass
    return sentences


def generate_synthetic_sentences(
    intents_path: Path = DEFAULT_INTENTS_PATH,
    seed: int = DEFAULT_SEED,
    samples_per_intent: int = SYNTH_TRAIN_SAMPLES_PER_INTENT,
    exclude_val_test: bool = True,
    include_out_of_scope: bool = False,
) -> list[dict[str, Any]]:
    """
    Sinh tập câu huấn luyện tổng hợp (Synthetic Training Sentences) một cách tất định và tái lập.

    Tuân thủ các yêu cầu:
    1. Thiết lập hạt giống ngẫu nhiên tất định (set_seed).
    2. Khử trùng lặp độc quyền cho tập huấn luyện (deduplication exclusively for train split).
    3. Đảm bảo không trùng với tập val.csv và test.csv (Honesty Rule #4).
    4. Sinh đúng số lượng câu quy định cho mỗi intent (mặc định 10 câu/intent).

    Tham số:
        intents_path: Đường dẫn tới intents.json.
        seed: Hạt giống ngẫu nhiên (mặc định 42).
        samples_per_intent: Số lượng mẫu cần sinh cho mỗi intent (mặc định 10).
        exclude_val_test: Nếu True, loại bỏ mọi câu trùng với val.csv và test.csv.
        include_out_of_scope: Có bao gồm intent OUT_OF_SCOPE hay không.

    Trả về:
        Danh sách các bản ghi từ điển gồm: 'text', 'label', 'category', 'origin', 'synthetic'.
    """
    set_seed(seed)
    rng = random.Random(seed)

    # Đọc danh sách intent
    with open(intents_path, encoding="utf-8") as f:
        intents_data = json.load(f)

    # Nạp các câu cấm trùng từ val.csv và test.csv nếu có
    forbidden_sentences: set[str] = set()
    if exclude_val_test:
        forbidden_sentences = load_existing_sentences(VAL_CSV_PATH, TEST_CSV_PATH)

    # Tập hợp theo dõi câu đã sinh riêng cho train split (deduplication exclusively for train split)
    train_seen_normalized: set[str] = set(forbidden_sentences)

    generated_records: list[dict[str, Any]] = []

    for item in intents_data:
        intent_name = item.get("intent", "")
        if not intent_name:
            continue

        # Bỏ qua OUT_OF_SCOPE nếu không có cờ yêu cầu
        if intent_name == "OUT_OF_SCOPE" and not include_out_of_scope:
            continue

        category = item.get("category", "tex_misc")
        slots_def = item.get("slots", {})
        frames = SENTENCE_FRAMES.get(intent_name) or build_frames_for_intent(item)

        intent_samples: list[str] = []
        max_attempts = samples_per_intent * 40
        attempts = 0

        while len(intent_samples) < samples_per_intent and attempts < max_attempts:
            attempts += 1
            frame = rng.choice(frames)
            candidate = populate_frame(frame, slots_def, rng)
            norm_cand = normalize_sentence(candidate)

            # Kiểm tra tính độc nhất và không trùng lặp
            if norm_cand not in train_seen_normalized:
                train_seen_normalized.add(norm_cand)
                intent_samples.append(candidate)
                generated_records.append({
                    "text": candidate,
                    "label": intent_name,
                    "category": category,
                    "origin": "synthetic_frame_v2",
                    "synthetic": True,
                })

        # Nếu sau số lần thử vẫn chưa đủ câu do bộ khung quá hẹp, tạo thêm biến thể starter có seed
        if len(intent_samples) < samples_per_intent:
            action = extract_action_name(item.get("describe", ""), item.get("aliases", []))
            fallback_ctx = (
                "trong Obsidian"
                if category == "obsidian"
                else ("trong toán học" if category == "math_shared" else "trong LaTeX")
            )
            for starter in VIETNAMESE_PROMPT_STARTERS:
                if len(intent_samples) >= samples_per_intent:
                    break
                fallback_candidate = f"{starter} {action} {fallback_ctx} để sử dụng"
                norm_fb = normalize_sentence(fallback_candidate)
                if norm_fb not in train_seen_normalized:
                    train_seen_normalized.add(norm_fb)
                    intent_samples.append(fallback_candidate)
                    generated_records.append({
                        "text": fallback_candidate,
                        "label": intent_name,
                        "category": category,
                        "origin": "synthetic_frame_v2",
                        "synthetic": True,
                    })

    return generated_records


def export_synthetic_train_dataset(
    train_csv_path: Path = TRAIN_CSV_PATH,
    metadata_csv_path: Path = TRAIN_METADATA_PATH,
    intents_path: Path = DEFAULT_INTENTS_PATH,
    seed: int = DEFAULT_SEED,
    samples_per_intent: int = SYNTH_TRAIN_SAMPLES_PER_INTENT,
) -> int:
    """
    Xuất tập dữ liệu huấn luyện tổng hợp ra tệp train.csv và train_metadata.csv.
    Tuyệt đối không chỉnh sửa test.csv theo Honesty Rule #3 (AGENTS.md).

    Tham số:
        train_csv_path: Đường dẫn tệp train.csv đích.
        metadata_csv_path: Đường dẫn tệp train_metadata.csv đích.
        intents_path: Đường dẫn intents.json.
        seed: Hạt giống ngẫu nhiên.
        samples_per_intent: Số mẫu mỗi intent.

    Trả về:
        Tổng số câu đã xuất vào train.csv.
    """
    records = generate_synthetic_sentences(
        intents_path=intents_path,
        seed=seed,
        samples_per_intent=samples_per_intent,
        exclude_val_test=True,
    )

    # 1. Ghi train.csv (cột: text, label, category)
    train_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_csv_path, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "category"])
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "text": rec["text"],
                "label": rec["label"],
                "category": rec["category"],
            })

    # 2. Ghi train_metadata.csv (cột: text, label, origin, synthetic)
    metadata_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_csv_path, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "origin", "synthetic"])
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "text": rec["text"],
                "label": rec["label"],
                "origin": rec["origin"],
                "synthetic": rec["synthetic"],
            })

    return len(records)


def main(args: Sequence[str] | None = None) -> int:
    """
    Hàm thực thi giao diện dòng lệnh (CLI) cho trình sinh câu huấn luyện tổng hợp.
    """
    parser = argparse.ArgumentParser(
        description="Trình sinh câu huấn luyện tổng hợp tiếng Việt cho dự án Vietnamese LaTeX Assistant."
    )
    parser.add_argument(
        "--intents",
        type=Path,
        default=DEFAULT_INTENTS_PATH,
        help="Đường dẫn tệp intents.json (mặc định: data/intents.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Hạt giống ngẫu nhiên tất định (mặc định: 42)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=SYNTH_TRAIN_SAMPLES_PER_INTENT,
        help="Số lượng mẫu trên mỗi intent (mặc định: 10)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Xuất tập câu sinh ra vào data/train.csv và data/train_metadata.csv",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Kiểm tra tính hợp lệ và cấu trúc của SENTENCE_FRAMES",
    )

    parsed = parser.parse_args(args)

    if parsed.check:
        frames = build_sentence_frames(parsed.intents)
        print(f"Tổng số intent đã lập khung câu: {len(frames)}")
        for intent_name, f_list in list(frames.items())[:5]:
            print(f"- {intent_name}: {len(f_list)} frames (mẫu 1: {f_list[0]})")
        return 0

    if parsed.export:
        count = export_synthetic_train_dataset(
            intents_path=parsed.intents,
            seed=parsed.seed,
            samples_per_intent=parsed.samples,
        )
        print(f"Đã xuất thành công {count} câu huấn luyện tổng hợp vào train.csv và train_metadata.csv.")
        return 0

    # Chạy sinh thử nghiệm và hiển thị thống kê
    records = generate_synthetic_sentences(
        intents_path=parsed.intents,
        seed=parsed.seed,
        samples_per_intent=parsed.samples,
    )
    print(f"Đã sinh {len(records)} câu huấn luyện tổng hợp cho {len(set(r['label'] for r in records))} intents.")
    if records:
        print("Mẫu 3 câu đầu tiên:")
        for r in records[:3]:
            print(f"  [{r['label']}] {r['text']}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

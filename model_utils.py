
import math
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Giảm độ trễ suy luận trên CPU khi deploy demo nhỏ.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
try:
    torch.backends.mkldnn.enabled = False
except Exception:
    pass



TOKEN_PATTERN = re.compile(
    r"""
    %[0-9a-fA-F]{2}
    |&#x?[0-9a-fA-F]+;
    |[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*
    |\d+
    |[<>/='"():;.&?#%{}\[\]\+\-\*,\\]
    |\S
    """,
    re.VERBOSE,
)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_CHAR = "<PAD>"
UNK_CHAR = "<UNK>"


def xss_tokenizer(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(str(text))


def calculate_entropy(text: str) -> float:
    text = str(text)
    if len(text) == 0:
        return 0.0
    counter = Counter(text)
    probs = [count / len(text) for count in counter.values()]
    return float(-sum(p * math.log2(p) for p in probs))


def extract_structural_features(text: str) -> Dict[str, float]:
    text = str(text)
    lower = text.lower()
    features = {}

    length = len(text)
    features["length"] = length
    features["num_digits"] = sum(c.isdigit() for c in text)
    features["num_letters"] = sum(c.isalpha() for c in text)
    features["num_spaces"] = sum(c.isspace() for c in text)
    features["num_unique_chars"] = len(set(text))
    features["entropy"] = calculate_entropy(text)

    if length > 0:
        features["digit_ratio"] = features["num_digits"] / length
        features["letter_ratio"] = features["num_letters"] / length
        features["space_ratio"] = features["num_spaces"] / length
        features["special_char_ratio"] = sum(not c.isalnum() and not c.isspace() for c in text) / length
    else:
        features["digit_ratio"] = 0.0
        features["letter_ratio"] = 0.0
        features["space_ratio"] = 0.0
        features["special_char_ratio"] = 0.0

    special_chars = {
        "lt": "<", "gt": ">", "slash": "/", "backslash": "\\", "equal": "=",
        "double_quote": "\"", "single_quote": "'", "left_parenthesis": "(",
        "right_parenthesis": ")", "semicolon": ";", "colon": ":", "percent": "%",
        "ampersand": "&", "hash": "#", "question": "?", "plus": "+",
        "minus": "-", "dot": ".",
    }
    for name, ch in special_chars.items():
        features[f"count_{name}"] = text.count(ch)

    html_tags = [
        "script", "img", "svg", "iframe", "input", "body", "style", "a", "meta",
        "object", "embed", "form", "link", "video", "audio", "details", "marquee",
    ]
    for tag in html_tags:
        features[f"tag_{tag}"] = lower.count(tag)

    features["num_open_angle_brackets"] = text.count("<")
    features["num_close_angle_brackets"] = text.count(">")

    event_handler_matches = re.findall(r"on[a-zA-Z]+\s*=", lower)
    features["num_event_handlers"] = len(event_handler_matches)

    common_events = [
        "onload", "onerror", "onclick", "onmouseover", "onfocus", "onblur",
        "onmouseenter", "onmouseleave", "onpointerdown", "onpointerup", "onsubmit",
        "onchange", "onkeyup", "onkeydown",
    ]
    for ev in common_events:
        features[f"event_{ev}"] = lower.count(ev)

    js_keywords = {
        "alert": "alert", "prompt": "prompt", "confirm": "confirm", "eval": "eval",
        "document_cookie": "document.cookie", "document_write": "document.write",
        "window_location": "window.location", "settimeout": "settimeout",
        "setinterval": "setinterval", "fromcharcode": "fromcharcode",
        "constructor": "constructor", "function": "function", "return": "return",
    }
    for name, kw in js_keywords.items():
        features[f"js_{name}"] = lower.count(kw)

    percent_encoding = re.findall(r"%[0-9a-fA-F]{2}", text)
    html_entity = re.findall(r"&#x?[0-9a-fA-F]+;", text)
    unicode_escape = re.findall(r"\\u[0-9a-fA-F]{4}", text)
    hex_escape = re.findall(r"\\x[0-9a-fA-F]{2}", text)

    features["num_percent_encoding"] = len(percent_encoding)
    features["num_html_entity"] = len(html_entity)
    features["num_unicode_escape"] = len(unicode_escape)
    features["num_hex_escape"] = len(hex_escape)

    total_encoded = (
        features["num_percent_encoding"] + features["num_html_entity"] +
        features["num_unicode_escape"] + features["num_hex_escape"]
    )
    features["total_encoded_patterns"] = total_encoded
    features["encoded_ratio"] = total_encoded / length if length > 0 else 0.0

    dangerous_protocols = {
        "javascript_protocol": "javascript:",
        "data_protocol": "data:",
        "vbscript_protocol": "vbscript:",
    }
    for name, pattern in dangerous_protocols.items():
        features[name] = int(pattern in lower)

    dangerous_attrs = [
        "src=", "href=", "style=", "formaction=", "background=", "srcdoc=", "action=",
    ]
    for attr in dangerous_attrs:
        safe_name = attr.replace("=", "").replace("-", "_")
        features[f"attr_{safe_name}"] = lower.count(attr)

    return features


def encode_text_to_ids(text: str, vocab: Dict[str, int], max_len: int) -> np.ndarray:
    arr = np.zeros((1, max_len), dtype=np.int64)
    tokens = xss_tokenizer(text)
    token_ids = [vocab.get(tok, vocab.get(UNK_TOKEN, 1)) for tok in tokens][:max_len]
    arr[0, : len(token_ids)] = token_ids
    return arr


def encode_text_to_char_ids(text: str, char_vocab: Dict[str, int], max_len: int) -> np.ndarray:
    arr = np.zeros((1, max_len), dtype=np.int64)
    char_ids = [char_vocab.get(ch, char_vocab.get(UNK_CHAR, 1)) for ch in list(str(text))][:max_len]
    arr[0, : len(char_ids)] = char_ids
    return arr


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2

    def forward(self, text_ids):
        embedded = self.embedding(text_ids)
        _, (hidden, _) = self.lstm(embedded)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        text_vector = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.dropout(text_vector)


class StructuralEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_dim = output_dim

    def forward(self, struct_features):
        return self.encoder(struct_features)


class CharEncoder(nn.Module):
    def __init__(self, char_vocab_size, embedding_dim=64, num_filters=128, kernel_sizes=(3, 5, 7), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, char_ids):
        embedded = self.embedding(char_ids).transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            x = torch.relu(conv(embedded))
            x = torch.max(x, dim=2).values
            conv_outputs.append(x)
        char_vector = torch.cat(conv_outputs, dim=1)
        return self.dropout(char_vector)


class MultimodalFusionModel(nn.Module):
    def __init__(
        self, vocab_size, struct_input_dim, char_vocab_size,
        use_text=True, use_struct=True, use_char=True,
        text_embedding_dim=128, text_hidden_dim=128,
        struct_hidden_dim=128, struct_output_dim=128,
        char_embedding_dim=64, char_num_filters=128, char_kernel_sizes=(3, 5, 7),
        fusion_hidden_dim=256, dropout=0.3,
    ):
        super().__init__()
        self.use_text = use_text
        self.use_struct = use_struct
        self.use_char = use_char
        fusion_input_dim = 0
        if use_text:
            self.text_encoder = TextEncoder(vocab_size, text_embedding_dim, text_hidden_dim, dropout=dropout)
            fusion_input_dim += self.text_encoder.output_dim
        if use_struct:
            self.struct_encoder = StructuralEncoder(struct_input_dim, struct_hidden_dim, struct_output_dim, dropout)
            fusion_input_dim += self.struct_encoder.output_dim
        if use_char:
            self.char_encoder = CharEncoder(char_vocab_size, char_embedding_dim, char_num_filters, char_kernel_sizes, dropout)
            fusion_input_dim += self.char_encoder.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

    def forward(self, inputs):
        vectors = []
        input_index = 0
        if self.use_text:
            vectors.append(self.text_encoder(inputs[input_index]))
            input_index += 1
        if self.use_struct:
            vectors.append(self.struct_encoder(inputs[input_index]))
            input_index += 1
        if self.use_char:
            vectors.append(self.char_encoder(inputs[input_index]))
            input_index += 1
        return self.classifier(torch.cat(vectors, dim=1)).squeeze(1)


class XSSPredictor:
    def __init__(self, artifact_dir: str):
        self.artifact_dir = Path(artifact_dir)
        with open(self.artifact_dir / "xss_preprocessing_objects.pkl", "rb") as f:
            self.preprocessing = pickle.load(f)

        self.token_vocab = self.preprocessing["token_vocab"]
        self.char_vocab = self.preprocessing["char_vocab"]
        self.struct_feature_names = list(self.preprocessing["struct_feature_names"])
        self.scaler = self.preprocessing["scaler"]
        self.max_text_len = int(self.preprocessing["MAX_TEXT_LEN"])
        self.max_char_len = int(self.preprocessing["MAX_CHAR_LEN"])

        self.device = torch.device("cpu")
        self.model = MultimodalFusionModel(
            vocab_size=len(self.token_vocab),
            struct_input_dim=len(self.struct_feature_names),
            char_vocab_size=len(self.char_vocab),
            use_text=True,
            use_struct=True,
            use_char=True,
        ).to(self.device)

        state = torch.load(self.artifact_dir / "Text + Structural + Char_best.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.metrics = self._read_optional_csv("text_+_structural_+_char_test_metrics.csv")
        self.all_results = self._read_optional_csv("all_single_and_multimodal_results.csv")
        self.confusion = self._read_optional_csv("text_+_structural_+_char_confusion_matrix.csv")
        self.global_ig = self._read_optional_csv("global_text_ig_token_summary_v2_full_test.csv")
        self.global_lime = self._read_optional_csv("global_text_lime_token_summary_v2_full_test.csv")
        self.global_struct = self._read_optional_csv("global_structural_shap_summary_full_test.csv")
        self.global_modality = self._read_optional_csv("xai_global_modality_contribution.csv")
        self.dataset_summary = self._read_optional_csv("dataset_summary.csv")
        self.label_distribution = self._read_optional_csv("label_distribution.csv")

    def _read_optional_csv(self, filename: str) -> pd.DataFrame:
        path = self.artifact_dir / filename
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def preprocess(self, payload: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
        text_ids = encode_text_to_ids(payload, self.token_vocab, self.max_text_len)
        char_ids = encode_text_to_char_ids(payload, self.char_vocab, self.max_char_len)

        raw_features = extract_structural_features(payload)
        feature_df = pd.DataFrame([{name: raw_features.get(name, 0.0) for name in self.struct_feature_names}])
        struct_scaled = self.scaler.transform(feature_df).astype(np.float32)

        return (
            torch.tensor(text_ids, dtype=torch.long, device=self.device),
            torch.tensor(struct_scaled, dtype=torch.float32, device=self.device),
            torch.tensor(char_ids, dtype=torch.long, device=self.device),
            feature_df,
        )

    def predict(self, payload: str) -> Dict:
        text_tensor, struct_tensor, char_tensor, feature_df = self.preprocess(payload)
        with torch.no_grad():
            logit = self.model((text_tensor, struct_tensor, char_tensor))
            prob_xss = torch.sigmoid(logit).item()

        tokens = xss_tokenizer(payload)
        result = {
            "prob_xss": float(prob_xss),
            "prob_benign": float(1.0 - prob_xss),
            "label": "XSS" if prob_xss >= 0.5 else "Benign",
            "risk_level": self.risk_level(prob_xss),
            "tokens": tokens,
            "structural_features": feature_df.iloc[0].to_dict(),
            "top_local_signals": self.top_local_signals(payload, tokens, feature_df),
        }
        return result

    @staticmethod
    def risk_level(prob_xss: float) -> str:
        if prob_xss >= 0.80:
            return "Nguy hiểm"
        if prob_xss >= 0.50:
            return "Cảnh báo"
        if prob_xss >= 0.30:
            return "Cần kiểm tra thêm"
        return "An toàn"

    def top_local_signals(self, payload: str, tokens: List[str], feature_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        token_set = {str(t).lower() for t in tokens}

        if not self.global_ig.empty:
            df = self.global_ig.copy()
            token_col = next((c for c in df.columns if "token" in c.lower() or "ngram" in c.lower()), None)
            score_col = next((c for c in df.columns if "mean" in c.lower() and ("attr" in c.lower() or "importance" in c.lower())), None)
            if token_col and score_col:
                for _, row in df.head(200).iterrows():
                    tok = str(row[token_col]).lower()
                    if tok and tok in token_set:
                        rows.append({"Nhóm": "Text IG toàn cục", "Tín hiệu": str(row[token_col]), "Điểm/giá trị": float(row[score_col])})

        if not self.global_struct.empty:
            df = self.global_struct.copy()
            feat_col = next((c for c in df.columns if "feature" in c.lower()), None)
            score_col = next((c for c in df.columns if "mean" in c.lower() and ("shap" in c.lower() or "abs" in c.lower())), None)
            if feat_col and score_col:
                for _, row in df.head(100).iterrows():
                    feat = str(row[feat_col])
                    val = float(feature_df.iloc[0].get(feat, 0.0))
                    if abs(val) > 0:
                        rows.append({"Nhóm": "Structural SHAP toàn cục", "Tín hiệu": feat, "Điểm/giá trị": val})

        heuristic = [
            ("Có thẻ <script>", "<script" in payload.lower()),
            ("Có event handler on*=...", bool(re.search(r"on[a-z]+\s*=", payload.lower()))),
            ("Có javascript:", "javascript:" in payload.lower()),
            ("Có mã hóa %xx / entity / unicode", bool(re.search(r"%[0-9a-fA-F]{2}|&#x?[0-9a-fA-F]+;|\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}", payload))),
            ("Có hàm JS nhạy cảm alert/eval/document.cookie", bool(re.search(r"alert|eval|document\.cookie|document\.write", payload.lower()))),
        ]
        for name, ok in heuristic:
            if ok:
                rows.append({"Nhóm": "Dấu hiệu bảo mật", "Tín hiệu": name, "Điểm/giá trị": 1.0})

        if not rows:
            rows.append({"Nhóm": "Dấu hiệu", "Tín hiệu": "Không tìm thấy tín hiệu XSS rõ ràng theo bộ luật mô tả", "Điểm/giá trị": 0.0})

        return pd.DataFrame(rows).drop_duplicates().head(12)

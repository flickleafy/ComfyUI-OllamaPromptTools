import hashlib
import math
import re
from dataclasses import dataclass, field


DEFAULT_PLACEHOLDER_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PLACEHOLDER_SIMILARITY_THRESHOLD = 0.34
LOG_PREFIX = "[ComfyUI-OllamaPromptTools]"

_PLACEHOLDER_HEADER_RE = re.compile(r"^\s*\[([A-Z][A-Z0-9_]*)\]\s*$")
_PLACEHOLDER_MARKER_RE = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")
_SEPARATOR_RE = re.compile(r"^\s*---\s*$")
_PLACEHOLDER_METADATA_RE = re.compile(r"^\s*@(match|weight|require_any)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_NUMBERED_LINE_PREFIX_RE = re.compile(r"^\s*\d+[\.)]\s*")
_LOW_PRIORITY_INSTRUCTION_RE = re.compile(
    r"^(avoid|do not|don't|mute|remove|replace|output only|do not explain)\b",
    re.IGNORECASE,
)
_SEARCH_TOKEN_RE = re.compile(r"[a-z0-9]+")

_MODEL_CACHE = {}
_TEMPLATE_EMBEDDING_CACHE = {}


class PlaceholderResolutionError(ValueError):
    pass


@dataclass
class PlaceholderCandidate:
    text: str
    match_texts: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    require_any: list = field(default_factory=list)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.text == other
        if isinstance(other, PlaceholderCandidate):
            return (
                self.text == other.text
                and self.match_texts == other.match_texts
                and self.weights == other.weights
                and self.require_any == other.require_any
            )
        return NotImplemented


@dataclass
class PlaceholderCatalog:
    candidates: dict = field(default_factory=dict)
    defaults: dict = field(default_factory=dict)


def parse_placeholder_templates(placeholder_templates):
    catalog = PlaceholderCatalog()
    current_label = None
    current_lines = []

    def flush_current():
        nonlocal current_label, current_lines
        if current_label is None:
            return

        body = "\n".join(current_lines).strip()
        if body:
            candidate = _parse_placeholder_candidate(body)
            if not candidate.text:
                current_label = None
                current_lines = []
                return

            if current_label.endswith("_DEFAULT"):
                base_label = current_label[: -len("_DEFAULT")]
                catalog.defaults.setdefault(base_label, candidate.text)
            else:
                catalog.candidates.setdefault(current_label, []).append(candidate)

        current_label = None
        current_lines = []

    for raw_line in (placeholder_templates or "").splitlines():
        header_match = _PLACEHOLDER_HEADER_RE.match(raw_line)
        if header_match:
            flush_current()
            current_label = header_match.group(1)
            current_lines = []
            continue

        if _SEPARATOR_RE.match(raw_line):
            flush_current()
            continue

        if current_label is not None:
            current_lines.append(raw_line)

    flush_current()
    return catalog


def find_placeholder_labels(text):
    labels = []
    seen = set()
    for match in _PLACEHOLDER_MARKER_RE.finditer(text or ""):
        label = match.group(1)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def materialize_prompt_placeholders(
    prompt,
    placeholder_templates,
    delimiter,
    embedding_model=DEFAULT_PLACEHOLDER_EMBEDDING_MODEL,
    similarity_threshold=DEFAULT_PLACEHOLDER_SIMILARITY_THRESHOLD,
    embedder=None,
):
    if not (placeholder_templates or "").strip():
        return prompt or ""

    prompt = prompt or ""
    base_prompt, delimiter_found, query_text = split_prompt_for_placeholder_matching(prompt, delimiter)
    resolved_base = resolve_placeholders(
        base_prompt,
        query_text,
        placeholder_templates,
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        embedder=embedder,
    )
    if delimiter_found:
        return f"{resolved_base}{delimiter}{query_text}"
    return resolved_base


def split_prompt_for_placeholder_matching(prompt, delimiter):
    prompt = prompt or ""
    delimiter = delimiter or ""
    if delimiter:
        base_prompt, found, remainder = prompt.partition(delimiter)
        if found:
            return base_prompt, True, remainder
    return prompt, False, prompt


def resolve_placeholders(
    base_prompt,
    query_text,
    placeholder_templates,
    embedding_model=DEFAULT_PLACEHOLDER_EMBEDDING_MODEL,
    similarity_threshold=DEFAULT_PLACEHOLDER_SIMILARITY_THRESHOLD,
    embedder=None,
):
    base_prompt = base_prompt or ""
    labels = find_placeholder_labels(base_prompt)
    if not labels:
        return base_prompt

    catalog = parse_placeholder_templates(placeholder_templates)
    replacements = _resolve_replacements(
        labels,
        query_text or "",
        catalog,
        placeholder_templates or "",
        embedding_model,
        float(similarity_threshold),
        embedder,
    )

    resolved = base_prompt
    for label in labels:
        resolved = resolved.replace(f"[{label}]", replacements[label])
    return resolved


def _resolve_replacements(
    labels,
    query_text,
    catalog,
    placeholder_templates,
    embedding_model,
    similarity_threshold,
    embedder,
):
    candidates_by_label = {}
    candidate_match_texts = []
    for label in labels:
        candidates = catalog.candidates.get(label, [])
        match_texts = _build_candidate_match_texts(candidates, catalog.defaults.get(label))
        candidates_by_label[label] = list(zip(candidates, match_texts))
        candidate_match_texts.extend(match_texts)

    candidate_vectors = []
    query_vector = None
    if candidate_match_texts:
        query_vector = _embed_texts([query_text], embedding_model, embedder)[0]
        candidate_vectors = _get_candidate_embeddings(
            placeholder_templates,
            embedding_model,
            candidate_match_texts,
            embedder,
        )

    replacements = {}
    candidate_offset = 0
    normalized_query_text = _normalize_search_text(query_text)
    for label in labels:
        best_text = None
        best_score = None
        best_source = "semantic_match"

        for candidate, _match_text in candidates_by_label.get(label, []):
            candidate_vector = candidate_vectors[candidate_offset]
            candidate_offset += 1
            score = _score_candidate(query_vector, candidate_vector, candidate, normalized_query_text)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_text = candidate.text
                best_source = "weighted_semantic_match" if candidate.weights else "semantic_match"

        if best_score is not None and best_score >= similarity_threshold:
            _log_selected_placeholder_block(
                label,
                best_text,
                source=best_source,
                score=best_score,
                similarity_threshold=similarity_threshold,
            )
            replacements[label] = best_text
            continue

        default_text = catalog.defaults.get(label)
        if default_text is not None:
            _log_selected_placeholder_block(
                label,
                default_text,
                source="default_fallback",
                score=best_score,
                similarity_threshold=similarity_threshold,
            )
            replacements[label] = default_text
            continue

        raise PlaceholderResolutionError(
            f"No semantic match or default block found for placeholder [{label}]. "
            f"Add a [{label}_DEFAULT] block or lower the placeholder_similarity_threshold."
        )

    return replacements


def _parse_placeholder_candidate(body):
    body_lines = []
    match_texts = []
    weights = {}
    require_any = []

    for line in (body or "").splitlines():
        metadata_match = _PLACEHOLDER_METADATA_RE.match(line)
        if not metadata_match:
            body_lines.append(line)
            continue

        key = metadata_match.group(1).casefold()
        value = metadata_match.group(2).strip()
        if not value:
            continue

        if key == "match":
            match_texts.append(value)
        elif key == "weight":
            weights.update(_parse_weight_metadata(value))
        elif key == "require_any":
            require_any.extend(_parse_list_metadata(value))

    return PlaceholderCandidate(
        text="\n".join(body_lines).strip(),
        match_texts=match_texts,
        weights=weights,
        require_any=require_any,
    )


def _parse_weight_metadata(value):
    weights = {}
    for item in _parse_list_metadata(value):
        term = item
        raw_weight = "1"
        if "=" in item:
            term, raw_weight = item.rsplit("=", 1)
        elif ":" in item:
            term, raw_weight = item.rsplit(":", 1)

        term = _normalize_search_text(term)
        if not term:
            continue

        try:
            weight = float(raw_weight.strip())
        except ValueError:
            weight = 1.0
        if weight > 0.0:
            weights[term] = weight
    return weights


def _parse_list_metadata(value):
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _build_candidate_match_texts(candidates, default_text=None):
    common_lines = _find_common_body_lines(candidates, default_text)
    return [_candidate_match_text(candidate, common_lines) for candidate in candidates]


def _find_common_body_lines(candidates, default_text=None):
    body_texts = [candidate.text for candidate in candidates]
    if default_text:
        body_texts.append(default_text)

    line_counts = {}
    for text in body_texts:
        seen_lines = set()
        for line in (text or "").splitlines():
            normalized = _normalize_body_line(line)
            if normalized:
                seen_lines.add(normalized)
        for line in seen_lines:
            line_counts[line] = line_counts.get(line, 0) + 1

    return {line for line, count in line_counts.items() if count > 1}


def _candidate_match_text(candidate, common_lines):
    if candidate.match_texts:
        return "\n".join(candidate.match_texts)

    high_priority_lines = []
    low_priority_lines = []
    for line in (candidate.text or "").splitlines():
        stripped = line.strip()
        normalized = _normalize_body_line(stripped)
        if not normalized or normalized in common_lines:
            continue

        clean_line = _NUMBERED_LINE_PREFIX_RE.sub("", stripped).strip()
        if _LOW_PRIORITY_INSTRUCTION_RE.match(clean_line):
            low_priority_lines.append(clean_line)
        else:
            high_priority_lines.append(clean_line)

    selected_lines = high_priority_lines or low_priority_lines
    if selected_lines:
        return "\n".join(selected_lines)
    return candidate.text


def _score_candidate(query_vector, candidate_vector, candidate, normalized_query_text):
    if candidate.require_any and not _any_term_matches(normalized_query_text, candidate.require_any):
        return None

    semantic_score = cosine_similarity(query_vector, candidate_vector)
    lexical_score = _weighted_lexical_score(normalized_query_text, candidate.weights)
    if not candidate.weights:
        return semantic_score

    hybrid_score = (semantic_score * 0.75) + (lexical_score * 0.25)
    return max(semantic_score, hybrid_score, lexical_score)


def _weighted_lexical_score(normalized_text, weights):
    if not weights:
        return 0.0

    matched_weight = 0.0
    max_weight = 0.0
    for term, weight in weights.items():
        max_weight = max(max_weight, weight)
        if _term_matches(normalized_text, term):
            matched_weight += weight

    if max_weight <= 0.0:
        return 0.0
    return min(1.0, matched_weight / max_weight)


def _any_term_matches(normalized_text, terms):
    return any(_term_matches(normalized_text, term) for term in terms)


def _term_matches(normalized_text, term):
    normalized_term = _normalize_search_text(term)
    if not normalized_term:
        return False
    pattern = rf"(?:^|\s){re.escape(normalized_term)}(?:\s|$)"
    return re.search(pattern, normalized_text) is not None


def _normalize_body_line(line):
    line = _NUMBERED_LINE_PREFIX_RE.sub("", line or "").strip()
    return _normalize_search_text(line)


def _normalize_search_text(text):
    tokens = _SEARCH_TOKEN_RE.findall((text or "").casefold())
    return " ".join(tokens)


def _log_selected_placeholder_block(label, text, source, score, similarity_threshold):
    if score is None:
        score_text = "n/a"
    else:
        score_text = f"{score:.4f}"
    message = (
        f"{LOG_PREFIX} Placeholder [{label}] selected block "
        f"(source={source}, score={score_text}, threshold={similarity_threshold:.4f}):\n{text}"
    )
    print(message, flush=True)


def _get_candidate_embeddings(placeholder_templates, embedding_model, candidate_texts, embedder):
    if embedder is not None:
        return _embed_texts(candidate_texts, embedding_model, embedder)

    cache_key = (
        embedding_model,
        hashlib.sha256((placeholder_templates or "").encode("utf-8")).hexdigest(),
        tuple(candidate_texts),
    )
    cached = _TEMPLATE_EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached

    vectors = _embed_texts(candidate_texts, embedding_model, embedder)
    _TEMPLATE_EMBEDDING_CACHE[cache_key] = vectors
    return vectors


def _embed_texts(texts, embedding_model, embedder):
    text_list = [text or "" for text in texts]
    if embedder is not None:
        vectors = embedder(text_list)
    else:
        model = _load_embedding_model(embedding_model)
        vectors = model.encode(text_list, convert_to_numpy=True, normalize_embeddings=True)
    return [_coerce_vector(vector) for vector in vectors]


def _load_embedding_model(embedding_model):
    model_name = (embedding_model or DEFAULT_PLACEHOLDER_EMBEDDING_MODEL).strip()
    if not model_name:
        model_name = DEFAULT_PLACEHOLDER_EMBEDDING_MODEL

    model = _MODEL_CACHE.get(model_name)
    if model is not None:
        return model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise PlaceholderResolutionError(
            "Placeholder semantic matching requires the 'sentence-transformers' Python package. "
            "Install it in the ComfyUI Python environment or leave placeholder_templates empty."
        ) from exc

    try:
        model = SentenceTransformer(model_name, device="cpu")
    except Exception as exc:
        raise PlaceholderResolutionError(
            f"Could not load placeholder embedding model '{model_name}' on CPU. "
            "Install the model/package in the ComfyUI Python environment or choose another "
            "placeholder_embedding_model."
        ) from exc

    _MODEL_CACHE[model_name] = model
    return model


def _coerce_vector(vector):
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if vector and isinstance(vector[0], (list, tuple)):
        vector = vector[0]
    return [float(value) for value in vector]


def cosine_similarity(left, right):
    if left is None or right is None:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))

"""Evaluation Engine for Validator."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from openclaw_llm import parse_json_response

try:
    from local_evaluation_engine import is_local_scorer_enabled, local_evaluate
except ImportError:
    def is_local_scorer_enabled() -> bool:
        return False
    def local_evaluate(*args, **kwargs):
        return None

log = logging.getLogger("validator.evaluation")

DEFAULT_TIMEOUT = 120


@dataclass
class EvaluationResult:
    """Result of data evaluation."""
    result: str  # "match" | "mismatch"
    verdict: str  # "accepted" | "rejected"
    consistent: bool
    score: int  # 0-100, meaningful only when result="match"


import concurrent.futures

# 单例线程池，用于在已有 event loop 的情况下执行 async LLM 调用。
# 避免每次 evaluate 都创建新的 ThreadPoolExecutor。
_LLM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _default_llm_call(
    prompt: str,
    *,
    timeout: float,
    model_config: dict[str, Any] | None,
) -> str:
    """Sync wrapper around the shared llm_enrich routing."""
    from crawler.enrich.generative.llm_enrich import enrich_with_llm

    async def _run() -> str:
        result = await enrich_with_llm(
            prompt,
            model_config=model_config,
            timeout=timeout,
        )
        if not result.success:
            raise RuntimeError(result.error or "LLM call failed")
        return result.content

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return _LLM_EXECUTOR.submit(lambda: asyncio.run(_run())).result()
    except RuntimeError:
        pass
    return asyncio.run(_run())


class EvaluationEngine:
    """
    Single-pass evaluation engine for data quality assessment.

    Uses one LLM call to perform authenticity check (M0 vs M1), consistency check,
    and quality scoring (completeness, accuracy, type correctness, sufficiency).
    """

    def __init__(
        self,
        *,
        llm_call: Callable[[str], str] | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        model_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the evaluation engine.

        Args:
            llm_call: Optional callable for LLM calls. If None, uses the shared
                llm_enrich routing (CLI → gateway → API) so the validator works
                on hosts that don't have the openclaw binary installed.
            timeout: Timeout in seconds for LLM calls.
            model_config: Optional model config dict for gateway/API fallback.
                When empty, only the CLI path is available.
        """
        self.timeout = timeout
        self.model_config = model_config or {}
        if llm_call is None:
            cfg = self.model_config
            self.llm_call = lambda prompt: _default_llm_call(
                prompt, timeout=timeout, model_config=cfg
            )
        else:
            self.llm_call = llm_call

    def evaluate(
        self,
        cleaned_data: str | dict[str, Any],
        structured_data: dict[str, Any],
        schema_fields: list[str],
        repeat_cleaned_data: str = "",
        dataset_schema: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Single-pass evaluation per protocol: authenticity + consistency + quality in one LLM call.

        When LOCAL_SCORER=1 env var is set, uses deterministic rule-based scoring
        instead of LLM. Falls back to LLM if local scorer cannot handle the input.

        Args:
            cleaned_data: Original miner submission (M0).
            structured_data: Miner-extracted structured data.
            schema_fields: List of field names from schema.
            repeat_cleaned_data: Re-crawled data from repeat crawl miner (M1).
            dataset_schema: Full dataset schema definition with types and required fields.
        """
        # Local scorer bypass: skip LLM entirely for data quality evaluation
        if is_local_scorer_enabled():
            local_result = local_evaluate(
                cleaned_data, structured_data, schema_fields,
                repeat_cleaned_data=repeat_cleaned_data,
                dataset_schema=dataset_schema,
            )
            if local_result is not None:
                return EvaluationResult(
                    result=local_result["result"],
                    verdict=local_result["verdict"],
                    consistent=local_result["consistent"],
                    score=local_result["score"],
                )
            log.debug("Local scorer returned None, falling back to LLM")

        if isinstance(cleaned_data, dict):
            cleaned_data_str = json.dumps(cleaned_data, ensure_ascii=False, separators=(",", ":"))
        else:
            cleaned_data_str = str(cleaned_data)

        structured_json = json.dumps(structured_data, ensure_ascii=False, separators=(",", ":"))
        if dataset_schema:
            schema_json = json.dumps(dataset_schema, ensure_ascii=False, separators=(",", ":"))
        else:
            schema_json = json.dumps({"fields": schema_fields}, ensure_ascii=False, separators=(",", ":"))

        has_repeat = bool(repeat_cleaned_data and repeat_cleaned_data.strip())

        # Pre-LLM optimization: reduce M0 and M1 with identical rules
        cleaned_data_str = _optimize_for_eval(cleaned_data_str)
        if has_repeat:
            repeat_cleaned_data = _optimize_for_eval(str(repeat_cleaned_data))

        # Build single prompt covering all evaluation phases
        sections = []
        sections.append("You are a data quality evaluator for a decentralized data mining network.")
        sections.append("")

        if has_repeat:
            sections.append("## Data")
            sections.append("")
            sections.append("### Original crawl (M0)")
            sections.append(cleaned_data_str)
            sections.append("")
            sections.append("### Re-crawl (M1)")
            sections.append(str(repeat_cleaned_data))
        else:
            sections.append("## Original data")
            sections.append(cleaned_data_str)

        sections.append("")
        sections.append("## Structured data extracted by miner")
        sections.append(structured_json)
        sections.append("")
        sections.append("## Dataset schema")
        sections.append(schema_json)

        sections.append("")
        sections.append("## Evaluation instructions")
        if has_repeat:
            sections.append("Evaluate in this order:")
            sections.append("")
            sections.append("1. **Check if M1 is unusable**: If M1 is ANY of the following, it is")
            sections.append("   unusable for comparison — set result to \"match\" and score M0 alone:")
            sections.append("   - CAPTCHA, login wall, access-denied, anti-bot challenge")
            sections.append("   - Cookie consent page, age verification gate, paywall")
            sections.append("   - HTTP error page (403, 404, 500, etc.)")
            sections.append("   - Empty page, placeholder, or boilerplate-only page")
            sections.append("   - Content in a completely different language than M0")
            sections.append("   - Redirect landing page unrelated to the original URL")
            sections.append("   - Any page that is clearly NOT the same content as M0")
            sections.append("   Do NOT penalize the miner when M1 is broken — the re-crawler")
            sections.append("   may have been blocked or the page may have changed.")
            sections.append("")
            sections.append("2. **Authenticity check (M0 vs M1)**: Compare M0 and M1 with a")
            sections.append("   **lenient / forgiving standard**. Web pages are dynamic — content")
            sections.append("   is cleaned, reformatted, and timestamps/ads/navigation change")
            sections.append("   between crawls. Apply these rules:")
            sections.append("   - If the CORE SEMANTIC CONTENT is similar, report \"match\".")
            sections.append("     Same topic, same key facts, same entity = match.")
            sections.append("   - Differences that are NORMAL and should be ignored:")
            sections.append("     timestamps, dates, view counts, ad blocks, navigation menus,")
            sections.append("     related-article links, formatting/whitespace, section ordering,")
            sections.append("     minor wording changes, dynamic pricing, stock availability,")
            sections.append("     comment counts, social media widgets, cookie banners.")
            sections.append("   - Only report \"mismatch\" when M0 is clearly FABRICATED:")
            sections.append("     entirely invented content with no basis in M1, a completely")
            sections.append("     different page/topic, or data that contradicts M1 on key facts.")
            sections.append("   - When in doubt, report \"match\". Err on the side of leniency.")
            sections.append("   If mismatch, set score to 0 and skip quality scoring.")
            sections.append("")
            sections.append("3. **Quality scoring**: If match, score structured_data quality (0-100) based on:")
        else:
            sections.append("No re-crawl data (M1) is available. This is normal — not every")
            sections.append("submission has a re-crawl. Set result to \"match\" unconditionally")
            sections.append("and score structured_data quality based on M0 alone:")

        sections.append("   - Completeness (30 points): are all required schema fields present and non-empty?")
        sections.append("   - Accuracy (40 points): do values correctly reflect the original data?")
        sections.append("   - Type correctness (15 points): do values match their schema-defined types?")
        sections.append("   - Information sufficiency (15 points): is obvious information from the source missing?")
        sections.append("   Total: 100 points maximum. A perfect extraction with all fields correct scores 100.")
        sections.append("")
        sections.append("## Output (strict JSON only, no markdown, no explanation)")
        sections.append('{"result": "match" or "mismatch", "score": <integer 0 to 100>}')

        prompt = "\n".join(sections)

        try:
            response = self.llm_call(prompt)
            result = parse_json_response(response)

            # Normalize keys to lowercase for case-insensitive matching
            if result:
                result = {k.lower(): v for k, v in result.items()}

            eval_result, eval_score = self._extract_result_and_score(result, response, has_repeat)

            eval_score = max(0, min(100, eval_score))

            if eval_result == "mismatch":
                return EvaluationResult(
                    result="mismatch",
                    verdict="rejected",
                    consistent=False,
                    score=0,
                )

            # match 始终 verdict="accepted"——score 只反映数据质量高低，
            # 不影响 match/mismatch 判定。之前 score=0 时 verdict="rejected"
            # 导致宿主 LLM 误解为"被平台拒绝"。
            return EvaluationResult(
                result="match",
                verdict="accepted",
                consistent=True,
                score=eval_score,
            )

        except Exception as e:
            log.error("evaluation failed (infrastructure): %s", str(e))
            # Infrastructure failure — don't penalize miners for evaluator faults
            return EvaluationResult(
                result="match",
                verdict="accepted",
                consistent=True,
                score=50,
            )

    # score 解析失败时的默认分——避免因为 LLM 返回格式问题惩罚 miner。
    # 70 分 = "数据大概率有效但无法精确评分"。
    _DEFAULT_SCORE_ON_PARSE_FAILURE = 70

    @staticmethod
    def _parse_score_value(raw: Any) -> int | None:
        """尝试从各种 LLM 输出格式中提取整数分值。

        支持: 85, 85.5, "85", "85/100", "85%", "around 85", "~85",
        "score: 85", null/None, 空字符串。
        返回 None 表示无法解析。
        """
        if raw is None:
            return None

        s = str(raw).strip()
        if not s:
            return None

        # 直接数字 (int/float)
        try:
            return int(float(s))
        except (TypeError, ValueError):
            pass

        # "85/100" 或 "85 / 100"
        m = re.match(r"(\d+(?:\.\d+)?)\s*/\s*100", s)
        if m:
            return int(float(m.group(1)))

        # "85%" 或 "85 %"
        m = re.match(r"(\d+(?:\.\d+)?)\s*%", s)
        if m:
            return int(float(m.group(1)))

        # "around 85", "~85", "approximately 85"
        m = re.search(r"(?:around|approximately|about|~)\s*(\d+(?:\.\d+)?)", s, re.IGNORECASE)
        if m:
            return int(float(m.group(1)))

        # 最后兜底：字符串里的第一个纯数字
        m = re.search(r"\b(\d{1,3})\b", s)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val

        return None

    @staticmethod
    def _extract_result_and_score(
        parsed: dict[str, Any] | None,
        raw_response: str,
        has_repeat: bool,
    ) -> tuple[str, int]:
        """Extract result and score from LLM response with maximum tolerance.

        Handles: key case variations, value case, non-JSON text, missing fields.
        Returns (result, score) tuple.
        """
        default_score = EvaluationEngine._DEFAULT_SCORE_ON_PARSE_FAILURE

        # Try from parsed JSON first (keys already lowercased)
        if parsed:
            raw_result = str(parsed.get("result", ""))
            raw_score = parsed.get("score")

            # Normalize result value
            if raw_result.lower() in ("match", "true", "yes", "authentic", "same"):
                eval_result = "match"
            elif raw_result.lower() in ("mismatch", "false", "no", "fraud", "different", "fabricated"):
                eval_result = "mismatch"
            elif not raw_result:
                # Result key empty — use parsed score if available, else text fallback
                score = EvaluationEngine._parse_score_value(raw_score)
                if score is not None and score > 0:
                    return "match", max(0, min(100, score))
                return EvaluationEngine._extract_result_and_score(None, raw_response, has_repeat)
            else:
                # Ambiguous verdict — fall back to raw text extraction
                return EvaluationEngine._extract_result_and_score(None, raw_response, has_repeat)

            # Normalize score value
            score = EvaluationEngine._parse_score_value(raw_score)
            if score is None:
                log.warning(
                    "[eval] score 解析失败，使用默认分 %d (raw_score=%r, response=%s)",
                    default_score, raw_score, raw_response[:200],
                )
                score = default_score
            return eval_result, score

        # Fallback: extract from raw text when JSON parsing failed entirely
        text = raw_response.lower()

        # Detect result from text
        if "mismatch" in text or "fabricat" in text or "fraud" in text:
            eval_result = "mismatch"
        else:
            eval_result = "match"

        # Detect score from text (look for number near "score")
        score_patterns = [
            r'score["\s:]*(\d+)',
            r'(\d+)["\s]*/?\s*100',
            r'(\d{1,3})\s*(?:out of|/)\s*100',
        ]
        for pattern in score_patterns:
            m = re.search(pattern, text)
            if m:
                try:
                    val = int(m.group(1))
                    if 0 <= val <= 100:
                        return eval_result, val
                except ValueError:
                    pass

        # 文本中也找不到分数——对 match 使用默认分
        if eval_result == "match":
            log.warning(
                "[eval] 文本 fallback 无法提取分数，使用默认分 %d (response=%s)",
                default_score, raw_response[:200],
            )
            return eval_result, default_score
        return eval_result, 0


# M0/M1 每侧最大字符数。Accuracy 占 40% 评分权重，LLM 必须看到足够多的
# 源文本才能验证结构化数据——之前 20000 chars (~5K tokens) 导致长页面被
# 大幅截断后 LLM 无法核对，给出 0 分。50000 chars (~12.5K tokens/侧) 在
# 现代 128K+ context 模型上完全安全。
_EVAL_MAX_CHARS = 50000

_LOW_VALUE_HEADING = re.compile(
    r"(?im)^#{1,3}\s*("
    r"references|bibliography|citations|notes|footnotes|"
    r"see also|further reading|external links|sources|"
    r"related articles|related pages|navigation|categories|"
    r"disclaimers?|copyright"
    r")\s*$"
)
_CITATION_RE = re.compile(r"\[\s*(?:\d+|note\s+\d+|citation needed)\s*\]")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _optimize_for_eval(text: str) -> str:
    """Reduce M0/M1 text before sending to LLM.

    Applies identical rules to both sides so comparison remains fair.
    """
    if not text or len(text) < _EVAL_MAX_CHARS:
        text = _CITATION_RE.sub("", text)
        text = _MULTI_BLANK_RE.sub("\n\n", text)
        return text.strip()

    lines = text.split("\n")
    result = []
    skip = False
    skip_level = 0
    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            if _LOW_VALUE_HEADING.match(f"{'#' * level} {heading_match.group(2).strip()}"):
                skip = True
                skip_level = level
                continue
            if skip and level <= skip_level:
                skip = False
        if not skip:
            result.append(line)
    text = "\n".join(result)

    text = _CITATION_RE.sub("", text)

    paragraphs = re.split(r"\n{2,}", text)
    seen: set[str] = set()
    unique = []
    for para in paragraphs:
        key = re.sub(r"\s+", " ", para.strip().lower())
        if len(key) < 20 or key not in seen:
            if len(key) >= 20:
                seen.add(key)
            unique.append(para)
    text = "\n\n".join(unique)

    text = _MULTI_BLANK_RE.sub("\n\n", text).strip()

    if len(text) > _EVAL_MAX_CHARS:
        text = text[:_EVAL_MAX_CHARS].rsplit("\n", 1)[0] + "\n..."

    return text


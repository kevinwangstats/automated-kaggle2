"""
LLM Configuration and Provider Resolver.

Defaults to DeepSeek (DEEPSEEK_API_KEY -> deepseek/deepseek-chat),
with built-in support for Moonshot (MOONSHOT_API_KEY) and other providers.
"""

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Dict, Optional, Tuple
import litellm


@dataclass
class LLMConfig:
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1500


def get_default_llm_config(preferred_model: Optional[str] = None) -> LLMConfig:
    """
    Resolves the LLM configuration based on available API keys.
    Tiering policy:
    - Chinese Providers (DeepSeek / Kimi): Default to TOP-TIER flagship models
      - DeepSeek: 'deepseek/deepseek-chat' (DeepSeek-V3 671B MoE) or 'deepseek/deepseek-reasoner' (DeepSeek-R1)
      - Kimi / Moonshot: 'kimi-k3' (Flagship 2.8T params, 1M context) / 'kimi-k2.7-code'
    - Western Providers (Anthropic / Gemini / OpenAI): Default to MID-TIER workhorse models
      - Anthropic: 'claude-3-5-sonnet-20241022' (Sonnet mid-tier workhorse)
      - Gemini: 'gemini/gemini-2.0-flash' (Gemini Flash mid-tier)
      - OpenAI: 'gpt-4o' (Mid-tier multimodal workhorse)
    """
    if preferred_model:
        # Route Kimi models to Moonshot OpenAI-compatible endpoint if needed
        if "kimi" in preferred_model.lower():
            moonshot_key = os.environ.get("MOONSHOT_API_KEY")
            return LLMConfig(
                model=f"openai/{preferred_model}" if not preferred_model.startswith("openai/") else preferred_model,
                api_key=moonshot_key,
                api_base="https://api.moonshot.cn/v1",
                temperature=0.2,
            )
        return LLMConfig(model=preferred_model)

    # 1. DeepSeek (Top-Tier Default: DeepSeek-V3 / deepseek-chat)
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        return LLMConfig(
            model="deepseek/deepseek-chat",
            api_key=deepseek_key,
            temperature=0.2,
        )

    # 2. Kimi / Moonshot (Top-Tier Flagship: kimi-k3 via Moonshot API)
    moonshot_key = os.environ.get("MOONSHOT_API_KEY")
    if moonshot_key:
        return LLMConfig(
            model="openai/kimi-k3",
            api_key=moonshot_key,
            api_base="https://api.moonshot.cn/v1",
            temperature=0.2,
        )

    # 3. Anthropic (Mid-Tier Workhorse: Claude 3.5 Sonnet)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        return LLMConfig(model="claude-3-5-sonnet-20241022", api_key=anthropic_key)

    # 4. Gemini (Mid-Tier Workhorse: Gemini 2.0 Flash)
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        return LLMConfig(model="gemini/gemini-2.0-flash", api_key=gemini_key)

    # 5. OpenAI (Mid-Tier Workhorse: GPT-4o)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return LLMConfig(model="gpt-4o", api_key=openai_key)

    # Fallback default: DeepSeek top-tier
    return LLMConfig(model="deepseek/deepseek-chat")


def call_llm_for_json(
    config: LLMConfig,
    messages: list[dict[str, str]],
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls the LLM via litellm and robustly extracts JSON dictionary.
    Handles providers that support native json_object and those that return markdown code blocks.
    """
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    call_kwargs: Dict[str, Any] = {
        "model": config.model,
        "messages": full_messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.api_key:
        call_kwargs["api_key"] = config.api_key
    if config.api_base:
        call_kwargs["api_base"] = config.api_base

    try:
        call_kwargs["response_format"] = {"type": "json_object"}
        response = litellm.completion(**call_kwargs)
    except Exception:
        # Retry without explicit response_format for models that don't support it
        call_kwargs.pop("response_format", None)
        response = litellm.completion(**call_kwargs)

    raw_text = response.choices[0].message.content.strip()

    # Extract JSON string if wrapped in markdown code fence ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if match:
        raw_text = match.group(1).strip()

    return json.loads(raw_text)

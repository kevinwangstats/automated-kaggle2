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
    Priority:
    1. Explicit preferred_model if passed
    2. DEEPSEEK_API_KEY -> 'deepseek/deepseek-chat' (Default)
    3. MOONSHOT_API_KEY -> 'moonshot/moonshot-v1-8k'
    4. OPENAI_API_KEY -> 'gpt-4o-mini'
    5. ANTHROPIC_API_KEY -> 'claude-3-5-haiku-20241022'
    6. GEMINI_API_KEY -> 'gemini/gemini-1.5-flash'
    """
    if preferred_model:
        return LLMConfig(model=preferred_model)

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        return LLMConfig(
            model="deepseek/deepseek-chat",
            api_key=deepseek_key,
            temperature=0.2,
        )

    moonshot_key = os.environ.get("MOONSHOT_API_KEY")
    if moonshot_key:
        return LLMConfig(
            model="moonshot/moonshot-v1-8k",
            api_key=moonshot_key,
            temperature=0.2,
        )

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return LLMConfig(model="gpt-4o-mini", api_key=openai_key)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        return LLMConfig(model="claude-3-5-haiku-20241022", api_key=anthropic_key)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        return LLMConfig(model="gemini/gemini-1.5-flash", api_key=gemini_key)

    # Fallback default
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

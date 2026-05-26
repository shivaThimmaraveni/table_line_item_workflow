import json
from typing import Any, AsyncGenerator, Dict, List

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings
from app.logging_setup import get_logger

logger = get_logger(__name__)


def _extract_text_from_response(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: List[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    out.append(text)
        return "".join(out)
    return str(content or "")


def _looks_like_foundation_model_id(model_id: str) -> bool:
    m = (model_id or "").strip()
    if not m:
        return False
    # Typical direct model IDs that map to foundation-model ARNs.
    if m.startswith(("amazon.", "anthropic.", "meta.", "cohere.", "mistral.")):
        return True
    # Explicit foundation-model ARN.
    if m.startswith("arn:aws:bedrock:") and ":foundation-model/" in m:
        return True
    return False


def _require_inference_profile(model_id: str, setting_name: str) -> None:
    if not (model_id or "").strip():
        raise ValueError(
            f"{setting_name} is empty. Set a Bedrock inference profile model ID/ARN."
        )
    if _looks_like_foundation_model_id(model_id):
        raise ValueError(
            f"{setting_name} points to a foundation model ID/ARN ({model_id}). "
            "Use an inference profile model ID/ARN instead."
        )


def _is_model_arn(model_id: str) -> bool:
    return (model_id or "").strip().startswith("arn:aws:bedrock:")


class BedrockConnector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        _require_inference_profile(settings.bedrock_chat_model, "BEDROCK_CHAT_MODEL")
        _require_inference_profile(settings.bedrock_summary_model, "BEDROCK_SUMMARY_MODEL")
        _require_inference_profile(settings.bedrock_embed_model, "BEDROCK_EMBED_MODEL")
        for i, model_id in enumerate(settings.bedrock_embed_fallback_models, start=1):
            _require_inference_profile(model_id, f"BEDROCK_EMBED_FALLBACK_MODELS[{i}]")
        self._embedder = BedrockEmbeddings(
            model_id=settings.bedrock_embed_model,
            region_name=settings.aws_region,
        )
        logger.info(
            "Bedrock models configured (inference profiles only): chat=%s chat_provider=%s summary=%s summary_provider=%s embed=%s fallbacks=%s",
            settings.bedrock_chat_model,
            settings.bedrock_chat_provider,
            settings.bedrock_summary_model,
            settings.bedrock_summary_provider,
            settings.bedrock_embed_model,
            list(settings.bedrock_embed_fallback_models),
        )

    def _provider_for_model(self, model: str, payload: Dict[str, Any]) -> str | None:
        explicit = str(payload.get("provider") or "").strip()
        if explicit:
            return explicit
        if not _is_model_arn(model):
            return None
        if model == self.settings.bedrock_summary_model:
            return self.settings.bedrock_summary_provider
        return self.settings.bedrock_chat_provider

    async def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        model = payload.get("model") or self.settings.bedrock_chat_model
        _require_inference_profile(str(model), "payload.model")
        provider = self._provider_for_model(str(model), payload)
        if _is_model_arn(str(model)) and not provider:
            raise ValueError(
                "Model ARN requires provider. Set payload.provider or configure "
                "BEDROCK_CHAT_PROVIDER/BEDROCK_SUMMARY_PROVIDER."
            )
        llm = ChatBedrockConverse(
            model=model,
            provider=provider,
            region_name=self.settings.aws_region,
            temperature=float(payload.get("temperature", 0.0)),
            max_tokens=int(payload.get("max_tokens", 1024)),
        )
        response = await llm.ainvoke(
            [
                SystemMessage(content=str(payload.get("system_prompt", ""))),
                HumanMessage(content=str(payload.get("user_prompt", ""))),
            ]
        )
        text = _extract_text_from_response(response.content)
        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = text
        return {"response": parsed, "raw_text": text}

    async def stream_invoke(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        model = payload.get("model") or self.settings.bedrock_chat_model
        _require_inference_profile(str(model), "payload.model")
        provider = self._provider_for_model(str(model), payload)
        if _is_model_arn(str(model)) and not provider:
            raise ValueError(
                "Model ARN requires provider. Set payload.provider or configure "
                "BEDROCK_CHAT_PROVIDER/BEDROCK_SUMMARY_PROVIDER."
            )
        llm = ChatBedrockConverse(
            model=model,
            provider=provider,
            region_name=self.settings.aws_region,
            temperature=float(payload.get("temperature", 0.0)),
            max_tokens=int(payload.get("max_tokens", 2048)),
        )
        async for chunk in llm.astream(
            [
                SystemMessage(content=str(payload.get("system_prompt", ""))),
                HumanMessage(content=str(payload.get("user_prompt", ""))),
            ]
        ):
            text = _extract_text_from_response(chunk.content)
            if text:
                yield text

    async def generate_embedding(self, text: str) -> List[float]:
        # embed_query is sync in langchain_aws; run in thread-like wrapper
        import asyncio

        candidate_models = [self.settings.bedrock_embed_model] + list(
            self.settings.bedrock_embed_fallback_models
        )
        errors: List[str] = []
        for model_id in candidate_models:
            try:
                embedder = self._embedder
                if model_id != self.settings.bedrock_embed_model:
                    embedder = BedrockEmbeddings(
                        model_id=model_id,
                        region_name=self.settings.aws_region,
                    )
                return await asyncio.to_thread(embedder.embed_query, text)
            except Exception as e:
                errors.append(f"{model_id}: {e}")
                continue
        raise RuntimeError(
            "Embedding failed for all configured inference profiles. "
            f"Tried: {candidate_models}. Errors: {errors}"
        )


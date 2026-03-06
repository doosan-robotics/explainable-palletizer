"""OpenAI-compatible async client for the vLLM inference server."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class InferenceClient:
    """Wraps the ``openai.AsyncOpenAI`` client for vLLM communication.

    vLLM exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint,
    so the standard ``openai`` SDK works with just a ``base_url`` override.

    Parameters
    ----------
    base_url:
        vLLM server URL (e.g. ``http://inference-server:8200/v1``).
    model:
        Model name as registered in vLLM (e.g. ``nvidia/Cosmos-Reason2-2B``).
    timeout:
        Request timeout in seconds.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 30.0) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key="unused",
            timeout=timeout,
        )
        self._model = model

    async def close(self) -> None:
        await self._client.close()

    async def health(self) -> bool:
        """Check if the inference server is reachable by listing models."""
        try:
            await self._client.models.list()
            return True
        except Exception:
            logger.warning("Inference server health check failed", exc_info=True)
            return False

    async def get_plan(self, system_prompt: str, scenario_text: str) -> str:
        """Send a palletizing scenario to the model and return the response text.

        Parameters
        ----------
        system_prompt:
            System message defining the robot controller persona.
        scenario_text:
            The user-facing scenario description (boxes, pallets, etc.).

        Returns
        -------
        str
            The model's generated response text.
        """
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": scenario_text},
            ],
        )
        if not response.choices:
            return ""
        choice = response.choices[0]
        return choice.message.content or ""

    async def get_action(
        self,
        system_prompt: str,
        images: list[bytes],
        scenario_text: str,
        *,
        max_tokens: int = 2048,
    ) -> str:
        """Send multimodal palletizing prompt and return the model's response.

        Parameters
        ----------
        system_prompt:
            System message.
        images:
            PNG image bytes (one per box).
        scenario_text:
            Full user-turn text (task prompt + scenario).
        max_tokens:
            Maximum completion tokens.
        """
        import base64

        content_parts: list[dict] = []
        for img in images:
            b64 = base64.b64encode(img).decode()
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        content_parts.append({"type": "text", "text": scenario_text})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ],
            max_tokens=max_tokens,
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    async def continue_response(
        self,
        system_prompt: str,
        user_text: str,
        partial_response: str,
        *,
        max_tokens: int = 512,
    ) -> str:
        """Continue a truncated model response by sending it back as assistant context.

        Sends the partial response as an assistant turn, then asks the model
        to output only the JSON answer block.
        """
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": partial_response},
                {
                    "role": "user",
                    "content": (
                        "Your thinking was cut off. Now output ONLY the JSON "
                        "action object, nothing else. No <think> block, no "
                        "explanation, just the raw JSON."
                    ),
                },
            ],
            max_tokens=max_tokens,
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

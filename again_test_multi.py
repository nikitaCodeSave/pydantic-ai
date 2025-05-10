import os
import asyncio
import logging
from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

###############################################################################
# Настройка логирования и провайдера
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ollama_provider = OpenAIProvider(base_url="http://localhost:11434/v1")
ollama_model = OpenAIModel(model_name="qwen2.5:32b", provider=ollama_provider)

###############################################################################
# Модели данных
###############################################################################
@dataclass
class AgentIdentifier:
    ROUTER: str = "router_agent"
    WEATHER: str = "weather_agent"
    TRANSLATOR: str = "translator_agent"
    FINALIZER: str = "finalizer_agent"


class WeatherResponse(BaseModel):
    location: str
    temperature: str
    conditions: str
    additional_info: Optional[str] = None


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str


class FinalResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)


@dataclass
class OrchestrationContext:
    original_user_query: str
    accumulated_data: List[Dict[str, Any]] = field(default_factory=list)
    processing_complete: bool = False
    used_agents: Dict[str, int] = field(default_factory=dict)


@dataclass
class OrchestratorAction:
    thought: str
    action_type: str
    query_for_next_agent: str

###############################################################################
# Определение агентов
###############################################################################
router_agent = Agent(
    model=ollama_model,
    deps_type=OrchestrationContext,
    name=AgentIdentifier.ROUTER,
    end_strategy="early",
    model_settings={"temperature": 0.2},
    output_type=OrchestratorAction,
    tools=[],  # router_agent не вызывает инструменты
    retries=3,
)


@router_agent.system_prompt
def router_system_prompt_func(ctx: RunContext[OrchestrationContext]) -> str:
    accumulated_data_str = ""
    if ctx.deps and ctx.deps.accumulated_data:
        accumulated_data_str = "\nСобранные данные:\n"
        for item in ctx.deps.accumulated_data:
            src = item.get("source_agent", "unknown")
            rslt = item.get("result", {})
            accumulated_data_str += f"- От {src}: {rslt}\n"

    user_question = ctx.deps.original_user_query if ctx.deps else "Неизвестный запрос"

    base_prompt = (
        "Вы — Router-агент. Проанализируйте запрос пользователя и уже собранные данные, "
        "чтобы решить, нужен ли ещё вызов weather_agent или translator_agent, "
        "или пора передать всё finalizer_agent.\n\n"
        f"Текущий запрос пользователя: {user_question}\n"
        f"{accumulated_data_str}\n"
        "Правила:\n"
        "1. Если нужна погода и её ещё нет → action_type=weather_agent.\n"
        "2. Если нужен перевод и его ещё нет → action_type=translator_agent.\n"
        "3. Иначе → action_type=finalizer_agent.\n\n"
        "Выводите JSON-объект OrchestratorAction с полями:\n"
        "thought, action_type, query_for_next_agent.\n"
        "Нельзя вызывать инструменты напрямую.\n"
    )
    return base_prompt


weather_agent = Agent(
    model=ollama_model,
    name=AgentIdentifier.WEATHER,
    model_settings={"temperature": 0.3},
    output_type=WeatherResponse,
)


@weather_agent.system_prompt
def weather_system_prompt() -> str:
    return (
        "Вы — Weather-агент. Получив запрос о погоде, верните WeatherResponse "
        "с location, temperature, conditions и т.п. Данные реалистичные, но вымышленные."
    )


translator_agent = Agent(
    model=ollama_model,
    name=AgentIdentifier.TRANSLATOR,
    model_settings={"temperature": 0.1},
    output_type=TranslationResponse,
)


@translator_agent.system_prompt
def translator_system_prompt() -> str:
    return (
        "Вы — Translation-агент. Переведите полученный текст на требуемый язык, "
        "вернув TranslationResponse. Определите язык-источник, если он не указан."
    )


finalizer_agent = Agent(
    model=ollama_model,
    deps_type=OrchestrationContext,
    name=AgentIdentifier.FINALIZER,
    model_settings={"temperature": 0.3},
    output_type=FinalResponse,
)


@finalizer_agent.system_prompt
def finalizer_system_prompt(ctx: RunContext[OrchestrationContext]) -> str:
    accumulated_data_str = ""
    source_agents = []
    if ctx.deps and ctx.deps.accumulated_data:
        accumulated_data_str = "Собранные данные:\n"
        for item in ctx.deps.accumulated_data:
            src = item.get("source_agent", "unknown")
            rslt = item.get("result", {})
            source_agents.append(src)
            accumulated_data_str += f"- От {src}: {rslt}\n"

    original_query = ctx.deps.original_user_query if ctx.deps else "Неизвестный запрос"

    return (
        "Вы — Финализатор. Соберите весь контекст и ответьте на исходный вопрос "
        "пользователя на том же языке, что и вопрос.\n\n"
        f"Исходный вопрос: {original_query}\n\n"
        f"{accumulated_data_str}\n"
        "Формат ответа: FinalResponse с полями answer и sources."
    )

###############################################################################
# Оркестратор
###############################################################################
async def orchestrate_agents(user_query: str) -> FinalResponse:
    context = OrchestrationContext(original_user_query=user_query)

    while not context.processing_complete:
        # 1. Router решает, что делать
        router_result = await router_agent.run(user_query, deps=context)
        action = router_result.output

        logger.debug(
            "[Router] action_type=%s thought=%s next_query=%s",
            action.action_type, action.thought, action.query_for_next_agent
        )

        if action.action_type == AgentIdentifier.FINALIZER:
            final_result = await finalizer_agent.run(
                action.query_for_next_agent, deps=context
            )
            context.processing_complete = True
            return final_result.output

        elif action.action_type == AgentIdentifier.WEATHER:
            weather_result = await weather_agent.run(action.query_for_next_agent)
            context.accumulated_data.append({
                "source_agent": AgentIdentifier.WEATHER,
                "result": weather_result.output.model_dump(),
            })
            # снова к Router с обновлённым контекстом
            user_query += f"\n(Получена погода: {weather_result.output.model_dump()})"

        elif action.action_type == AgentIdentifier.TRANSLATOR:
            translation_result = await translator_agent.run(action.query_for_next_agent)
            context.accumulated_data.append({
                "source_agent": AgentIdentifier.TRANSLATOR,
                "result": translation_result.output.model_dump(),
            })
            user_query += f"\n(Получен перевод: {translation_result.output.model_dump()})"

        else:
            logger.error("Неизвестный action_type %s. Завершаем.", action.action_type)
            context.processing_complete = True

    # Если вышли без финализатора
    return FinalResponse(
        answer="Произошла ошибка: цикл завершён без finalizer_agent.",
        sources=[d.get("source_agent") for d in context.accumulated_data],
    )


async def main() -> None:
    query = "Какая погода в Москве и переведи 'good morning' на русский?"
    result = await orchestrate_agents(query)
    print("\n===== РЕЗУЛЬТАТ =====")
    print("Answer :", result.answer)
    print("Sources:", result.sources)


if __name__ == "__main__":
    asyncio.run(main())

# Документация по Мульти-агентной системе (again_test_multi.py)

## Содержание
1. [Введение](#введение)
2. [Установка и зависимости](#установка-и-зависимости)
3. [Архитектура системы](#архитектура-системы)
4. [Модели данных](#модели-данных)
5. [Агенты системы](#агенты-системы)
   - [Router Agent](#router-agent)
   - [Weather Agent](#weather-agent)
   - [Translator Agent](#translator-agent)
   - [Finalizer Agent](#finalizer-agent)
6. [Оркестратор](#оркестратор)
7. [Поток выполнения](#поток-выполнения)
8. [Пример работы](#пример-работы)
9. [Расширение функционала](#расширение-функционала)
10. [Отладка и логирование](#отладка-и-логирование)
11. [FAQ](#faq)

## Введение

`again_test_multy.py` представляет собой пример реализации мульти-агентной системы на базе библиотеки Pydantic-AI. Система демонстрирует архитектуру, в которой несколько специализированных AI-агентов работают совместно для решения комплексных задач:

- Маршрутизация запросов 
- Получение информации о погоде
- Перевод текста
- Формирование финального ответа

Код использует асинхронное программирование (asyncio), типизацию данных (с помощью Pydantic) и демонстрирует подход к орхестрации нескольких LLM-агентов для формирования цепочки действий.

## Установка и зависимости

### Необходимые пакеты:

```bash
pip install pydantic pydantic-ai
```

### Для локального запуска:
- [Ollama](https://github.com/ollama/ollama) с предустановленной моделью qwen2.5:32b
- Python 3.8+

## Архитектура системы

Система построена на основе следующих компонентов:

1. **Модели данных** (Pydantic и dataclasses) - определяют структуру входных и выходных данных
2. **Агенты** - специализированные LLM-компоненты для решения конкретных задач
3. **Оркестратор** - координирует работу агентов в зависимости от запроса
4. **Контекст оркестрации** - хранит состояние системы и накопленные данные

## Модели данных

### Идентификаторы агентов

```python
@dataclass
class AgentIdentifier:
    ROUTER: str = "router_agent"
    WEATHER: str = "weather_agent"
    TRANSLATOR: str = "translator_agent"
    FINALIZER: str = "finalizer_agent"
```

Класс `AgentIdentifier` содержит константы для идентификации различных агентов в системе. Использование констант вместо строковых литералов напрямую помогает избежать опечаток и упрощает рефакторинг.

### Ответы агентов

#### WeatherResponse
```python
class WeatherResponse(BaseModel):
    location: str
    temperature: str
    conditions: str
    additional_info: Optional[str] = None
```

Модель для структурированного ответа от агента погоды. Содержит:
- `location` - местоположение для прогноза
- `temperature` - температура (в виде строки с единицами измерения)
- `conditions` - описание погодных условий
- `additional_info` - дополнительная информация (опционально)

#### TranslationResponse
```python
class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
```

Модель для ответа от переводчика. Содержит:
- `original_text` - исходный текст для перевода
- `translated_text` - результат перевода
- `source_language` - язык оригинала
- `target_language` - целевой язык перевода

#### FinalResponse
```python
class FinalResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
```

Модель для финального ответа. Содержит:
- `answer` - итоговый ответ для пользователя
- `sources` - список источников использованных данных

### Контекст и действия

#### OrchestrationContext
```python
@dataclass
class OrchestrationContext:
    original_user_query: str
    accumulated_data: List[Dict[str, Any]] = field(default_factory=list)
    processing_complete: bool = False
    used_agents: Dict[str, int] = field(default_factory=dict)
```

Класс для хранения контекста оркестрации:
- `original_user_query` - исходный запрос пользователя
- `accumulated_data` - список накопленных данных от всех агентов
- `processing_complete` - флаг завершения обработки
- `used_agents` - счетчик использования агентов (подготовлен в структуре, но не используется в текущей реализации)

#### OrchestratorAction
```python
@dataclass
class OrchestratorAction:
    thought: str
    action_type: str
    query_for_next_agent: str
```

Модель решения маршрутизатора:
- `thought` - обоснование решения (для отладки и прозрачности)
- `action_type` - тип действия/следующий агент для вызова
- `query_for_next_agent` - запрос для передачи следующему агенту

## Агенты системы

Система содержит четыре специализированных агента:

### Router Agent

```python
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
```

**Назначение**: Анализирует запрос пользователя и принимает решение о маршрутизации к нужному агенту.  

**Параметры**:
- **model**: экземпляр модели Ollama (qwen2.5:32b)
- **deps_type**: тип зависимостей - OrchestrationContext
- **name**: имя агента из AgentIdentifier
- **end_strategy**: "early" - стратегия раннего завершения (агент прекращает работу после первого валидного ответа)
- **model_settings**: температура 0.2 для более предсказуемых ответов
- **output_type**: результат должен соответствовать структуре OrchestratorAction
- **tools**: пустой список, т.к. этот агент не использует инструменты
- **retries**: 3 попытки, если что-то пойдет не так

**Системный промпт**:
```python
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
```

Системный промпт для Router Agent:
1. Формирует строку с накопленными данными от других агентов
2. Получает исходный запрос пользователя из контекста
3. Строит промпт с инструкциями по принятию решения
4. Объясняет правила маршрутизации и формат вывода

### Weather Agent

```python
weather_agent = Agent(
    model=ollama_model,
    name=AgentIdentifier.WEATHER,
    model_settings={"temperature": 0.3},
    output_type=WeatherResponse,
)
```

**Назначение**: Предоставляет информацию о погоде в указанной локации.

**Параметры**:
- **model**: тот же экземпляр модели Ollama
- **name**: имя агента
- **model_settings**: температура 0.3 для баланса между креативностью и точностью
- **output_type**: результат должен соответствовать структуре WeatherResponse

**Системный промпт**:
```python
@weather_agent.system_prompt
def weather_system_prompt() -> str:
    return (
        "Вы — Weather-агент. Получив запрос о погоде, верните WeatherResponse "
        "с location, temperature, conditions и т.п. Данные реалистичные, но вымышленные."
    )
```

Weather Agent имеет простой системный промпт, поскольку его задача узконаправленная. Важное замечание: агент возвращает реалистичные, но вымышленные данные о погоде, т.к. не имеет доступа к реальным данным.

### Translator Agent

```python
translator_agent = Agent(
    model=ollama_model,
    name=AgentIdentifier.TRANSLATOR,
    model_settings={"temperature": 0.1},
    output_type=TranslationResponse,
)
```

**Назначение**: Переводит текст с одного языка на другой.

**Параметры**:
- **model**: та же модель Ollama
- **name**: имя агента
- **model_settings**: низкая температура 0.1 для более точных переводов
- **output_type**: результат должен соответствовать структуре TranslationResponse

**Системный промпт**:
```python
@translator_agent.system_prompt
def translator_system_prompt() -> str:
    return (
        "Вы — Translation-агент. Переведите полученный текст на требуемый язык, "
        "вернув TranslationResponse. Определите язык-источник, если он не указан."
    )
```

Translator Agent также имеет четкую и узкую специализацию с простым промптом. Особенность: агент должен сам определить язык оригинала, если он явно не указан.

### Finalizer Agent

```python
finalizer_agent = Agent(
    model=ollama_model,
    deps_type=OrchestrationContext,
    name=AgentIdentifier.FINALIZER,
    model_settings={"temperature": 0.3},
    output_type=FinalResponse,
)
```

**Назначение**: Собирает информацию от других агентов и формирует финальный ответ.

**Параметры**:
- **model**: та же модель Ollama
- **deps_type**: тип зависимостей - OrchestrationContext
- **name**: имя агента
- **model_settings**: температура 0.3 для баланса между креативностью и точностью
- **output_type**: результат должен соответствовать структуре FinalResponse

**Системный промпт**:
```python
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
```

Finalizer Agent получает доступ ко всему накопленному контексту и:
1. Формирует строку с результатами работы предыдущих агентов
2. Собирает список источников информации
3. Получает исходный запрос пользователя
4. Строит промпт с инструкцией ответить на исходный вопрос с учетом всех полученных данных
5. Важно: ответ должен быть на том же языке, что и исходный запрос

## Оркестратор

Оркестратор - центральный компонент системы, управляющий последовательностью вызовов агентов:

```python
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
```

Оркестратор выполняет следующие функции:

1. **Инициализация контекста** - создает контекст оркестрации с исходным запросом пользователя
2. **Цикл обработки** - продолжается, пока флаг `processing_complete` не станет `True`
3. **Вызов Router Agent** - определяет дальнейшие действия
4. **Логирование** - записывает отладочную информацию о принятых решениях
5. **Ветвление логики** в зависимости от решения маршрутизатора:
   - Вызов Finalizer Agent - завершение обработки и возврат результата
   - Вызов Weather Agent - получение данных о погоде
   - Вызов Translator Agent - перевод текста
   - Обработка неизвестного действия - логирование ошибки и завершение
6. **Накопление результатов** - добавление результатов работы агентов в контекст
7. **Обновление запроса** - дополнение исходного запроса информацией о полученных результатах
8. **Обработка ошибок** - возврат сообщения об ошибке, если цикл завершен без вызова финализатора

## Поток выполнения

Полный цикл работы системы:

1. **Вход**: пользовательский запрос (например, "Какая погода в Москве и переведи 'good morning' на русский?")
2. **Начальная оркестрация**: создается контекст с исходным запросом
3. **Первый запуск Router Agent**:
   - Анализирует запрос пользователя
   - Принимает решение (например, "Нужна информация о погоде")
   - Возвращает результат с типом действия и запросом для следующего агента
4. **Вызов Weather Agent**:
   - Получает запрос о погоде
   - Возвращает структурированную информацию (WeatherResponse)
   - Результат добавляется в контекст
5. **Повторный запуск Router Agent**:
   - Анализирует исходный запрос + накопленные данные о погоде
   - Принимает решение (например, "Нужен перевод")
6. **Вызов Translator Agent**:
   - Получает запрос на перевод
   - Возвращает результат перевода (TranslationResponse)
   - Результат добавляется в контекст
7. **Очередной запуск Router Agent**:
   - Анализирует исходный запрос + все накопленные данные
   - Принимает решение ("Все данные собраны, нужен финальный ответ")
8. **Вызов Finalizer Agent**:
   - Получает весь контекст
   - Формирует итоговый ответ для пользователя с учетом всех данных
   - Включает в ответ список источников информации
9. **Выход**: структурированный финальный ответ (FinalResponse)

## Пример работы

```python
async def main() -> None:
    query = "Какая погода в Москве и переведи 'good morning' на русский?"
    result = await orchestrate_agents(query)
    print("\n===== РЕЗУЛЬТАТ =====")
    print("Answer :", result.answer)
    print("Sources:", result.sources)
```

Пример запроса: "Какая погода в Москве и переведи 'good morning' на русский?"

Возможный результат:
```
===== РЕЗУЛЬТАТ =====
Answer : В Москве сейчас солнечно и +22°C. Перевод фразы "good morning" на русский - "доброе утро".
Sources: ['weather_agent', 'translator_agent']
```

## Расширение функционала

Система спроектирована модульно, что позволяет легко расширять ее функциональность:

### Добавление нового агента

1. Создать модель для ответа нового агента
   ```python
   class NewAgentResponse(BaseModel):
       field1: str
       field2: int
       # другие поля
   ```

2. Добавить идентификатор в класс AgentIdentifier
   ```python
   @dataclass
   class AgentIdentifier:
       # существующие агенты
       NEW_AGENT: str = "new_agent"
   ```

3. Создать и настроить новый агент
   ```python
   new_agent = Agent(
       model=ollama_model,
       name=AgentIdentifier.NEW_AGENT,
       model_settings={"temperature": 0.2},
       output_type=NewAgentResponse,
   )

   @new_agent.system_prompt
   def new_agent_system_prompt() -> str:
       return "Инструкции для нового агента..."
   ```

4. Добавить обработку в оркестратор
   ```python
   elif action.action_type == AgentIdentifier.NEW_AGENT:
       new_agent_result = await new_agent.run(action.query_for_next_agent)
       context.accumulated_data.append({
           "source_agent": AgentIdentifier.NEW_AGENT,
           "result": new_agent_result.output.model_dump(),
       })
       user_query += f"\n(Получены данные: {new_agent_result.output.model_dump()})"
   ```

5. Обновить логику Router Agent, чтобы он знал о новом агенте

### Интеграция внешних API

Для интеграции с реальными API (например, для получения настоящих данных о погоде):

1. Добавить необходимые зависимости (например, `aiohttp` для асинхронных HTTP-запросов)
2. Создать инструмент (Tool) для вызова внешнего API
3. Добавить инструмент к соответствующему агенту

```python
from pydantic_ai import Tool

# Создание инструмента для запроса погоды
weather_api_tool = Tool(
    name="get_real_weather",
    description="Gets real weather information for a location",
    async_fn=fetch_real_weather,
    input_type=LocationInput,
    output_type=RealWeatherData,
)

# Обновление агента
weather_agent = Agent(
    model=ollama_model,
    name=AgentIdentifier.WEATHER,
    model_settings={"temperature": 0.3},
    output_type=WeatherResponse,
    tools=[weather_api_tool],  # добавление инструмента
)
```

## Отладка и логирование

Система использует стандартный модуль `logging` Python для отладки:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
```

Текущие настройки:
- Уровень логирования: INFO
- Формат: время - имя логгера - уровень - сообщение

Для расширенного логирования можно:

1. Изменить уровень логирования на DEBUG
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Добавить запись логов в файл
   ```python
   logging.basicConfig(
       level=logging.DEBUG,
       format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
       filename="agent_orchestration.log",
       filemode="a",
   )
   ```

3. Добавить специфичное логирование для каждого агента
   ```python
   weather_logger = logging.getLogger("weather_agent")
   weather_logger.debug("Processing weather request: %s", query)
   ```

## FAQ

### 1. Какую модель использует система?

Система использует модель `qwen2.5:32b` через Ollama - локальный инференс-сервер.

### 2. Как система обрабатывает несколько запросов одновременно?

Текущая реализация не поддерживает параллельную обработку запросов. Функция `orchestrate_agents` обрабатывает один запрос за раз. Для поддержки нескольких одновременных запросов потребуется добавление queue manager или использование фреймворка вроде FastAPI для создания асинхронного API-сервера.

### 3. Что делать, если агент выдает ошибку?

В текущей реализации Router Agent имеет параметр `retries=3`, что дает ему три попытки в случае ошибки. Для других агентов можно добавить обработку исключений:

```python
try:
    weather_result = await weather_agent.run(action.query_for_next_agent)
except Exception as e:
    logger.error("Weather agent failed: %s", str(e))
    weather_result = WeatherResponse(
        location="unknown",
        temperature="unknown",
        conditions="Error fetching weather data",
    )
```

### 4. Как добавить аутентификацию для использования OpenAI вместо Ollama?

Для использования OpenAI API вместо Ollama:

```python
from pydantic_ai.providers.openai import OpenAIProvider

openai_provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
openai_model = OpenAIModel(model_name="gpt-4", provider=openai_provider)

# Обновление всех агентов
router_agent = Agent(
    model=openai_model,
    # остальные параметры те же
)
# и так далее для всех агентов
```

### 5. Как улучшить производительность системы?

Для улучшения производительности:
- Используйте кэширование результатов для частых запросов
- Оптимизируйте промпты для сокращения токенов
- Используйте более легкие модели для простых задач
- Организуйте параллельное выполнение независимых агентов (например, Weather и Translator можно вызывать параллельно)

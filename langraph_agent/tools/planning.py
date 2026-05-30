from __future__ import annotations

import json

from langchain_core.tools import tool
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class AskHumanToolInput(BaseModel):
    """定义 ask_human 工具支持的两种入参格式。

    Description:
        将选择题固定为 choose_list 字典形式，将说明题固定为 question 文本形式；
        choose_list 可在一次工具调用中包含多个问题。
    Args:
        choose_list (dict[str, list[str]] | None): 选择题数据，键为问题，值为选项列表。
        question (str | None): 需要用户自由回答的说明题。
    Returns:
        None: Pydantic 模型类本身不返回运行值。
    """

    choose_list: dict[str, list[str]] | None = Field(
        default=None,
        description='选择题格式，例如 {"要采用哪种方案？": ["方案 A", "方案 B"]}。',
    )
    question: str | None = Field(
        default=None,
        description="说明题格式，例如“请描述需要兼容的输入格式”。",
    )

    @field_validator("choose_list", mode="before")
    @classmethod
    def parse_choose_list_json(cls, value: object) -> object:
        """兼容模型将选择题字典序列化为 JSON 字符串的情况。

        Description:
            当模型把 choose_list 的对象参数错误地包成 JSON 字符串时，先尝试
            将其还原为对象，再交给字段类型校验和内容校验处理。
        Args:
            value (object): 模型输出的原始 choose_list 参数值。
        Returns:
            object: 成功解析后的 JSON 对象，或保持原样交由 Pydantic 报告类型错误。
        """
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @model_validator(mode="after")
    def validate_question_mode(self) -> "AskHumanToolInput":
        """校验提问模式与选择题内容。

        Description:
            确保单次调用只使用选择题或说明题中的一种形式，并阻止空问题与
            空选项进入终端交互环节。
        Args:
            self (AskHumanToolInput): 当前待校验的工具入参模型。
        Returns:
            AskHumanToolInput: 校验通过的原始入参模型。
        """
        has_choices = bool(self.choose_list)
        has_question = bool(self.question and self.question.strip())
        if has_choices == has_question:
            raise ValueError("choose_list 与 question 必须且只能提供一项。")
        if self.choose_list:
            for prompt, options in self.choose_list.items():
                if not prompt.strip() or not options or any(
                    not option.strip() for option in options
                ):
                    raise ValueError("choose_list 的问题和每个选项都不能为空。")
        return self


def _format_ask_human_validation_error(error: ValidationError) -> str:
    """将 ask_human 入参校验异常转换为模型可读反馈。

    Description:
        把模型生成的非法工具参数错误直接返回到 ReAct 消息流，使计划模型能够
        修正参数后重试，而不是使整张图因 Pydantic 异常终止。
    Args:
        error (ValidationError): Pydantic 产生的工具入参校验异常。
    Returns:
        str: 包含错误明细和重试要求的工具执行结果文本。
    """
    return f"ask_human 参数校验失败，请修正参数后重新调用该工具: {error}"


@tool(args_schema=AskHumanToolInput)
def ask_human(
    choose_list: dict[str, list[str]] | None = None,
    question: str | None = None,
) -> str:
    """向用户发起计划阶段澄清问题并返回结构化回答。

    Description:
        根据包含一个或多个问题的 choose_list，或单条 question，自动生成终端
        展示文本，通过 LangGraph interrupt 暂停执行并在恢复后返回用户回答。
    Args:
        choose_list (dict[str, list[str]] | None): 选择题数据，键为问题，值为选项列表。
        question (str | None): 需要用户自由说明的开放问题。
    Returns:
        str: JSON 字符串，包含用户提交的 answer 文本。
    """
    try:
        if choose_list:
            answers: list[tuple[str, str]] = []
            questions = list(choose_list.items())
            for question_index, (prompt, options) in enumerate(questions, start=1):
                display = "\n".join(
                    [
                        prompt,
                        *[
                            f"{index}. {option}"
                            for index, option in enumerate(options, start=1)
                        ],
                    ]
                )
                resume_value = interrupt(
                    {
                        "type": "ask_human",
                        "mode": "choice",
                        "display": display,
                        "question": prompt,
                        "questions": choose_list,
                        "options": options,
                        "current": question_index,
                        "total": len(questions),
                    }
                )

                answer = ""
                if isinstance(resume_value, dict) and isinstance(
                    resume_value.get("answer"), str
                ):
                    answer = resume_value["answer"].strip()
                elif isinstance(resume_value, str):
                    answer = resume_value.strip()
                if answer.isdigit():
                    index = int(answer)
                    if 1 <= index <= len(options):
                        answer = options[index - 1]
                answers.append((prompt, answer or "用户未提供补充信息。"))

            if len(answers) == 1:
                answer = answers[0][1]
            else:
                answer = "；".join(
                    f"{prompt}: {selected_answer}"
                    for prompt, selected_answer in answers
                )
        else:
            display = question or "请补充说明需求。"
            resume_value = interrupt(
                {
                    "type": "ask_human",
                    "mode": "question",
                    "display": display,
                }
            )

            answer = ""
            if isinstance(resume_value, dict) and isinstance(
                resume_value.get("answer"), str
            ):
                answer = resume_value["answer"].strip()
            elif isinstance(resume_value, str):
                answer = resume_value.strip()
        return json.dumps(
            {"type": "ask_human_answer", "answer": answer or "用户未提供补充信息。"},
            ensure_ascii=False,
        )
    except GraphBubbleUp:
        raise
    except Exception as exc:
        return f"ask_human 执行失败，请修正参数或重试: {type(exc).__name__}: {exc}"


ask_human.handle_validation_error = _format_ask_human_validation_error

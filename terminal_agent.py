"""A small conversational terminal interface, ready for a future RAG backend.

The terminal UI deliberately knows nothing about retrieval or LLM providers.
Replace ``ConversationalAgent.respond`` (or subclass it) when a RAG pipeline is
available; it receives the user message and the complete conversation history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import click


@dataclass
class Message:
    """One turn in the conversation."""

    role: str
    content: str
    created_at: datetime = field(default_factory=datetime.now)


class ConversationalAgent:
    """Temporary local response engine with short-term conversational memory.

    A RAG implementation can use ``history`` to formulate a query and then
    return an answer grounded in retrieved documents.
    """

    def respond(self, message: str, history: list[Message]) -> str:
        normalized = message.strip().lower()

        if any(greeting in normalized for greeting in ("hello", "hi", "hey")):
            return "Hello! What would you like to explore?"

        if "what did i" in normalized or "remember" in normalized:
            previous_user_turns = [turn.content for turn in history if turn.role == "user"]
            if previous_user_turns:
                return f"Earlier, you said: “{previous_user_turns[-1]}”"
            return "This is the first thing you have said in this session."

        if normalized.endswith("?"):
            return (
                "I do not have a knowledge source connected yet, but I understand "
                f"your question: “{message}”. A RAG retriever can answer this here."
            )

        return (
            f"I heard: “{message}”. Tell me more, or ask a question. "
            "This reply engine can be replaced with your RAG pipeline later."
        )


HELP_TEXT = "Commands: /help, /clear, /history, /exit (or Ctrl+C/Ctrl+D)"


def print_turn(role: str, content: str) -> None:
    """Render a labelled conversation turn."""

    label = "You" if role == "user" else "🤖 Agent"
    color = "cyan" if role == "user" else "green"
    click.secho(f"{label}: ", fg=color, bold=True, nl=False)
    click.echo(content)


@click.command()
@click.option("--name", default="Agent", show_default=True, help="Name shown in the welcome message.")
def chat(name: str) -> None:
    """Start an interactive terminal conversation."""

    agent = ConversationalAgent()
    history: list[Message] = []

    click.secho(f"{name} is ready.", fg="green", bold=True)
    click.echo("Type a message to talk. " + HELP_TEXT)

    while True:
        try:
            message = click.prompt("You", prompt_suffix="> ").strip()
        except (click.Abort, EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye!")
            return

        if not message:
            continue

        command = message.lower()
        if command in {"/exit", "/quit"}:
            click.echo("Goodbye!")
            return
        if command == "/help":
            click.echo(HELP_TEXT)
            continue
        if command == "/clear":
            history.clear()
            click.secho("Conversation cleared.", fg="yellow")
            continue
        if command == "/history":
            if not history:
                click.echo("No conversation yet.")
            else:
                for turn in history:
                    print_turn(turn.role, turn.content)
            continue

        user_turn = Message(role="user", content=message)
        history.append(user_turn)
        answer = agent.respond(message, history[:-1])
        history.append(Message(role="agent", content=answer))
        print_turn("agent", answer)


if __name__ == "__main__":
    chat()

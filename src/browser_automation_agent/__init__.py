import asyncio

from .agent import main as _agent_main


def main() -> None:
    asyncio.run(_agent_main())

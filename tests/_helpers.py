import asyncio
import inspect


def collect_chunks(model, prompt, stream, response, conversation, key):
    """Run model.execute() and collect all chunks, handling sync/async transparently."""
    if inspect.isasyncgenfunction(model.execute):
        async def _collect():
            chunks = []
            async for chunk in model.execute(
                prompt, stream, response, conversation, key
            ):
                chunks.append(chunk)
            return chunks
        return asyncio.run(_collect())
    return list(model.execute(prompt, stream, response, conversation, key))

from asyncio import sleep as asleep

from src.contexts.repositories import TrainingHistoryRepo


async def train_model():
    async with TrainingHistoryRepo() as repo:
        history = await repo.get_ongoing()
        if history is None:
            return

        print("Training model!")
        await asleep(5)
        print("Model Trained")
        history.finish(0.1)
        repo.add(history)
        await repo.commit()

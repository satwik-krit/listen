from abc import ABC, abstractmethod
from typing import Any, Dict, List
import asyncio
from fastapi import HTTPException, status


class BaseMachinePipeline(ABC):
    def __init__(self, machine_id: str, metadata: Dict[str, Any]):
        self.machine_id = machine_id
        self.metadata = metadata
        self.results = {}

    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """Entry point for the pipeline logic."""
        pass

    def raise_pipeline_error(self, message: str, code: str = "PIPELINE_ERROR"):
        """Standardized error reporting as requested."""
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": code,
                "machine_id": self.machine_id,
                "message": message,
            },
        )

from typing import List, Union

from pydantic import BaseModel, Field


class ExternalAgentStep(BaseModel):
    agent: str = Field(max_length=200)
    action: float
    reward: float
    observation_space: Union[List[float], List[List[float]]]


class ExternalStep(BaseModel):
    number: int = Field(ge=0)
    image: bytes | None = None
    agent_steps: List[ExternalAgentStep] | None = None


class ExternalInference(BaseModel):
    parallel: bool = Field(default=False)
    config: dict = Field(default_factory=dict)
    steps: List[ExternalStep] = Field(min_length=1)


class ExternalTraining(BaseModel):
    environment: str = Field(max_length=200)
    algorithm: str = Field(max_length=200)
    parallel: bool = Field(default=False)
    num_gpus: float = Field(ge=0.0, default=0.0)
    iterations: int = Field(ge=1)
    config: dict = Field(default_factory=dict)


class ExternalImport(BaseModel):
    training: ExternalTraining
    inference: ExternalInference

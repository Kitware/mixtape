from pydantic import BaseModel, Field, field_validator


class ExternalAgentStep(BaseModel):
    agent: str = Field(max_length=200)
    action: float
    reward: float
    observation_space: list[float] | list[list[float]]


class ExternalStep(BaseModel):
    number: int = Field(ge=0)
    image: bytes | None = None
    agent_steps: list[ExternalAgentStep] | None = None


class ExternalInference(BaseModel):
    parallel: bool = Field(default=False)
    config: dict = Field(default_factory=dict)
    steps: list[ExternalStep] = Field(min_length=1)


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
    action_mapping: dict[int, str] | None = None

    @field_validator('action_mapping')
    @classmethod
    def validate_action_mapping(cls, v: dict[int, str] | None) -> dict[int, str] | None:
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError('Mapping must be a dictionary')

        for action, label in v.items():
            if not isinstance(action, int):
                raise ValueError('Action keys must be integers')
            if not isinstance(label, str):
                raise ValueError('Action labels must be strings')

        return v

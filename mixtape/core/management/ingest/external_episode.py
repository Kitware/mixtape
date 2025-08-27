from io import BytesIO

from PIL import Image
from pydantic import Base64Bytes, BaseModel, Field, field_validator, model_validator


class ExternalUnitStep(BaseModel):
    unit: str = Field(max_length=200)
    action: float
    # Support environments that use either single reward or multiple rewards
    reward: float | None = None
    rewards: list[float | int] | None = None
    action_distribution: list[float] | None = None
    health: dict | None = None
    value_estimate: float | None = None
    predicted_reward: float | None = None
    custom_metrics: dict | None = None

    @field_validator('rewards')
    @classmethod
    def validate_rewards_list(cls, v: list[float] | None) -> list[float] | None:
        if v is not None:
            if len(v) == 0:
                raise ValueError('Rewards list cannot be empty')
        return v

    @model_validator(mode='after')
    def validate_reward_fields(self) -> 'ExternalUnitStep':
        reward_defined = self.reward is not None
        rewards_defined = self.rewards is not None

        if reward_defined and rewards_defined:
            raise ValueError('Cannot define both reward and rewards - use only one')

        return self


class ExternalAgentStep(BaseModel):
    agent: str = Field(max_length=200)
    action: float
    # Support environments that use either single reward or multiple rewards
    reward: float | None = None
    rewards: list[float | int] | None = None
    observation_space: list[float] | list[list[float]]
    action_distribution: list[float] | None = None
    health: dict | None = None
    value_estimate: float | None = None
    predicted_reward: float | None = None
    custom_metrics: dict | None = None
    unit_steps: list[ExternalUnitStep] | None = None

    @field_validator('rewards')
    @classmethod
    def validate_rewards_list(cls, v: list[float] | None) -> list[float] | None:
        if v is not None:
            if len(v) == 0:
                raise ValueError('Rewards list cannot be empty')
        return v

    @model_validator(mode='after')
    def validate_reward_fields(self) -> 'ExternalAgentStep':
        reward_defined = self.reward is not None
        rewards_defined = self.rewards is not None

        if not reward_defined and not rewards_defined:
            raise ValueError('Either reward or rewards must be defined')

        if reward_defined and rewards_defined:
            raise ValueError('Cannot define both reward and rewards - use only one')

        return self


class ExternalStep(BaseModel):
    number: int = Field(ge=0)
    image: Base64Bytes | None = None
    agent_steps: list[ExternalAgentStep] | None = None

    @field_validator('image')
    @classmethod
    def validate_image(cls, v: Base64Bytes | None) -> Base64Bytes | None:
        if v is None:
            return v

        try:
            # Try to open decoded bytes as an image
            Image.open(BytesIO(v))
        except Exception as e:
            raise ValueError(f'Invalid image data: {str(e)}')
        return v


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
    reward_mapping: list[str] | None = None

    @field_validator('reward_mapping')
    @classmethod
    def validate_reward_mapping(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            if not isinstance(v, list):
                raise ValueError('Reward mapping must be a list')
            if len(v) == 0:
                raise ValueError('Reward mapping cannot be empty')
            for label in v:
                if not isinstance(label, str):
                    raise ValueError('All reward mapping labels must be strings')
        return v


class ExternalImport(BaseModel):
    training: ExternalTraining
    inference: ExternalInference
    action_mapping: dict | None = None

    @field_validator('action_mapping')
    @classmethod
    def validate_action_mapping(cls, v: dict | None) -> dict | None:
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError('Action mapping must be a dictionary')

        # Check top-level action mappings (excluding unit_mapping)
        for key, value in v.items():
            if key == "unit_mapping":
                continue

            try:
                # Try to convert key to int for action keys
                int(key)
            except (ValueError, TypeError):
                raise ValueError('Action keys must be convertible to integers')

            if not isinstance(value, str):
                raise ValueError('Action labels must be strings')

        # Check unit_mapping if present
        if "unit_mapping" in v:
            unit_mapping = v["unit_mapping"]
            if not isinstance(unit_mapping, dict):
                raise ValueError('Unit mapping must be a dictionary')

            for key, value in unit_mapping.items():
                try:
                    # Try to convert key to int for unit action keys
                    int(key)
                except (ValueError, TypeError):
                    raise ValueError('Unit action keys must be convertible to integers')

                if not isinstance(value, str):
                    raise ValueError('Unit action labels must be strings')

        return v

    @field_validator('inference')
    @classmethod
    def validate_reward_consistency(cls, v: ExternalInference) -> ExternalInference:
        """Validate that all agent steps in the episode use the same reward structure."""
        if not v.steps:
            return v

        # Check if any step has agent steps
        has_agent_steps = any(step.agent_steps for step in v.steps)
        if not has_agent_steps:
            return v

        # Determine the reward structure from the first agent step
        first_agent_step = None
        for step in v.steps:
            if step.agent_steps:
                first_agent_step = step.agent_steps[0]
                break

        if not first_agent_step:
            return v

        # Check that all agent steps follow the same pattern
        uses_rewards = first_agent_step.rewards is not None
        uses_reward = first_agent_step.reward is not None

        for step in v.steps:
            if step.agent_steps:
                for agent_step in step.agent_steps:
                    if (agent_step.rewards is not None) != uses_rewards:
                        raise ValueError(
                            'All agent steps in an episode must use the same reward structure '
                            '(either all use "reward" or all use "rewards")'
                        )
                    if (agent_step.reward is not None) != uses_reward:
                        raise ValueError(
                            'All agent steps in an episode must use the same reward structure '
                            '(either all use "reward" or all use "rewards")'
                        )

        return v

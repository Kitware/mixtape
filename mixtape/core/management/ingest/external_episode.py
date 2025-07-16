from io import BytesIO

from PIL import Image
from pydantic import Base64Bytes, BaseModel, Field, field_validator


class ExternalAgentStep(BaseModel):
    agent: str = Field(max_length=200)
    action: float
    # Support environments that use either single reward or multiple rewards
    reward: float | None = None
    rewards: list[float] | None = None
    observation_space: list[float] | list[list[float]]

    @field_validator('rewards')
    @classmethod
    def validate_rewards_list(cls, v: list[float] | None) -> list[float] | None:
        if v is not None:
            if not isinstance(v, list):
                raise ValueError('Rewards must be a list')
            if len(v) == 0:
                raise ValueError('Rewards list cannot be empty')
            for reward in v:
                if not isinstance(reward, (int, float)):
                    raise ValueError('All rewards must be float or integer values')
        return v


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

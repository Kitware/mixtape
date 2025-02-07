from typing import Any

from PIL import Image

from mixtape.core.ray_utils.logger import Logger


class InferenceLoggingCallbacks:
    def __init__(self, env: Any) -> None:
        self.user_data: dict[str, Any] = {}
        self.step = 0
        self.logger = Logger()
        self.env = env

    def on_begin_inference(self):
        self.user_data['frame_list'] = []
        self.user_data['step_data'] = {}
        self.user_data['step_data']['total_reward'] = 0

    def on_compute_action(
        self,
        actions: dict[str, float],
        rewards: dict[str, float],
        obss: dict[str, Any],
    ) -> None:
        data = self.user_data['step_data']
        data.setdefault(self.step, {'actions': {}, 'rewards': {}, 'obss': {}})

        for agent in actions.keys():
            data[self.step]['actions'][agent] = actions[agent]
        for agent in rewards.keys():
            data[self.step]['rewards'][agent] = rewards[agent]
            data['total_reward'] += rewards[agent]
        for agent in obss.keys():
            data[self.step]['obss'][agent] = obss[agent]

        img = Image.fromarray(self.env.render())
        self.user_data['frame_list'].append(img)

        self.step += 1

    def on_complete_inference(self, env_name: str, parallel: bool = True) -> None:
        self.logger.write_to_log(
            f'{"parallel" if parallel else "aec"}_{env_name}_inference.json',
            self.user_data['step_data'],
        )
        gif_file = (
            f'{self.logger.log_path}/{"parallel" if parallel else "aec"}_{env_name}_inference.gif'
        )
        frame_list = self.user_data['frame_list']
        frame_list[0].save(
            gif_file, save_all=True, append_images=frame_list[1:], duration=3, loop=0
        )

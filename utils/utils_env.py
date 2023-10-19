import gym
import copy


def safe_deepcopy_env(obj):
    """
    Perform a deep copy of an environment but without copying its viewer.
    - adopted from `https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/common/factory.py`
    """
    cls = obj.__class__
    result = cls.__new__(cls)
    memo = {id(obj): result}
    for k, v in obj.__dict__.items():
        if k not in ['viewer', 'automatic_rendering_callback', 'automatic_record_callback', 'grid_render']:
            if isinstance(v, gym.Env):
                setattr(result, k, safe_deepcopy_env(v))
            else:
                setattr(result, k, copy.deepcopy(v, memo=memo))
        else:
            setattr(result, k, None)
    return result

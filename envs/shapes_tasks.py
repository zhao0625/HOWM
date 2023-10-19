import gym

import copy
import numpy as np

# TODO - import directly
import envs


class ShapesTaskCreator:
    def __new__(cls, env_id=None, task=None, verbose=True, **kwargs):
        """
        Parameters:
            env_id: the env to be wrapped.
            task: name of task
            kwargs: all remaining parameters treated as kwargs to the env
        """

        if verbose:
            print('[Debug: other environment arguments]', kwargs)

        # TODO [specify the env_id for Shapes environment and if enabling Object Library]
        assert env_id is not None, '[env-id must be specified!]'
        env = gym.make(env_id, **kwargs)  # [other parameters passed to gym.make directly]

        # TODO [specify task wrapper]
        assert task in ['avoiding', 'reaching']
        if task == 'avoiding':
            return ShapesAvoidingWrapper(env=env)
        elif task == 'reaching':
            return ShapesReachingWrapper(env=env, **kwargs)
        else:
            raise ValueError('[non-existing task!]')


class ShapesAvoidingWrapper(gym.Wrapper):
    """
    a wrapper for shapes avoiding tasks
    - +1 reward for non-collision moves and 0 for collision actions
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        return self.env.reset()

    def step(self, action, render=True):
        _obs, _reward, _done, _info = self.env.step(action)
        assert _reward == 0  # check - original env doesn't have reward

        # +1 reward for avoiding other objects
        _reward = 1 if _info['success'] else 0
        return _obs, _reward, _done, _info


class ShapesReachingWrapper(gym.Wrapper):
    """
    a basic goal-reaching wrapper (for Shapes) without considering object permutations / shapes / ...
    """

    def __init__(self, env, goal_state=None, goal_mode='step', goal_step=10, **kwargs):
        super().__init__(env)
        self.goal_state, self.goal_mode, self.goal_step = goal_state, goal_mode, goal_step

    def _init_goal(self):
        if self.goal_state is None:
            self.goal_state, self.goal_locations = self.get_goal(
                env=self.env, goal_mode=self.goal_mode, goal_step=self.goal_step)
        else:
            # should be a valid (reachable) object state
            # self.goal_state = goal_state
            pass

    def reset(self):
        """
        reset the internal environment and then initialize goal
        """
        obs = self.env.reset()
        self._init_goal()  # initialize goal after resetting (will randomly sample locations)
        return obs

    def step(self, action):
        _obs, _reward, _done, _info = self.env.step(action)
        assert _reward == 0  # check - original env doesn't have reward
        _reward = -1  # set penalty to encourage finishing faster

        # terminate if goal state reached (or time limit reached)
        assert _obs[0].shape == self.goal_state.shape
        if np.all(_obs[0] == self.goal_state):
            _done = True
            _reward = 0

        return _obs, _reward, _done, _info

    @staticmethod
    def get_goal(goal_mode='step', goal_step=None, env=None):
        assert goal_mode in ['step', 'rand']
        if goal_mode == 'step':
            assert env is not None and goal_step is not None
            env_goal = copy.deepcopy(env)  # copy to avoid changing its state
            env_goal.reset()

            env_goal.set_objects(copy.deepcopy(env.objects))  # internal np state needs to be deepcopy!
            assert env_goal.unwrapped.objects == env.unwrapped.objects

            for i in range(goal_step):
                # [generate random actions for each object periodically]
                _action = (i % env_goal.num_objects) * 4 + np.random.randint(4)  # env_goal.action_space.sample()
                assert _action in env_goal.action_space
                env_goal.step(_action)
                print('[random moving]', env_goal.unwrapped.objects, _action)

            goal_state = env_goal.unwrapped.get_state()
            goal_locations = env_goal.unwrapped.objects

        elif goal_mode == 'rand':
            # [to get a random state as goal state]
            env_goal = gym.make('ShapesTrain-v0')  # TODO or `ShapesTrain-v3` for object library
            env_goal.reset()
            goal_state = env_goal.unwrapped.get_state()
            goal_locations = env_goal.unwrapped.objects

        else:
            return ValueError

        return goal_state, goal_locations


class ShapesMovingWrapper:
    """
    A task wrapper for Shapes with object library
    - The goal is to reach goal configurations satisfying given predicates
    - This should be easier than reaching a single goal state
    """
    def __init__(self):
        pass



if __name__ == '__main__':
    # from environments.shapes_cswm.gym_shapes.envs.block_pushing import BlockPushing
    # from gym.wrappers.time_limit import TimeLimit
    # # env = gym.make('ShapesTrain-v0')
    # env = BlockPushing(render_type='shapes_train')  # to env
    # env = TimeLimit(env, max_episode_steps=100)
    # env = ShapesReachingWrapper(env)
    #
    # env.reset()
    #
    # for _ in range(10):
    #     obs, reward, done, info = env.step(env.action_space.sample())
    #     print(reward, done)

    # TODO creator
    env = ShapesTaskCreator(
        env_id='ShapesLibraryTrain-v0',
        task='reaching',
        num_objects=3,
        goal_step=3,
    )
    env.reset()
    # print('env reset', env.reset()[0])

    print('obj location', env.unwrapped.objects)
    # print('env.goal_state:', env.goal_state)
    print('env.goal_locations:', env.goal_locations)

    for _ in range(10):
        action = input('action? ')
        assert int(action) in env.action_space
        print(type(action))

        obs, reward, done, _ = env.step(int(action))
        print(env.objects, reward, done)

        if done:
            print('terminate!', done)
            break

from gym.envs.registration import register



register(
    'ObjectLibraryTrainObjectConfig-v0',
    entry_point='envs.block_pushing:BlockPushingObjectConfig',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes_train'},
)
register(
    'ObjectLibraryEvalObjectConfig-v0',
    entry_point='envs.block_pushing:BlockPushingObjectConfig',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes_eval'},
)




# TODO - task wrapping creator
register(
    'ShapesTaskCreator-v0',
    entry_point='envs.shapes_tasks:ShapesTaskCreator',
    max_episode_steps=100,
    # TODO - default task wrapper arguments
    kwargs={
        'env_id': 'ShapesLibraryTrain-v0',
        'task': 'avoiding',
        # (other arguments) passed to wrapped env directly
        'render_type': 'shapes_train',
        'goal_step': 5,
    },
)

# TODO - move new environments here
register(
    'ShapesLibraryTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes_train'},
)

register(
    'ShapesLibraryEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes_eval'},
)

register(
    'ShapesLibraryTrainRushHours-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes_train', 'rush_hours': 'rush_hours'},
)
register(
    'ShapesLibraryEvalRushHours-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes_eval', 'rush_hours': 'rush_hours'},
)





register(
    'ShapesLibraryTrainAdversarialAllRandom-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes_train', 'adversarial': 'all_random'},
)
register(
    'ShapesLibraryEvalAdversarialAllRandom-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes_eval', 'adversarial': 'all_random'},
)

register(
    'ShapesLibraryTrainAdversarialHalfRandom-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes_train', 'adversarial': 'half_random'},
)
register(
    'ShapesLibraryEvalAdversarialHalfRandom-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes_eval', 'adversarial': 'half_random'},
)

register(
    'ShapesLibraryTrainAdversarialHalfConsistent-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes_train', 'adversarial': 'half_consistent_permute'},
)
register(
    'ShapesLibraryEvalAdversarialHalfConsistent-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes_eval', 'adversarial': 'half_consistent_permute'},
)



# TODO - original Shapes environment (need to merge)
# register(
#     'ShapesTrain-v0',
#     entry_point='envs.block_pushing:BlockPushing',
#     max_episode_steps=100,
#     kwargs={'render_type': 'shapes'},
# )
#
# register(
#     'ShapesEval-v0',
#     entry_point='envs.block_pushing:BlockPushing',
#     max_episode_steps=10,
#     kwargs={'render_type': 'shapes'},
# )

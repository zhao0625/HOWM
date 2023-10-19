# Data Generation

## Run

    python -m gen_data.run_data_gen \
    gen_env='Shapes' \
    data_shapes.envid=ObjectLibraryTrainObjectConfig-v0 \
    data_shapes.fname='datasets/shapes_library_n10k5_train_N-config_neg0.5diff_shuffle_debug1.h5' \
    data_shapes.num_episodes=100 \
    data_shapes.action_mapping=True \
    data_shapes.num_object_total=10 \
    data_shapes.shuffle_color=True

## Notes

- `shuffle_color` is useful for large N. It's enabled by default now.
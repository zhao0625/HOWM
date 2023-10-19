# ICML 2022: Toward Compositional Generalization in Object-Oriented World Modeling

This is the official codebase for the following paper, which includes the imlementation on Homomorphic Object-oriented World Model (**HOWM**).

### **Toward Compositional Generalization in Object‑Oriented World Modeling**
- **ICML 2022, Long Presentation (2%)**
- Linfeng Zhao, Lingzhi Kong, Robin Walters, Lawson L.S. Wong
- [[arXiv]](https://arxiv.org/abs/2204.13661) [[Poster]](https://lfzhao.com/poster/poster-oowm-icml2022.pdf) [[Slides (ICML Oral)]](https://lfzhao.com/slides/slides-oowm-icml2022-oral.pdf) [[ICML page]](https://icml.cc/virtual/2022/oral/18212)

## Install required packages (Python 3.6) 
    pip install -r requirements.txt


## Step 1: Generate data
The script and default config file are located in `./gen_data` folder. Run the following commands under root folder to generate data for basic `Shapes` environment (with all shapes and absolute-orientation actions: north, south, west, and east) and `Rush Hour` environment (with only triangles relative-and orientation actions: forward, backward, left, right). 

- We use the block pushing environments, which has two tasks: Shapes and Rush Hour. Shapes task has all different shapes that we can push, while Rush Hour only has
- A key concept in our work is Object Library. It is like a vocabulary in natural languages, which contains all possible objects appeared in each scene. We first generate an object library $\mathbb{L}$ with $|\mathbb{L}| = N$ objects, and then sample different scenes $\mathbb{O}_i$ with a fixed number of objects $|\mathbb{O}_i| = K$.
- We sample $100$ scenes, and then collect episodes or transitions from those scenes. We sample a fixed number of episodes for each scene $n=100$ with a fixed number of length $l=100$.
    - TODO: check number


### Basic Shapes environment (saving pickle file for persistent object library)

    python -m gen_data.run_data_gen \
    gen_env='Shapes' \
    gen_mode='joint' \
    config_shapes.data_folder='./datasets' \
    config_shapes.data_prefix='shapes_library_100train1eval' \
    config_shapes.shapes='all' \
    config_shapes.cascade_gen=True \
    config_shapes.num_objects_total_list='[5, 10, 15, 20, 30, 40, 50]' \
    config_shapes.num_objects_scene=5 \
    config_shapes.shuffle_color=True \
    config_shapes.num_episodes.train=1000 \
    config_shapes.num_episodes.eval=10000 \
    config_shapes.num_config_eval=1 \
    config_shapes.num_config_train=100 \
    config_shapes.pickle_library=True \
    config_shapes.load_pickle_library=False \
    config_shapes.load_pickle_file=None

### Rush Hour environment

    python -m gen_data.run_data_gen \
    gen_env='Shapes' \
    gen_mode='joint' \
    config_shapes.data_folder='./datasets' \
    config_shapes.data_prefix='rush-hour_library_100train1eval' \
    config_shapes.shapes='rush_hour' \
    config_shapes.cascade_gen=True \
    config_shapes.num_objects_total_list='[5, 10, 15, 20, 30, 40, 50]' \
    config_shapes.num_objects_scene=5 \
    config_shapes.shuffle_color=True \
    config_shapes.num_episodes.train=1000 \
    config_shapes.num_episodes.eval=10000 \
    config_shapes.num_config_eval=1 \
    config_shapes.num_config_train=100 \
    config_shapes.pickle_library=True \
    config_shapes.load_pickle_library=False \
    config_shapes.load_pickle_file=None

## Step 2: Training a representation module checkpoint

- Overview
    - For learning object representations (unsupervised object discovery), we use Slot Attention (Locatello, NeurIPS’20) and freeze the network.
    - The most important characteristic is that objects do not naturally have canonical ordering. Thus, when we train the downstream model, such as a transition model, it needs to align objects between different steps in order to correctly compute the prediction error and other quantities.
- Implementation
    - The code of Slot Attention is adopted from an open-source PyTorch implementation (https://github.com/untitled-ai/slot_attention), where the training is done by PyTorch Lightning.
    - It will train a separate network and save a checkpoint, which will be later freeze in the training of the transition model.
- Training strategy
    - Since we assume we have an object library $\mathbb{L}$ with $|\mathbb{L}| = N$ objects (e.g., $N=10$), even though each scene has $K$ objects (e.g., $K=5$), we only need to train one representation module on all $N$ objects.
    - The model is trained to reconstruct all objects using mean-square reconstruction loss in the pixel space.
    - Empirically, we found this strategy works well: the trained model can accurately reconstruct al objects in the library even with $N = 50$ objects.


#### Command

    python -m scripts.run -p -l INFO train_separately with config_model \
    enable_wandb=True watch_model=True \
    model_train.decoupled_homo_att=True \
    train_components='r' \
    enable_pl=True \
    use_obj_config_dataset=True \
    model_train.dataset='datasets/shapes_library_100train1eval_jan24_cascade_n50k5_train.h5' \
    model_eval.dataset='datasets/shapes_library_100train1eval_jan24_cascade_n50k5_eval.h5' \
    model_train.num_objects=5 \
    model_train.num_objects_total=50 \
    model_train.encoder_type='specific-small2' \
    model_train.embedding_dim=4 \
    model_train.slot_size=16 \
    model_train.hidden_dims_encoder='(32,32,16)' \
    model_train.representation.epochs=200 \
    model_train.transition.epochs=0 \
    representation_checkpoint=None \
    stats_collection='homo_slot_att_debug' \
    save_folder='checkpoints/decoupled-homo-slot-attention-experiments' \
    description='train: representation'

#### Command (previous)

    python -m scripts.run -p -l INFO train with config_model \
    model_train.dataset='datasets/shapes_library_n10k5_train_debug_config_unlimited.h5' \
    model_eval.dataset='datasets/shapes_library_n10k5_eval_debug_config_unlimited.h5' \
    model_train.num_objects=5 \
    model_train.num_objects_total=10 \
    model_train.epochs=1 \
    save_folder='checkpoints/homo-slot-attention-experiments' \
    model_train.homo_slot_att=True \
    model_train.decoupled_homo_att=False \
    model_train.batch_size=1024 \
    model_train.learning_rate=3e-3 \
    model_train.encoder_type=specific \
    model_train.embedding_dim=4 \
    model_train.num_iterations=3 \
    num_training_episodes=1000 \
    data_config=config_unlimited_neg0.5diff \
    model_train.same_config_ratio=0.5 \
    stats_collection='homo_slot_att_debug' \
    description='debug - config + recon'


## Step 3: learn transition model with trained representation module 

The last step should give you a checkpoint of representation module.
Use that as the input to the "representation_checkpoint" argument, and run the following command.

**TODO: Finish this.**

Note that the last step should use N=50 dataset, but here we only train for smaller N, such as N=10,20,30. So you would need to use the corresponding dataset and change `model_train.num_objects_total` accordingly.


#### Command

    python -m scripts.run -p -l INFO train_separately with config_model \
    enable_wandb=True watch_model=True \
    model_train.decoupled_homo_att=True \
    train_components='t' \
    enable_pl=False \
    use_obj_config_dataset=True \
    model_train.dataset='datasets/shapes_library_100train1eval_jan24_cascade_n50k5_train.h5' \
    model_eval.dataset='datasets/shapes_library_100train1eval_jan24_cascade_n50k5_eval.h5' \
    model_train.num_objects=5 \
    model_train.num_objects_total=20 \
    model_train.encoder_type='specific-small2' \
    model_train.embedding_dim=4 \
    model_train.slot_size=16 \
    model_train.hidden_dims_encoder='(32,32,16)' \
    model_train.representation.epochs=0 \
    model_train.transition.epochs=100 \
    dir_wandb='/mnt_host/zlf-local-data' \
    stats_collection='homo_slot_att_debug' \
    save_folder='checkpoints' \
    representation_checkpoint=<PATH TO YOUR TRAINED REPRESENTATION MODULE> \
    description='train: transition with representation checkpoint'


## Structure

- Environments (Static/Actionable) - `envs/`
- Data Generation - `gen_data/`
- Algorithms - `algorithms/`
- Utilities - `util`
- Running entrance scripts - `runs/`
- Configuration files - `Config/`


## TODO
- [ ] Update the readme
- [ ] Add a notebook to run the entire pipeline and visualize the attention binding matrices
- [ ] Upload pretrained representation and transition model checkpoints
- [ ] Reduce memory use in evaluating the contrastive transition model
- [ ] Double-check an attention matrix use in loss function

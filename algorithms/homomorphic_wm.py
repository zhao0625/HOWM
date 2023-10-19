import copy

import numpy as np
import torch
import torchvision
from torch import nn

import utils.utils_dataset as utils
import utils.utils_func
from algorithms.modules_attention import ActionBindingAttention, DecoupledSlotAttentionModel
from algorithms.modules_encoders_temp import TransitionMLP, TransitionGNN


class DecoupledHomomorphicWM(nn.Module):
    """
    Our Homomorphic Object-oriented World Model (HOWM)
    The training of object representation module (Slot Attention) and transition module (Action Attention)
        is decoupled.
    The training of object representation uses reconstruction loss, while the contrastive loss uses contrastive loss.
    """

    def __init__(self, embedding_dim, hidden_dim, action_dim,
                 num_objects, num_objects_total,
                 num_iterations,
                 slot_size,
                 kernel_size, hidden_dims_encoder, action_hidden_dims,
                 encoder_type,
                 encoder_batch_norm,
                 no_last_slot,
                 transition_type,

                 first_kernel_size,
                 input_resolution,

                 action_encoding=False,
                 identity_encoding=False,

                 hinge=1., sigma=0.5,
                 ignore_action=False, copy_action=False,
                 **kwargs
                 ):
        super().__init__()

        print('>>> kwargs - extra', kwargs.keys())

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.hinge = hinge
        self.sigma = sigma
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        self.pos_loss = 0
        self.neg_loss = 0

        self.slot_attention = DecoupledSlotAttentionModel(
            input_resolution=input_resolution,
            num_slots=num_objects + 1,

            num_objects=num_objects,
            num_objects_total=num_objects_total,
            num_iterations=num_iterations,
            kernel_size=kernel_size,
            slot_size=slot_size,
            hidden_dims=hidden_dims_encoder,
            first_kernel_size=first_kernel_size,
            action_hidden_dims=action_hidden_dims,
            encoder_type=encoder_type,
            encoder_batch_norm=encoder_batch_norm,
            enable_decoder=True,
        )

        # >>> Output layer after slot attention to map to lower dimension
        self.pixel_slots_out = nn.Linear(slot_size, embedding_dim)

        if action_encoding:
            self.latent_action_dim = action_dim + 1  # > Don't confuse
            self.action_slots_out = nn.Linear(slot_size, self.latent_action_dim)
        else:
            self.latent_action_dim = action_dim
            self.action_slots_out = nn.Identity()

        # >>> Handle extra +1 slot (for background)
        num_slots = num_objects if no_last_slot else (num_objects + 1)
        self.no_last_slot = no_last_slot
        self.num_slots = num_slots  # FIXME without extra slot would cause error

        # > Init transition net, GNN or MLP
        if transition_type == 'gnn':
            transition_class = TransitionGNN
        elif transition_type == 'mlp':
            transition_class = TransitionMLP
        else:
            raise ValueError("[transition_net doesn't exist]")

        self.transition_model = transition_class(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=self.latent_action_dim,  # >>> latent action!
            num_objects=num_slots,  # taking slots
            ignore_action=ignore_action,
            copy_action=copy_action)
        print('[transition net]', transition_class, self.transition_model)

        # >>> Init action attention
        action_features = action_hidden_dims[-1]

        self.num_objects_total = num_objects_total

        self.action_attention = ActionBindingAttention(
            num_slots=num_slots,
            slot_size=slot_size,
            num_objects_total=num_objects_total,

            action_encoding=action_encoding,
            identity_encoding=identity_encoding,

            action_size=action_dim,
            action_features=action_features,
        )

    def get_representation_module(self):
        modules = [
            self.slot_attention
        ]
        return torch.nn.ModuleList(modules)

    def get_transition_module(self):
        modules = [
            self.transition_model,
            self.pixel_slots_out,
            self.action_slots_out,
            self.action_attention,
        ]
        return torch.nn.ModuleList(modules)

    def get_representation_params(self, filter_grad=False):
        # Note: this should only include representation module and shouldn't need to further filter
        if filter_grad:
            return filter(lambda p: p.requires_grad, self.get_representation_module().parameters())
        else:
            return self.get_representation_module().parameters()

    def get_transition_params(self, filter_grad=False):
        if filter_grad:
            return filter(lambda p: p.requires_grad, self.get_transition_module().parameters())
        else:
            return self.get_transition_module().parameters()

    def freeze_representation_params(self):
        for param in self.get_representation_params():
            param.requires_grad = False

    def energy(self, state, action, next_state,
               action_attention, next_action_attention,
               no_trans=False, pseudo_inverse_loss=False):
        """Energy function based on normalized squared L2 norm."""

        # >>> action_attention: [batch_size, N, K], state: [batch_size, K, D]

        norm = 0.5 / (self.sigma ** 2)

        # > add normalization given N - disabled
        # norm = norm / self.num_objects_total

        if pseudo_inverse_loss:
            action_attention = self.get_pseudo_inverse(action_attention)
            next_action_attention = self.get_pseudo_inverse(next_action_attention)

        if no_trans:
            # >>> Compute loss in the projected N-object full MDP
            diff = (action_attention.bmm(state) - next_action_attention.bmm(next_state))

        else:
            pred_trans = self.transition_model(state, action)

            # >>> Compute loss in the N-object full MDP, equivalently: torch.einsum('bnk,bkd->bnd')
            diff = (action_attention.bmm(state + pred_trans) - next_action_attention.bmm(next_state))

        return norm * diff.pow(2).sum(2).mean(1)

    def get_slots(self, obs, no_grad=False, grad_hook=False):
        if no_grad:
            with torch.no_grad():
                slots = self.slot_attention.encode(obs)
        else:
            slots = self.slot_attention.encode(obs)

        if grad_hook:
            slots.register_hook(lambda grad: print('[grad of slots is called!]'))

        return slots

    def decode_slots(self, slots, no_grad=False):
        if no_grad:
            with torch.no_grad():
                recon_combined, recons, masks, _ = self.slot_attention.decode(slots)
        else:
            recon_combined, recons, masks, _ = self.slot_attention.decode(slots)
        return recon_combined, recons, masks, None

    def action2oh(self, action):
        """
        convert action id to one-hot action
        """
        # [convert to one-hot actions of N objects]
        action_vec = utils.utils_func.to_one_hot(action, self.action_dim * self.num_objects_total)
        # [reshape ground actions to (B x) N x 4]
        action_vec = action_vec.view(len(action), self.num_objects_total, self.action_dim)

        return action_vec

    def get_encoding(self, obs, action, no_grad=False, out_enc=False):
        """
        Encode obs (to slots) and action (to action slots)
        Args:
            out_enc: go through output layer before return
        """
        action_vec = self.action2oh(action)

        # >>> Encode objects in pixels to slots
        slots = self.get_slots(obs, no_grad=no_grad)  # just use the inner layer
        # slots: [batch_size, num_slots, slot_size]

        # >>> Encode actions from slots
        action_slots, action_attention = self.action_attention(slots=slots, actions=action_vec)

        # > Include an extra slot or not: action_attention = [batch_size, N, K] or [batch_size, N, K + 1]
        if self.no_last_slot:
            slots, action_slots, action_attention = (
                slots[:, :-1].contiguous(), action_slots[:, :-1].contiguous(), action_attention[:, :, :-1].contiguous()
            )

        if out_enc:
            latent_state = self.pixel_slots_out(slots)
            latent_action = self.action_slots_out(action_slots)
            assert latent_state.size(-1) == self.embedding_dim
            assert latent_action.size(-1) == self.latent_action_dim
            return latent_state, latent_action, action_attention
        else:
            return slots, action_slots, action_attention

    def get_align_action(self, action, action_attention, out_enc=True):
        """
        Input: (1) ground action @ t, (2) action-attention / binding matrix M_t, output: bound action
        Note: action should always use soft binding, because it's input to the transition model!
        Args:
            action: int action id
            action_attention: desired binding matrix
            out_enc: use action slot output layer
        """

        # > For single data
        if action.dim() == 0:  # > becomes scalar after indexing
            action = action.unsqueeze(0)
        if action_attention.dim() == 2:
            action_attention = action_attention.unsqueeze(0)

        action_vec = self.action2oh(action)
        action_slots = self.action_attention.transform_action(action=action_vec, action_attention=action_attention)

        if out_enc:
            return self.action_slots_out(action_slots)
        else:
            return action_slots

    def lift_slot2full(self, slot_state, attention, hard_bind=False, pseudo_inverse=False):
        """
        Lift from slot MDP to full MDP
        Args:
            slot_state: embedding in slot MDP (Note: after output layer)
            hard_bind: use hard max for deterministic binding
            pseudo_inverse: use MP pseudo-inverse of the binding matrix
        """
        if hard_bind:
            attention = self.get_hard_binding(attention)
        if pseudo_inverse:
            attention = self.get_pseudo_inverse(attention)

        # > [N, K] x [K, D] = [N, D]
        if slot_state.dim() == 2 and attention.dim() == 2:
            full_state = attention @ slot_state
        elif slot_state.dim() == 3 and attention.dim() == 3:
            full_state = attention.bmm(slot_state)
        else:
            raise ValueError('Unmatched dimension')

        assert full_state.size(-2) == self.num_objects_total
        return full_state

    def project_full2slot(self, full_state, attention, hard_bind=False):
        """
        Project from full MDP to slot MDP
        Args:
            full_state: embedding in full state (should from lifted state)
            hard_bind: use hard max for deterministic binding
        """
        if hard_bind:
            attention = self.get_hard_binding(attention)

        # > [K, N] x [N, D] = [K, D]
        if full_state.dim() == 2 and attention.dim() == 2:
            slot_state = attention.t().contiguous() @ full_state
        elif full_state.dim() == 3 and attention.dim() == 3:
            slot_state = attention.permute(0, 2, 1).bmm(full_state)
        else:
            raise ValueError('Unmatched dimension')

        assert slot_state.size(-2) == self.num_slots
        return slot_state

    def align_state(self, pred_state, target_state, pred_attention, target_attention, space,
                    hard_bind=False, pseudo_inverse=False):
        """
        Get predicted state aligned in full or slot MDP using given attention (soft binding)
        Args:
            pred_state: the state to be aligned
            target_state: the state to align to (for slot MDP case)
            pred_attention: soft binding info for predict state
            target_attention: soft binding info for target state, used in projecting to slot MDP
            space: choose the space/MDP to compute pair-wise distance between latent states
            hard_bind: use hard attention for binding
            pseudo_inverse: use MP pseudo-inverse (intuitively, normalize slots by number of bound objects)
        """

        if hard_bind:
            pred_attention = self.get_hard_binding(pred_attention)
            target_attention = self.get_hard_binding(target_attention)

        # >>> Option 1: transform both to the N-object canonical full MDP - flatten later
        if space == 'full-mdp':
            pred_state = self.lift_slot2full(pred_state, pred_attention, False, pseudo_inverse)
            target_state = self.lift_slot2full(target_state, target_attention, False, pseudo_inverse)

        # TODO check
        # >>> Option 2: transform to slot-mdp
        elif space == 'slot-mdp':
            target_state = self.lift_slot2full(target_state, pred_attention, False, pseudo_inverse)
            target_state = self.project_full2slot(target_state, target_attention, False)

            # > If not use pseudo_inverse, we need to make sure target and pred states are in same scale
            if not pseudo_inverse:
                pred_state = self.lift_slot2full(pred_state, pred_attention, False, pseudo_inverse)
                pred_state = self.project_full2slot(pred_state, target_attention, False)

        else:
            raise ValueError('Choose full MDP (N-object) or slot MDP (K-slot) to compute pair-wise distance')

        return pred_state, target_state

    @staticmethod
    # @torch.no_grad()
    def get_pseudo_inverse(attention):
        """
        Since attention matrix is assumed to be a projection (from full to slot MDP)
        When we want to map back (from slot to full MDP), we need to use pseudo-inverse
        """
        # > pseudo-inverse
        pinv = torch.linalg.pinv(attention)
        # > also need to transpose, since we originally just use transpose instead of pseudo-inverse
        pinv_t = pinv.permute((0, 2, 1))

        return torch.clone(pinv_t)

    def compute_representation_loss(self, data_batch, with_neg_obs=False, with_vis=False,
                                    representation_loss_config='reconstruction'):
        """
        for training representation
        """
        info = {}
        loss = 0

        obs, action, next_obs, neg_obs = data_batch

        slots, action_slots, action_attention = self.get_encoding(
            obs=obs, action=action, no_grad=False
        )
        next_slots, _, next_action_attention = self.get_encoding(
            obs=next_obs, action=copy.deepcopy(action), no_grad=False
        )  # (a_t)
        # >>> Additional output for action/identity-slot attention

        # > Join the obs and train - to utilize both for training
        slots = torch.cat([slots, next_slots], dim=0)
        obs = torch.cat([obs, next_obs], dim=0)

        if representation_loss_config == 'reconstruction':
            recon_combined, recons, masks, _ = self.decode_slots(slots, no_grad=False)  # > no grad
            recon_loss = nn.MSELoss()(recon_combined, obs)
            loss += recon_loss

            info.update({
                'Train/ReconstructionLoss': recon_loss,
            })

            if with_vis:
                info.update({
                    # >>> sample reconstructed images for vis
                    'Visualization/Reconstruction': self.sample_images(
                        obs, recon_combined, recons, masks, n_samples=4, pair=True
                    )
                })

        else:
            raise NotImplementedError

        return loss, info

    def visualize_reconstruction(self, obs):
        """
        Visualize reconstruction and reconstructed slots given obs
        """
        with torch.no_grad():
            slots = self.get_slots(obs)
            recon_combined, recons, masks, _ = self.decode_slots(slots, no_grad=False)
            vis = self.sample_images(
                obs, recon_combined, recons, masks, n_samples=4, pair=False
            )
        return vis

    def compute_transition_loss(self, data_batch, with_neg_obs=False, with_vis=False, with_vis_recon=False,
                                transition_loss_config='contrastive', pseudo_inverse_loss=False):
        """
        Loss for training transition model - using contrastive loss like C-SWM
        """
        info = {}
        loss = 0

        obs, action, next_obs, neg_obs = data_batch

        slots, action_slots, action_attention = self.get_encoding(
            obs=obs, action=action, no_grad=True
        )
        next_slots, _, next_action_attention = self.get_encoding(
            obs=next_obs, action=copy.deepcopy(action), no_grad=True
        )  # (a_t)
        # >>> Additional output for action/identity-slot attention

        # >>> Encode pixel/action slots to lower dimensions (output mapping)
        latent_state = self.pixel_slots_out(slots)
        latent_action = self.action_slots_out(action_slots)
        next_latent_state = self.pixel_slots_out(next_slots)
        assert latent_state.size(-1) == self.embedding_dim
        assert latent_action.size(-1) == self.latent_action_dim

        if transition_loss_config == 'contrastive':
            # > Compute loss with learned action binding (attention) for
            pos_loss = self.energy(
                latent_state, latent_action, next_latent_state,
                action_attention, next_action_attention,  # >>> with attention
                pseudo_inverse_loss=pseudo_inverse_loss  # TODO add
            )

            # >>> Using negative samples: loading from data or random shuffle
            if with_neg_obs:
                # >>> Encode negative states directly
                neg_slots = self.get_slots(neg_obs, no_grad=True)  # > with grad
                neg_latent_state = self.pixel_slots_out(neg_slots)
            else:
                # Sample negative state across episodes at random
                batch_size = latent_state.size(0)
                perm = np.random.permutation(batch_size)
                neg_latent_state = latent_state[perm]

            # >>> Compute negative loss using neg samples
            neg_loss = torch.max(
                torch.zeros_like(pos_loss),
                self.hinge - self.energy(
                    latent_state, latent_action, neg_latent_state,
                    action_attention, next_action_attention,  # >>> with attention
                    no_trans=True,
                    pseudo_inverse_loss=pseudo_inverse_loss  # TODO add
                )
            )

            contrastive_loss = pos_loss.mean() + neg_loss.mean()
            info.update({
                'Train/ContrastiveLoss': contrastive_loss.item(),
                'Train/ContrastiveLoss-Positive': pos_loss.mean().item(),
                'Train/ContrastiveLoss-Negative': neg_loss.mean().item(),
            })

            vis_id = 0
            obs_concat = torch.stack([obs[vis_id], next_obs[vis_id]], dim=0)
            slots_concat = torch.stack([slots[vis_id], next_slots[vis_id]], dim=0)

            # > Add recon in transition training - visualize if transition has any effect
            if with_vis_recon:
                with torch.no_grad():
                    # > Decode current state and visualize reconstruction
                    # > Use concat slots and concat obs
                    recon_combined, recons, masks, _ = self.decode_slots(slots_concat, no_grad=True)
                    info.update({
                        'Visualization/Reconstruction': self.sample_images(
                            obs_concat, recon_combined, recons, masks, n_samples=None, pair=True
                        )
                    })

            # > Compute binding information for visualization
            if with_vis:
                info_update = self.get_binding_instance(info, vis_id,
                                                        action, action_attention, latent_action, latent_state,
                                                        next_action_attention, next_latent_state)
                info.update(info_update)

            loss += contrastive_loss

        else:
            raise NotImplementedError

        return loss, info

    def get_binding_instance(self, info, vis_id,
                             action, action_attention, latent_action, latent_state, next_action_attention,
                             next_latent_state):
        """
        Compute (1) state embedding and (2) slot
        For (1) difference (between time t and t+1) and (2) L2 error (between t+1 and predicted t+1)
        """

        info.update({
            'Visualization/ActionAttentionMatrix': action_attention[vis_id].cpu().numpy(),
            'Visualization/ActionAttentionNextMatrix': next_action_attention[vis_id].cpu().numpy(),
            'Visualization/LatentActionMatrix': latent_action[vis_id].detach().cpu().numpy(),
            'Visualization/ActionMatrix': self.action2oh(action)[vis_id].cpu().numpy()
        })

        with torch.no_grad():
            # > Project N-object embeddings using the 'full-MDP' strategy
            # > [N x K+1] x [K+1 x D] = [N, D]
            state_projected = self.lift_slot2full(latent_state[vis_id], action_attention[vis_id],
                                                  False, False)
            state_projected_next = self.lift_slot2full(next_latent_state[vis_id], next_action_attention[vis_id],
                                                       False, False)
            # state_projected = action_attention[vis_id] @ latent_state[vis_id]
            # state_projected_next = next_action_attention[vis_id] @ next_latent_state[vis_id]
            state_diff = state_projected - state_projected_next

            # > Project next state back to K-slot space (alignment two time steps)
            # > Note: still need to multiply the attention matrix, since M^T * M is not identity
            slot_next_aligned = self.project_full2slot(state_projected_next, action_attention[vis_id])
            slot_aligned = self.project_full2slot(state_projected, action_attention[vis_id])
            # slot_next_aligned = action_attention[vis_id].t().contiguous() @ state_projected_next
            # slot_aligned = action_attention[vis_id].t().contiguous() @ state_projected
            slot_diff = slot_aligned - slot_next_aligned

            # > Hard binding
            attention_hard = self.get_hard_binding(action_attention[vis_id])
            attention_hard_next = self.get_hard_binding(next_action_attention[vis_id])

            state_hard_projected = attention_hard @ latent_state[vis_id]
            state_hard_projected_next = attention_hard_next @ next_latent_state[vis_id]
            state_diff_hard = state_hard_projected - state_hard_projected_next

            slot_next_hard_aligned = attention_hard.t().contiguous() @ state_hard_projected_next
            slot_hard_aligned = attention_hard.t().contiguous() @ state_hard_projected
            slot_diff_hard = slot_hard_aligned - slot_next_hard_aligned

            # > Compute state/embedding error (L2) (with predicted next step), soft or hard version
            state_pred = self.transition_model(
                latent_state[vis_id].unsqueeze(0), latent_action[vis_id].unsqueeze(0)
            )
            state_pred = state_pred.squeeze(0) + latent_state[vis_id]

            # > Project state using binding at time t
            state_pred_projected = action_attention[vis_id] @ state_pred
            state_pred_hard_projected = attention_hard @ state_pred
            state_error = (state_pred_projected - state_projected_next).pow(2)
            state_error_hard = (state_pred_hard_projected - state_hard_projected_next).pow(2)

            # > Slot error
            slot_pred_aligned = action_attention[vis_id].t().contiguous() @ state_pred_projected
            slot_pred_hard_aligned = attention_hard.t().contiguous() @ state_pred_projected
            slot_error = (slot_pred_aligned - slot_next_aligned).pow(2)
            slot_error_hard = (slot_pred_hard_aligned - slot_next_hard_aligned).pow(2)

            # > Slot error using ordering of T+1, for verification
            # > The result (slot_error_t1) should be the same as the one from T (slot_error), up to row permutation
            action_aligned_t1 = self.get_align_action(
                action=action[vis_id], action_attention=next_action_attention[vis_id]
            )
            # action_aligned_t1_hard = self.get_align_action(
            #     action=action[vis_id], action_attention=attention_hard_next
            # )

            # > compute transition T: Z x U |-> Z, using the ordering of T+1 to align
            slot_aligned_t1 = next_action_attention[vis_id].t().contiguous() @ state_projected
            slot_hard_aligned_t1 = attention_hard_next.t().contiguous() @ state_projected

            # > Note: still use soft action?
            # TODO the input state is incorrect! muse use pseudo-inverse
            slot_pred_t1 = self.transition_model(slot_aligned_t1.unsqueeze(0), action_aligned_t1)
            slot_pred_hard_t1 = self.transition_model(slot_hard_aligned_t1.unsqueeze(0), action_aligned_t1)

            slot_next_aligned_t1 = next_action_attention[vis_id].t().contiguous() @ state_projected_next
            slot_next_hard_aligned_t1 = attention_hard_next.t().contiguous() @ state_hard_projected_next

            # > compute slot error with ordering@t+1
            slot_error_t1 = (slot_pred_t1.squeeze(0) - slot_next_aligned_t1).pow(2)
            slot_error_hard_t1 = (slot_pred_hard_t1.squeeze(0) - slot_next_hard_aligned_t1).pow(2)

        info_update = {
            # > Soft embeddings/slots and differences
            'Visualization/Embedding-(FullMDP)': state_projected,
            'Visualization/EmbeddingNext-(FullMDP)': state_projected_next,
            'Visualization/Embedding-Difference-(FullMDP)': state_diff,
            'Visualization/Slot-Difference-(SlotMDP)': slot_diff,

            # > Hard version
            'Visualization/Embedding-Hard-(FullMDP)': state_hard_projected,
            'Visualization/EmbeddingNext-Hard-(FullMDP)': state_hard_projected_next,
            'Visualization/Embedding-Difference-Hard-(FullMDP)': state_diff_hard,
            'Visualization/Slot-Difference-Hard-(SlotMDP)': slot_diff_hard,

            # > Soft and hard slot/embedding error
            'Visualization/Embedding-Error-(FullMDP)': state_error,
            'Visualization/Slot-Error-(SlotMDP)': slot_error,
            'Visualization/Embedding-Error-Hard-(FullMDP)': state_error_hard,
            'Visualization/Slot-Error-Hard-(SlotMDP)': slot_error_hard,

            # > T+1 aligned version
            'Visualization/Slot-Error-T1-(SlotMDP)': slot_error_t1,
            'Visualization/Slot-Error-Hard-T1-(SlotMDP)': slot_error_hard_t1,
        }
        # > convert to numpy for plt visualization
        info_update = {k: v.detach().cpu().numpy() for k, v in info_update.items()}
        info.update(info_update)

        return info

    def get_hard_binding(self, attention: torch.Tensor, strategy='one-hot', to_float=True):
        # > Input attention should be [B, N, K] or [N, K], then we take max for every N objects
        assert attention.dim() in [2, 3], 'Incorrect dimension'

        if strategy == 'one-hot':
            # > Hard version has N of 1's (one for every slot), and every object has up to K(+1) choices/classes
            hard_attention = torch.nn.functional.one_hot(
                attention.argmax(dim=-1, keepdim=False), num_classes=self.num_slots
            )
            assert hard_attention.sum() == self.num_objects_total

            # > Convert to float, since `"addmm_cuda" not implemented for 'Long'` (until 2021/12)
            if to_float:
                hard_attention = hard_attention.float()
        else:
            raise NotImplementedError

        return hard_attention

    def get_action_binding(self, obs, action):
        with torch.no_grad():
            slots, action_slots, action_attention = self.get_encoding(
                obs=obs, action=action,
            )
        return action_attention[0].cpu().numpy()

    def predict(self, observations, actions, num_steps, device='cpu', intermediate_states=False,
                visualize=False, space='full-mdp', hard_bind=False, pseudo_inverse=False):
        """
        Predict trajectories in latent space
        Note: used in evaluation loop
        # TODO: add visualization for multi-step prediction

        Args:
            intermediate_states: if also return intermediate states, or just the final state
            visualize: if visualize a predicted trajectory
            space: the space to compute error (K-slot MDP or N-object full MDP)
            hard_bind: use hard max in binding for computing loss
        """

        latent_states = []
        binding_attentions = []
        latent_actions_aligned = []

        with torch.no_grad():

            actions = actions + [actions[-1]]  # > Additional action for action_attention (to align obs)
            for step in range(num_steps + 1):
                # > Encode to latent space
                slots = self.get_slots(obs=observations[step], no_grad=True)
                latent_state = self.pixel_slots_out(slots)
                assert latent_state.size(-1) == self.embedding_dim
                action_attention = self.action_attention.get_attention(slots=slots, detach=True)

                latent_states.append(latent_state)
                binding_attentions.append(action_attention)

                # >>> Get actions aligned using the first object-slot's order (M_t)
                # > Note: don't use hard binding for action - it's input to the model and learns with soft binding
                latent_action_aligned = self.get_align_action(
                    action=actions[step], action_attention=binding_attentions[0],
                    out_enc=True
                )
                latent_actions_aligned.append(latent_action_aligned)

            # > Check action ordering
            del latent_actions_aligned[-1]
            assert len(latent_actions_aligned) == num_steps

            # > Option: Preprocess - use hard attention at test time, so no average over objects
            if hard_bind:
                hard_binding_attentions = []
                for attention in binding_attentions:
                    hard_binding_attentions.append(self.get_hard_binding(attention=attention))
                binding_attentions = hard_binding_attentions

            pred_states = []
            pred_state = latent_states[0]
            # > Predict states with aligned action, instead of aligning states iteratively (blow up!)
            for i in range(num_steps):
                pred_trans = self.transition_model(pred_state, latent_actions_aligned[i])
                pred_state = pred_state + pred_trans
                pred_states.append(pred_state)

            aligned_pred_states = []
            aligned_target_states = []
            # > Align state: pred states follow (step 0)'s ordering, and target is the (step i)'s ordering
            for i in range(num_steps):
                # > Make sure: (1) states are correspond, (2) use correct binding
                aligned_pred_state, aligned_target_state = self.align_state(
                    pred_state=pred_states[i], target_state=latent_states[i],
                    pred_attention=binding_attentions[0], target_attention=binding_attentions[i],
                    space=space, hard_bind=False, pseudo_inverse=pseudo_inverse
                )

                aligned_pred_states.append(aligned_pred_state)
                aligned_target_states.append(aligned_target_state)

        assert len(pred_states) == len(aligned_pred_states)
        assert len(aligned_pred_states) == len(aligned_target_states)

        if intermediate_states:
            return aligned_pred_states, aligned_target_states
        else:
            return aligned_pred_states[-1].to(device), aligned_target_states[-1].to(device)

    @staticmethod
    def sample_images(in_obs, recon_combined, recons, masks, n_samples=None, pair=True):
        """
        adapted from Slot Attention
        sample images from a batch and create grid
        """

        # > Only sample when n_samples is int
        if n_samples is not None:
            if not pair:
                perm = torch.randperm(len(in_obs))
                idx = perm[: n_samples]

            else:
                # > Sample pairs of current and next obs - from first and second half
                perm = torch.randperm(len(in_obs) // 2)
                idx1 = perm[: n_samples // 2]
                idx2 = idx1 + len(in_obs) // 2
                idx = torch.cat([idx1, idx2])

            in_obs = in_obs[idx]
            recon_combined, recons, masks = recon_combined[idx], recons[idx], masks[idx]

        def to_rgb_from_tensor(x):
            return (x * 0.5 + 0.5).clamp(0, 1)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    in_obs.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1 - masks),  # each slot
                ],
                dim=1,
            )
        )

        shape = recons.shape
        batch_size, num_slots, C, H, W = shape
        images = torchvision.utils.make_grid(
            out.view(batch_size * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
        )

        return images.detach().cpu().numpy().transpose(1, 2, 0)

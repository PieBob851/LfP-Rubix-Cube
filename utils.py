import torch.distributions as dist
import torch.optim as optim

def train_sample(batch_size, beta, encoder, actor, planner, encoder_optimizer, actor_optimizer, planner_optimizer):
    sample = data.sample_batch(batch_size).to(device)
    current_state = sample[:, 0, :-18]
    current_action = sample[:, 0, -18:]
    goal_state = sample[:, -1, :-18]
    goal_action = sample[:, -1, -18:]

    z, mu_phi, sigma_phi = encoder.forward(sample)
    mu_psi, sigma_psi = planner.forward(current_state, goal_state)

    phi_gaussian = dist.Normal(mu_phi, sigma_phi)

    psi_gaussian = dist.Normal(mu_psi, sigma_psi)

    KL_loss = torch.sum(dist.kl.kl_divergence(phi_gaussian, psi_gaussian))

    policy_action, _ = actor.forward(current_state.unsqueeze(1), z.unsqueeze(1), goal_state.unsqueeze(1))

    action_loss = F.cross_entropy(policy_action.squeeze(1), current_action)

    loss = beta * KL_loss + action_loss

    encoder_optimizer.zero_grad()
    planner_optimizer.zero_grad()
    actor_optimizer.zero_grad()

    loss.backward()

    encoder_optimizer.step()
    planner_optimizer.step()
    actor_optimizer.step()
    return loss
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
g = torch.tensor(9.81, requires_grad=False)  # Gravitational acceleration in m/s^2


# Simulation parameters
initial_speed = 5.0  # Initial speed of the ball
optim_steps = 5 # Number of optimization steps
ramp_angle = torch.tensor(1)  # angle of ramp
target_x = 5.0  # Target horizontal position in meters



# Function calaculate the landing distance of a ball based on an initials speed v0_val and a ramp angle ramp_angle
def simulate_ball_landing_distance(v0_val):
    x_range = (v0_val**2 * torch.sin(2 * ramp_angle)) / g
    return x_range



# Function to animate the trajectory
def animate_trajectory(v0_val):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.2, 10)
    ax.set_ylim(-0.2, max(initial_speed, v0_val.item()) + 1)

    h_val = (v0_val**2) / (2 * g)
    

    # Calculate final trajectory based on v0_val
    x_range = simulate_ball_landing_distance(v0_val)
    trajectory_x = np.linspace(0, x_range.item(), num=100)
    trajectory_y =  trajectory_x * np.tan(ramp_angle.item()) - \
                    (0.5 * g.numpy() * trajectory_x**2) / (v0_val.detach().numpy()**2 * np.cos(ramp_angle.item())**2)

    # Plot the target
    ax.plot(target_x, 0, 'gx', markersize=10, label='Target')

    # Ball point and trajectory line
    ball, = ax.plot([], [], 'ro', label='Ball')
    line, = ax.plot([], [], 'r-', lw=2)

    # Initialization function for the animation
    def init():
        line.set_data([], [])
        ball.set_data([], [])
        return line, ball,

    # Update function for the animation
    def update(frame):
        line.set_data(trajectory_x[:frame], trajectory_y[:frame])
        ball.set_data(trajectory_x[frame], trajectory_y[frame])
        return line, ball,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(trajectory_x), init_func=init, blit=True, interval=50, repeat=False)
    plt.legend()
    plt.show()



# Optionally, perform optimization here and adjust v0 accordingly
def optimize_v0():
    # Optimize v0 to minimize the difference between landing position and target
    optimizer = torch.optim.SGD([v0], lr=0.01)
    loss_fn = torch.nn.L1Loss()
    target_tensor = torch.tensor(target_x)
    for i in range(optim_steps):  # Run for 100 steps or until convergence
        optimizer.zero_grad()
        x_range = simulate_ball_landing_distance(v0)
        loss = loss_fn(x_range, target_tensor)
        loss.backward()
        optimizer.step()
        # Print progress
        print(f"Step {i}, v0: {v0.item()}, Loss: {loss.item()}")
    return v0    

v0 = torch.tensor(initial_speed, requires_grad=True)  # Initial speed, possibly to be optimized
# Uncomment the next line to run the animation with the current value of v0
v0 = optimize_v0()
animate_trajectory(v0)

# PPO LunarLander Training

This project implements a PPO (Proximal Policy Optimization) agent for the LunarLander-v3 environment from OpenAI Gymnasium. The agent learns to land a spacecraft with continuous actions, dealing with wind effects for added challenge.

## Project Structure

- `models.py`: Contains the PPO implementation, including the Actor-Critic network and Memory class
- `train.py`: Contains the training logic and environment setup
- `test.py`: Contains the testing and evaluation logic with visualization
- `main.py`: CLI interface to run training or testing

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

The project can be run in two modes: training and testing. Both modes support wind effect parameters.

### Environment Parameters

- `--enable-wind`: Enable wind effects (flag, default: False)
- `--wind-power`: Wind power when wind is enabled (default: 15.0)
- `--turbulence-power`: Turbulence power when wind is enabled (default: 1.5)

### Training Mode

To train a new agent with default parameters:

```bash
python main.py train
```

To train with wind effects:

```bash
python main.py train --enable-wind --wind-power 15.0 --turbulence-power 1.5
```

Training parameters that can be customized:
- `--decay-threshold`: Reward threshold for starting action std decay (default: 100)
- `--decay-speed`: Speed of action std decay (default: 0.97)
- `--max-episodes`: Maximum number of training episodes (default: 2500)
- `--max-timesteps`: Maximum timesteps per episode (default: 350)
- `--render-interval`: Interval between rendered episodes (default: 500)

Example with custom parameters:

```bash
python main.py train --enable-wind --wind-power 20.0 --decay-threshold 150 --max-episodes 3000
```

During training:
- Model checkpoints are saved every 500 episodes as `PPO_continuous_LunarLander-v3.pth`
- Training progress is logged to the console
- A history of average rewards is saved to `history_avg_reward.txt`
- The environment is rendered every `render-interval` episodes to visualize progress

### Testing Mode

To test a trained model:

```bash
python main.py test
```

To test with wind effects:

```bash
python main.py test --enable-wind --wind-power 15.0 --turbulence-power 1.5
```

Testing parameters that can be customized:
- `--model-path`: Path to the trained model file (default: "trained_model.pth", a model trained on an environment with no wind nor turbulences)
- `--n-simulations`: Number of test simulations to run (default: 100)

Example with custom parameters:

```bash
python main.py test --enable-wind --wind-power 10.0 --model-path "my_model.pth" --n-simulations 50
```

During testing:
- Runs multiple non-rendered simulations for statistical analysis
- Final simulation is rendered for visual inspection
- Generates histograms of rewards and episode lengths
- Saves visualization to `test_results.png`
- Prints summary statistics including mean and standard deviation of rewards and episode lengths

## Output Files

- `PPO_continuous_LunarLander-v3.pth`: Trained model checkpoint
- `history_avg_reward.txt`: Training history with average rewards and episode lengths
- `test_results.png`: Visualization of test results with reward and episode length distributions

## Notes

- GPU acceleration is automatically used if available (CUDA)
- The training includes action standard deviation decay to balance exploration and exploitation
- Testing runs multiple episodes to ensure reliable performance metrics
- The final test simulation is always rendered for visual inspection

## About the use of AI
AI tools were used to make the plots, help refacto the code, add comments and docstrings, create the CLI interface and write the README. The model used was Claude-3.5-sonnet in the IDE Cursor.
import argparse
import time
import gc
from train import Train
from test import Test


def main():
    parser = argparse.ArgumentParser(description="Train or test PPO agent for LunarLander")
    parser.add_argument("mode", choices=["train", "test"], help="Choose whether to train or test the agent")

    # Environment parameters
    env_group = parser.add_argument_group("Environment parameters")
    env_group.add_argument(
        "--enable-wind",
        action="store_true",
        help="Enable wind effects (default: False)",
    )
    env_group.add_argument(
        "--wind-power",
        type=float,
        default=15.0,
        help="Wind power when wind is enabled (default: 15.0)",
    )
    env_group.add_argument(
        "--turbulence-power",
        type=float,
        default=1.5,
        help="Turbulence power when wind is enabled (default: 1.5)",
    )

    # Training specific arguments
    training_group = parser.add_argument_group("Training parameters")
    training_group.add_argument(
        "--decay-threshold",
        type=float,
        default=100,
        help="Reward threshold for starting action std decay (default: 100)",
    )
    training_group.add_argument(
        "--decay-speed", type=float, default=0.97, help="Speed of action std decay (default: 0.97)"
    )
    training_group.add_argument(
        "--max-episodes", type=int, default=2500, help="Maximum number of training episodes (default: 2500)"
    )
    training_group.add_argument(
        "--max-timesteps", type=int, default=350, help="Maximum number of timesteps per episode (default: 350)"
    )
    training_group.add_argument(
        "--render-interval",
        type=int,
        default=500,
        help="Interval between rendered episodes during training (default: 500)",
    )

    # Testing specific arguments
    testing_group = parser.add_argument_group("Testing parameters")
    testing_group.add_argument(
        "--model-path",
        type=str,
        default="trained_model.pth",
        help="Path to the model file for testing (default: trained_model.pth)",
    )
    testing_group.add_argument(
        "--n-simulations", type=int, default=100, help="Number of test simulations to run (default: 100)"
    )

    args = parser.parse_args()

    # Create environment kwargs
    env_kwargs = {
        "continuous": True,
        "enable_wind": args.enable_wind,
        "wind_power": args.wind_power if args.enable_wind else 0.0,
        "turbulence_power": args.turbulence_power if args.enable_wind else 0.0,
    }

    if args.mode == "train":
        print("Starting training...")
        start_time = time.time()
        train = Train(
            decay_threshold=args.decay_threshold,
            decay_speed=args.decay_speed,
            max_episodes=args.max_episodes,
            max_timesteps=args.max_timesteps,
            render_interval=args.render_interval,
            env_kwargs=env_kwargs,
        )
        history_avg_reward = train.train()
        execution_time = time.time() - start_time
        print("Training completed in %s seconds" % (execution_time))
        gc.collect()

    else:  # test mode
        print("Starting testing...")
        test = Test(
            model_path=args.model_path,
            n_simulations=args.n_simulations,
            env_kwargs=env_kwargs,
        )
        rewards, timesteps = test.load_and_test_model()


if __name__ == "__main__":
    main()

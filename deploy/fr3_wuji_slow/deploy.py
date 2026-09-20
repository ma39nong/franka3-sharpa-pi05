"""Run the existing slow executor through the deploy namespace.

The implementation remains in experiments.weight_motion_eval.oneshot so
existing imports, model-specific wrappers, and commands continue to work.
"""

from experiments.weight_motion_eval.oneshot.deploy import main


if __name__ == "__main__":
    main()

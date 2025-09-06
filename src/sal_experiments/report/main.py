from .hyperparameter_scaling import hyperparameter_scaling_report
from .model_scaling import model_scaling_report


def main():
    hyperparameter_scaling_report()
    model_scaling_report()


if __name__ == "__main__":
    main()

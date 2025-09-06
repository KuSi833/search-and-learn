from pathlib import Path

import matplotlib.pyplot as plt

from .colors import BLUE, GREEN, PURPLE


def model_scaling_report():
    """Generate model scaling report showing accuracy vs compute efficiency."""

    # Data points from the LaTeX figure
    data = {
        # Baselines
        "Q4": {
            "flops": 0.20,
            "accuracy": 84.4,
            "marker": "o",  # Square
            "color": BLUE.triplet.hex,
        },
        "Q8": {
            "flops": 0.33,
            "accuracy": 86.9,
            "marker": "s",  # Square
            "color": BLUE.triplet.hex,
        },
        "Instruct": {
            "flops": 1.00,
            "accuracy": 100.0,
            "marker": "o",  # Circle
            "color": PURPLE.triplet.hex,
        },
        # CGAI methods (Blue color, different shapes for variants)
        "CGAI Q4→Inst 10%": {
            "flops": 0.30,
            "accuracy": 91.1,
            "marker": "o",  # Circle
            "color": GREEN.triplet.hex,
        },
        "CGAI Q4→Inst 20%": {
            "flops": 0.40,
            "accuracy": 94.4,
            "marker": "s",  # Square
            "color": GREEN.triplet.hex,
        },
        "CGAI Q8→Inst 10%": {
            "flops": 0.43,
            "accuracy": 91.8,
            "marker": "D",  # Diamond
            "color": GREEN.triplet.hex,
        },
        "CGAI Q8→Inst 20%": {
            "flops": 0.53,
            "accuracy": 96.7,
            "marker": "^",  # Triangle
            "color": GREEN.triplet.hex,
        },
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each data point
    for label, point in data.items():
        ax.scatter(
            point["flops"],
            point["accuracy"],
            marker=point["marker"],
            color=point["color"],
            s=120,  # Good size for visibility
            label=label,
            edgecolors="black",
            linewidth=0.8,
            zorder=3,
        )

    # Add annotation for the headline result
    ax.annotate(
        "96.7% accuracy at 53% compute",
        xy=(0.53, 96.7),
        xytext=(0.62, 98.5),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8
        ),
    )

    # Styling to match the LaTeX figure
    ax.set_xlabel("Relative FLOPs (Instruct = 1.0)", fontsize=12)
    ax.set_ylabel("Accuracy relative to Instruct (%)", fontsize=12)
    ax.set_xlim(0.18, 1.05)
    ax.set_ylim(82, 101)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Position legend in the bottom right
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save the plot
    output_dir = Path("figures/report/model_scaling")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "accuracy_vs_compute.pdf", bbox_inches="tight")

    print(f"Model scaling plot saved to {output_dir}/accuracy_vs_compute.pdf")

    # plt.show()


if __name__ == "__main__":
    model_scaling_report()

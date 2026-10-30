
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    accuracy_score
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
RADIUS = 2.0

NUM_NATURAL = 1000
NUM_GENERATED = 1000

DEVICE = "cpu"

np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("results", exist_ok=True)

print("Device:", DEVICE)


# ============================================================
# 2. GENERATE NATURAL POINTS ON A CIRCLE
# ============================================================

def generate_natural_points(n):

    theta = np.random.uniform(0, 2 * np.pi, n)

    x = RADIUS * np.cos(theta)
    y = RADIUS * np.sin(theta)

    points = np.column_stack((x, y))

    return points, theta


natural_points, natural_theta = generate_natural_points(
    NUM_NATURAL
)


# ============================================================
# 3. GENERATE OFF-MANIFOLD POINTS
# ============================================================

def generate_generated_points(n):

    theta = np.random.uniform(0, 2 * np.pi, n)

    # Normal displacement
    displacement = np.random.uniform(0.03, 0.35, n)

    radius = RADIUS + displacement

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    points = np.column_stack((x, y))

    return points, theta, displacement


generated_points, generated_theta, generated_displacement = (
    generate_generated_points(NUM_GENERATED)
)


# ============================================================
# 4. COMBINE DATA
# ============================================================

X = np.vstack([
    natural_points,
    generated_points
])

# 0 = natural, 1 = generated
y = np.concatenate([
    np.zeros(NUM_NATURAL),
    np.ones(NUM_GENERATED)
])

X_tensor = torch.tensor(
    X,
    dtype=torch.float32,
    requires_grad=True
)


# ============================================================
# 5. DEFINE THE TWO FUNCTIONS
# ============================================================

def f1(x):

    # Signed radial distance from circle
    radius = torch.sqrt(
        torch.sum(x ** 2, dim=1) + 1e-12
    )

    return radius - RADIUS


def f2(x):

    # Constant function on the manifold
    return torch.zeros(
        x.shape[0],
        device=x.device
    )


# ============================================================
# 6. CALCULATE CONSISTENCY / INCONSISTENCY SCORE
# ============================================================

with torch.no_grad():

    output1 = f1(X_tensor)
    output2 = f2(X_tensor)

    delta = torch.abs(output1 - output2)

scores = delta.cpu().numpy()


# ============================================================
# 7. VERIFY TANGENT AND NORMAL VECTORS
# ============================================================

def tangent_vector(theta):

    return np.array([
        -np.sin(theta),
        np.cos(theta)
    ])


def normal_vector(theta):

    return np.array([
        np.cos(theta),
        np.sin(theta)
    ])


test_angles = np.linspace(
    0, 2 * np.pi, 100, endpoint=False
)

dot_products = []
gradient_normal_alignment = []
gradient_tangent_alignment = []

for theta in test_angles:

    t = tangent_vector(theta)
    n = normal_vector(theta)

    dot_products.append(np.dot(t, n))

    point = torch.tensor(
        [[RADIUS * np.cos(theta),
          RADIUS * np.sin(theta)]],
        dtype=torch.float32,
        requires_grad=True
    )

    value = f1(point)

    gradient = torch.autograd.grad(
        value.sum(), point
    )[0].detach().numpy()[0]

    gradient_normal_alignment.append(
        np.dot(gradient, n)
        / (np.linalg.norm(gradient) * np.linalg.norm(n))
    )

    gradient_tangent_alignment.append(
        np.dot(gradient, t)
        / (np.linalg.norm(gradient) * np.linalg.norm(t))
    )

print("\nGEOMETRIC VERIFICATION")
print("----------------------")

print(
    "Maximum |tangent . normal|:",
    np.max(np.abs(dot_products))
)

print(
    "Mean gradient-normal alignment:",
    np.mean(gradient_normal_alignment)
)

print(
    "Mean absolute gradient-tangent alignment:",
    np.mean(np.abs(gradient_tangent_alignment))
)


# ============================================================
# 8. VERIFY THE TWO-FUNCTION ORTHOGONALITY PRINCIPLE
# ============================================================

# For f2, the gradient is zero everywhere.
# The zero vector belongs to every linear subspace.

point = torch.tensor(
    [[RADIUS + 0.2, 0.0]],
    dtype=torch.float32,
    requires_grad=True
)

grad_f1 = torch.autograd.grad(
    f1(point).sum(),
    point,
    retain_graph=True
)[0]

grad_f2 = torch.zeros_like(point)

print("\nGRADIENT VERIFICATION")
print("--------------------")

print("Gradient f1:", grad_f1.detach().numpy())
print("Gradient f2:", grad_f2.detach().numpy())


# ============================================================
# 9. TEST TANGENT AUGMENTATION: ROTATION
# ============================================================

def rotate_points(points, angle):

    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    return points @ rotation_matrix.T


rotation_angle = np.pi / 4

rotated_points = rotate_points(X, rotation_angle)

with torch.no_grad():

    original_score = f1(
        torch.tensor(X, dtype=torch.float32)
    )

    rotated_score = f1(
        torch.tensor(rotated_points, dtype=torch.float32)
    )

rotation_difference = torch.abs(
    original_score - rotated_score
).numpy()

print("\nROTATION EXPERIMENT")
print("-------------------")

print(
    "Mean score difference after rotation:",
    np.mean(rotation_difference)
)

print(
    "Natural mean difference:",
    np.mean(rotation_difference[y == 0])
)

print(
    "Generated mean difference:",
    np.mean(rotation_difference[y == 1])
)


# ============================================================
# 10. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(y, scores)

auc = roc_auc_score(y, scores)

print("\nDETECTION PERFORMANCE")
print("----------------------")

print("AUROC:", auc)


# ============================================================
# 11. THRESHOLD AND CLASSIFICATION
# ============================================================

best_index = np.argmax(tpr - fpr)

threshold = thresholds[best_index]

predictions = (scores >= threshold).astype(int)

accuracy = accuracy_score(y, predictions)

print("Threshold:", threshold)
print("Accuracy:", accuracy)

cm = confusion_matrix(y, predictions)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 12. PLOT 1: MANIFOLD AND GENERATED POINTS
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))

theta = np.linspace(0, 2 * np.pi, 500)

circle_x = RADIUS * np.cos(theta)
circle_y = RADIUS * np.sin(theta)

ax.plot(
    circle_x,
    circle_y,
    linewidth=2,
    label="Natural manifold"
)

ax.scatter(
    natural_points[:, 0],
    natural_points[:, 1],
    s=8,
    alpha=0.5,
    label="Natural points"
)

ax.scatter(
    generated_points[:, 0],
    generated_points[:, 1],
    s=8,
    alpha=0.5,
    label="Generated points"
)

# Show tangent and normal at one point
theta0 = np.pi / 4

p = np.array([
    RADIUS * np.cos(theta0),
    RADIUS * np.sin(theta0)
])

t = tangent_vector(theta0)
n = normal_vector(theta0)

ax.quiver(
    p[0], p[1],
    t[0], t[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="green",
    width=0.008,
    label="Tangent"
)

ax.quiver(
    p[0], p[1],
    n[0], n[1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="red",
    width=0.008,
    label="Normal"
)

ax.set_title("Natural Manifold and Off-Manifold Points")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.axis("equal")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/manifold.png", dpi=300)
plt.show()


# ============================================================
# 13. PLOT 2: CONSISTENCY SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    scores[y == 0],
    bins=40,
    alpha=0.65,
    label="Natural"
)

plt.hist(
    scores[y == 1],
    bins=40,
    alpha=0.65,
    label="Generated"
)

plt.axvline(
    threshold,
    linestyle="--",
    label="Classification threshold"
)

plt.xlabel("Inconsistency score")
plt.ylabel("Frequency")
plt.title("Natural vs Generated Score Distribution")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/score_distribution.png", dpi=300)
plt.show()


# ============================================================
# 14. PLOT 3: ROC CURVE
# ============================================================

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ConV-inspired detector (AUC={auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/roc_curve.png", dpi=300)
plt.show()


# ============================================================
# 15. PLOT 4: GEOMETRIC ORTHOGONALITY
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    test_angles,
    np.abs(dot_products),
    label="|Tangent . Normal|"
)

plt.plot(
    test_angles,
    np.abs(gradient_tangent_alignment),
    label="|Gradient f1 . Tangent|"
)

plt.xlabel("Angle (radians)")
plt.ylabel("Absolute alignment")
plt.title("Verification of Orthogonality")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/orthogonality.png", dpi=300)
plt.show()


# ============================================================
# 16. SAVE EXPERIMENT RESULTS
# ============================================================

results = pd.DataFrame({
    "x": X[:, 0],
    "y": X[:, 1],
    "true_label": y,
    "inconsistency_score": scores,
    "predicted_label": predictions
})

results.to_csv(
    "results/experiment_results.csv",
    index=False
)

summary = pd.DataFrame([{
    "AUROC": auc,
    "Accuracy": accuracy,
    "Threshold": threshold,
    "Mean_tangent_normal_dot": np.mean(
        np.abs(dot_products)
    ),
    "Mean_gradient_normal_alignment": np.mean(
        gradient_normal_alignment
    ),
    "Mean_gradient_tangent_alignment": np.mean(
        np.abs(gradient_tangent_alignment)
    ),
    "Mean_rotation_score_difference": np.mean(
        rotation_difference
    )
}])

summary.to_csv(
    "results/experiment_summary.csv",
    index=False
)

print("\nAll experiments completed.")
print("Results saved in the results/ directory.")

import torch
import cooper

class MaximumEntropy(cooper.ConstrainedMinimizationProblem):
    def __init__(self, mean_constraint):
        self.mean_constraint = mean_constraint
        super().__init__(is_constrained=True)

    def closure(self, probs):
        # Verify domain of definition of the functions
        # assert torch.all(probs >= 0)

        # Negative signed removed since we want to *maximize* the entropy
        entropy = torch.sum(probs * torch.log(probs))

        # Entries of p >= 0 (equiv. -p <= 0)
        ineq_defect = -probs

        # Equality constraints for proper normalization and mean constraint
        mean = torch.sum(torch.tensor(range(1, len(probs) + 1)) * probs)
        eq_defect = torch.stack([torch.sum(probs) - 1, mean - self.mean_constraint])

        return cooper.CMPState(loss=entropy, eq_defect=eq_defect, ineq_defect=ineq_defect)

# Define the problem and formulation
cmp = MaximumEntropy(mean_constraint=4.5)
formulation = cooper.LagrangianFormulation(cmp)

# Define the primal parameters and optimizer
probs = torch.nn.Parameter(torch.rand(6)) # Use a 6-sided die
print(probs)
print(f"Original Cross Entropy: {torch.sum(probs * torch.log(probs))}")
primal_optimizer = cooper.optim.ExtraSGD([probs], lr=3e-2, momentum=0.7)

# Define the dual optimizer. Note that this optimizer has NOT been fully instantiated
# yet. Cooper takes care of this, once it has initialized the formulation state.
dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraSGD, lr=9e-3, momentum=0.7)

# Wrap the formulation and both optimizers inside a ConstrainedOptimizer
coop = cooper.ConstrainedOptimizer(formulation, primal_optimizer, dual_optimizer)

# Here is the actual training loop.
# The steps follow closely the `loss -> backward -> step` Pytorch workflow.
for iter_num in range(5000):
    coop.zero_grad()
    lagrangian = formulation.composite_objective(cmp.closure, probs)
    formulation.custom_backward(lagrangian)
    coop.step(cmp.closure, probs)

print(f"Original Cross Entropy: {torch.sum(probs * torch.log(probs))}")
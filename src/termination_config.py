from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class TerminationCriteria:
    """
    Configuration class for defining termination conditions for the Adaptive Learning Process (ALP) loop.

    Attributes:
        max_iterations (Optional[int]): Maximum number of iterations allowed. 
            If None, iteration count is not a termination factor.
        performance_threshold (Optional[float]): Minimum performance threshold to 
            terminate the learning process. 
            Must be between 0.0 and 1.0 if specified.
        convergence_window (Optional[int]): Number of consecutive iterations 
            where performance must be stable to consider convergence.
        min_performance_change (Optional[float]): Minimum performance change 
            required to continue iterations.
    """

    max_iterations: Optional[int] = None
    performance_threshold: Optional[float] = None
    convergence_window: Optional[int] = None
    min_performance_change: Optional[float] = None

    def __post_init__(self):
        """
        Validate the input parameters after initialization.
        Raises:
            ValueError: If the input parameters violate constraints.
        """
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")

        if self.performance_threshold is not None:
            if not (0.0 <= self.performance_threshold <= 1.0):
                raise ValueError("performance_threshold must be between 0.0 and 1.0.")

        if self.convergence_window is not None and self.convergence_window <= 0:
            raise ValueError("convergence_window must be a positive integer.")

        if self.min_performance_change is not None:
            if not (0.0 <= self.min_performance_change <= 1.0):
                raise ValueError("min_performance_change must be between 0.0 and 1.0.")

    def should_terminate(
        self, 
        current_iteration: int, 
        current_performance: float, 
        performance_history: list[float]
    ) -> bool:
        """
        Determine if the learning process should terminate based on configured criteria.

        Args:
            current_iteration (int): Current iteration number.
            current_performance (float): Current performance metric.
            performance_history (list[float]): List of past performance metrics.

        Returns:
            bool: True if termination criteria are met, False otherwise.
        """
        # Check max iterations
        if self.max_iterations is not None and current_iteration >= self.max_iterations:
            return True

        # Check performance threshold
        if (self.performance_threshold is not None and 
            current_performance >= self.performance_threshold):
            return True

        # Check convergence window
        if (self.convergence_window is not None and 
            len(performance_history) >= self.convergence_window):
            recent_performances = performance_history[-self.convergence_window:]
            converged = all(
                abs(p1 - p2) < (self.min_performance_change or 0)
                for p1, p2 in zip(recent_performances, recent_performances[1:])
            )
            return converged

        return False
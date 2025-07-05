import pytest
from src.termination_config import TerminationCriteria


def test_default_configuration():
    """Test default configuration with no constraints."""
    config = TerminationCriteria()
    assert not config.should_terminate(0, 0.5, [])


def test_max_iterations_termination():
    """Test termination based on maximum iterations."""
    config = TerminationCriteria(max_iterations=5)
    assert config.should_terminate(5, 0.1, [])
    assert not config.should_terminate(4, 0.1, [])


def test_performance_threshold_termination():
    """Test termination based on performance threshold."""
    config = TerminationCriteria(performance_threshold=0.8)
    assert config.should_terminate(10, 0.9, [])
    assert not config.should_terminate(10, 0.7, [])


def test_convergence_window_termination():
    """Test termination based on performance convergence."""
    config = TerminationCriteria(
        convergence_window=3, 
        min_performance_change=0.02
    )
    # Not converged due to larger changes
    history_non_converged = [0.5, 0.6, 0.7]
    assert not config.should_terminate(3, 0.7, history_non_converged)
    
    # Converged with small changes
    history_converged = [0.5, 0.51, 0.52]
    assert config.should_terminate(3, 0.52, history_converged)


def test_invalid_configurations():
    """Test invalid configuration parameters."""
    with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
        TerminationCriteria(max_iterations=0)

    with pytest.raises(ValueError, match="performance_threshold must be between 0.0 and 1.0"):
        TerminationCriteria(performance_threshold=1.5)

    with pytest.raises(ValueError, match="convergence_window must be a positive integer"):
        TerminationCriteria(convergence_window=0)

    with pytest.raises(ValueError, match="min_performance_change must be between 0.0 and 1.0"):
        TerminationCriteria(min_performance_change=1.5)
import logging

logger = logging.getLogger("cwt_prediction.risk_manager.kelly")

def kelly_fraction(
    model_prob: float,
    market_implied_prob: float,
    max_fraction: float = 0.25
) -> float:
    """
    Computes the Kelly Criterion position sizing for binary markets.
    
    Simplified formula for binary options:
    f* = (p - m) / (1 - m)
    
    where:
      p = model probability of winning (0.0 < p < 1.0)
      m = market implied probability/price of the contract (0.0 < m < 1.0)
      
    Args:
      model_prob: The model's probability estimate of the chosen outcome.
      market_implied_prob: The market price (implied prob) of that same outcome.
      max_fraction: A safety cap on the maximum bet size (defaults to 0.25).
      
    Returns:
      Position fraction to allocate (0.0 to max_fraction).
    """
    # Validation
    if not (0.0 < model_prob < 1.0):
        raise ValueError(f"model_prob must be between 0 and 1, got {model_prob}")
    if not (0.0 < market_implied_prob < 1.0):
        raise ValueError(f"market_implied_prob must be between 0 and 1, got {market_implied_prob}")
    if max_fraction <= 0.0 or max_fraction > 1.0:
        raise ValueError(f"max_fraction must be in (0.0, 1.0], got {max_fraction}")

    # If the model has no positive edge (model_prob <= market_implied_prob), we do not bet
    if model_prob <= market_implied_prob:
        return 0.0

    # Calculate Kelly fraction
    try:
        fraction = (model_prob - market_implied_prob) / (1.0 - market_implied_prob)
    except ZeroDivisionError:
        return 0.0

    # Apply the safety cap and clip negative values
    final_fraction = max(0.0, min(fraction, max_fraction))
    
    logger.debug(
        f"Kelly calculation: model_p={model_prob:.4f}, market_p={market_implied_prob:.4f} "
        f"-> raw_fraction={fraction:.4f}, final_fraction={final_fraction:.4f}"
    )
    
    return final_fraction

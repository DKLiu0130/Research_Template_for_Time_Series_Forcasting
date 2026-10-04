"""Default forecast experiment used by the template entrypoint."""
from exp.base_exp import BaseForecastExp


class ForecastExp(BaseForecastExp):
    """Concrete projects can subclass and override hooks here."""
    pass

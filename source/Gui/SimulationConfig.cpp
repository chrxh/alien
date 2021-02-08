#include "EngineInterface/SimulationParameters.h"

#include "SimulationConfig.h"

 auto _SimulationConfig::validate(string & errorMsg) const -> ValidationResult
{
	 return ValidationResult::Ok;
}

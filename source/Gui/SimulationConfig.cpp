#include "ModelBasic/SimulationParameters.h"

#include "SimulationConfig.h"

auto _SimulationConfigCpu::validate(string & errorMsg) const -> ValidationResult
{
	IntVector2D unitSize = { universeSize.x / gridSize.x, universeSize.y / gridSize.y };
	int range = std::min(unitSize.x, unitSize.y);

	if (parameters->clusterMaxRadius + parameters->cellFunctionCommunicatorRange + 1 >= range) {
		errorMsg = "Unit size is too small for simulation parameters.";
		return ValidationResult::Error;
	}
	if (parameters->clusterMaxRadius + parameters->cellFunctionSensorRange + 1 >= range) {
		errorMsg = "Unit size is too small for simulation parameters.";
		return ValidationResult::Error;
	}
	return ValidationResult::Ok;
}

 auto _SimulationConfigGpu::validate(string & errorMsg) const -> ValidationResult
{
	 return ValidationResult::Ok;
}

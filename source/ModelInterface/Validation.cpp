#include "SimulationParameters.h"
#include "Validation.h"

ValidationResult Validation::validate(IntVector2D const& universeSize
	, IntVector2D const& gridSize
	, SimulationParameters const* parameters)
{
	IntVector2D unitSize = { universeSize.x / gridSize.x, universeSize.y / gridSize.y };
	int range = std::min(unitSize.x, unitSize.y);

	if (parameters->clusterMaxRadius + parameters->cellFunctionCommunicatorRange + 1 >= range) {
		return ValidationResult::ErrorUnitSizeTooSmall;
	}
	if (parameters->clusterMaxRadius + parameters->cellFunctionSensorRange + 1 >= range) {
		return ValidationResult::ErrorUnitSizeTooSmall;
	}
	return ValidationResult::Ok;
}

#include "ParametersValidationService.h"

void ParametersValidationService::validateAndCorrect(SimulationParameters& parameters) const
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        parameters.minCellEnergy.baseValue[i] = std::min(parameters.minCellEnergy.baseValue[i], parameters.normalCellEnergy.value[i] * 0.95f);
    }
}

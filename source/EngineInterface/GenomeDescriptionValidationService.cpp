#include "GenomeDescriptionValidationService.h"

void GenomeDescriptionValidationService::validateAndCorrect(GenomeDescription_New& genome)
{
    for (auto& gene : genome._genes) {
        for (auto& node : gene._nodes) {
            auto nodeType = node.getCellType();
            if (nodeType == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription_New>(node._cellTypeData);
                if (constructor._autoTriggerInterval.has_value()) {
                    auto& value = constructor._autoTriggerInterval.value();
                    value = std::max(value, 1);
                }
            } else if (nodeType == CellTypeGenome_Sensor) {
                auto& sensor = std::get<SensorGenomeDescription_New>(node._cellTypeData);
                if (sensor._autoTriggerInterval.has_value()) {
                    auto& value = sensor._autoTriggerInterval.value();
                    value = std::max(value, 1);
                }
            }
        }
    }
}

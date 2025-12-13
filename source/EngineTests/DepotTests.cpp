#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class DepotTests : public IntegrationTestFramework
{
public:
    DepotTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~DepotTests() = default;
};


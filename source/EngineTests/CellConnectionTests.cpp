#include <gtest/gtest.h>

#include "Base/NumberGenerator.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class CellConnectionTests : public IntegrationTestFramework
{
public:
    CellConnectionTests()
        : IntegrationTestFramework()
    {}

    ~CellConnectionTests() = default;
};

TEST_F(CellConnectionTests, decay)
{
    _parameters.baseValues.radiationAbsorption[0] = 0;
    _parameters.cellDeathConsequences = CellDeathConsquences_CreatureDies;
    _parameters.cellDeathProbability[0] = 0.5f;

    _simulationFacade->setSimulationParameters(_parameters);
    auto origData =
        DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters().width(10).height(10).energy(_parameters.baseValues.cellMinEnergy[0] / 2));

    _simulationFacade->setSimulationData(origData);
    for (int i = 0; i < 1000; ++i) {
        _simulationFacade->calcTimesteps(1);
    }

    auto data = _simulationFacade->getSimulationData();
    EXPECT_EQ(0, data.cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(origData)));
}


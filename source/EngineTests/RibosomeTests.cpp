#include <gtest/gtest.h>

#include "Base/NumberGenerator.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class RibosomeTests : public IntegrationTestFramework
{
public:
    RibosomeTests()
        : IntegrationTestFramework({1000, 1000})
    {}

    ~RibosomeTests() = default;
};

TEST_F(RibosomeTests, constructSingleCell)
{
    auto parameters = _simController->getSimulationParameters();

    RibosomeDescription ribosome;
    ribosome.createGenome({CellDescription().setColor(2)});

    DataDescription data;
    data.addCell(
        CellDescription().setId(1).setEnergy(parameters.cellNormalEnergy * 3).setMaxConnections(1).setExecutionOrderNumber(0).setCellFunction(ribosome));

    _simController->setSimulationData(data);
    auto actualData = _simController->getSimulationData();

    EXPECT_EQ(2, actualData.cells.size());
}

#include <gtest/gtest.h>

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class NerveTests : public IntegrationTestFramework
{
public:
    NerveTests()
        : IntegrationTestFramework({1000, 1000})
    {}

    ~NerveTests() = default;
};

TEST_F(NerveTests, activityTransfer)
{
    std::vector<float> activity = {1, 0, 1, 0, 0, 0, 0, 0};

    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({1.0f, 1.0f}).setMaxConnections(2).setExecutionOrderNumber(0).setInputBlocked(true).setActivity(activity),
        CellDescription().setId(2).setPos({2.0f, 1.0f}).setMaxConnections(2).setExecutionOrderNumber(1),
        CellDescription().setId(3).setPos({3.0f, 1.0f}).setMaxConnections(2).setExecutionOrderNumber(2).setOutputBlocked(true),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simController->setSimulationData(data);
    _simController->calcSingleTimestep();
    _simController->calcSingleTimestep();
    auto actualData = _simController->getSimulationData();
}

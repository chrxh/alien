#include <gtest/gtest.h>

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class DataTransferTests : public IntegrationTestFramework
{
public:
    DataTransferTests()
        : IntegrationTestFramework({1000, 1000})
    {}

    ~DataTransferTests() = default;
};

TEST_F(DataTransferTests, singleCell)
{
    DataDescription data;
    data.addCell(CellDescription()
                     .setPos({2.0f, 4.0f})
                     .setVel({0.5f, 1.0f})
                     .setMaxConnections(1)
                     .setExecutionOrderNumber(3)
                     .setAge(1)
                     .setColor(2)
                     .setBarrier(true)
                     .setUnderCOnstruction(false)
                     .setInputBlocked(true)
                     .setOutputBlocked(false));

    _simController->setSimulationData(data);
    auto newData = _simController->getSimulationData();
    EXPECT_TRUE(compare(data, newData));
}


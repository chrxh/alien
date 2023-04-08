#include <gtest/gtest.h>

#include "Base/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class DescriptionHelperTests 
    : public IntegrationTestFramework
{
public:
    DescriptionHelperTests()
        : IntegrationTestFramework(std::nullopt, {100, 100})
    {}
    virtual ~DescriptionHelperTests() = default;

protected:
    bool areAngelsCorrect(ClusteredDataDescription const& clusteredData) const
    {
        for (auto const& cluster : clusteredData.clusters) {
            for (auto const& cell : cluster.cells) {
                if (!cell.connections.empty()) {
                    float sumAngles = 0;
                    for (auto const& connection : cell.connections) {
                        sumAngles += connection.angleFromPrevious;
                    }
                    if (std::abs(sumAngles - 360.0f) > NEAR_ZERO) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
};


TEST_F(DescriptionHelperTests, correctConnections)
{
    auto data = DescriptionHelper::createRect(DescriptionHelper::CreateRectParameters().width(10).height(10).center({50.0f, 99.0f}));
    _simController->setSimulationData(data);
    auto clusteredData = _simController->getClusteredSimulationData();

    DescriptionHelper::correctConnections(clusteredData, {100, 100});

    EXPECT_TRUE(areAngelsCorrect(clusteredData));
}

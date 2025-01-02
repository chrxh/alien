#include <gtest/gtest.h>

#include "Base/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class DescriptionEditTests 
    : public IntegrationTestFramework
{
public:
    DescriptionEditTests()
        : IntegrationTestFramework(std::nullopt, {100, 100})
    {}
    virtual ~DescriptionEditTests() = default;

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


TEST_F(DescriptionEditTests, correctConnections)
{
    auto data = DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().width(10).height(10).center({50.0f, 99.0f}));
    _simulationFacade->setSimulationData(data);
    auto clusteredData = _simulationFacade->getClusteredSimulationData();

    DescriptionEditService::get().correctConnections(clusteredData, {100, 100});

    EXPECT_TRUE(areAngelsCorrect(clusteredData));
}


TEST_F(DescriptionEditTests, addThirdConnection1)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({0, 0}),
        CellDescription().setId(2).setPos({1, 0}),
        CellDescription().setId(3).setPos({0, 1}),
        CellDescription().setId(4).setPos({0, -1}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(1, 4);

    auto cellById = getCellById(data);
    auto cell = cellById.at(1);

    EXPECT_EQ(3, cell.connections.size());

    auto connection1 = cell.connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1.distance));
    EXPECT_TRUE(approxCompare(90.0f, connection1.angleFromPrevious));

    auto connection2 = cell.connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2.distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2.angleFromPrevious));

    auto connection3 = cell.connections.at(2);
    EXPECT_TRUE(approxCompare(1.0f, connection3.distance));
    EXPECT_TRUE(approxCompare(180.0f, connection3.angleFromPrevious));
}

TEST_F(DescriptionEditTests, addThirdConnection2)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({0, 0}),
        CellDescription().setId(2).setPos({1, 0}),
        CellDescription().setId(3).setPos({-1, 0}),
        CellDescription().setId(4).setPos({0, 1}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(1, 4);

    auto cellById = getCellById(data);
    auto cell = cellById.at(1);

    EXPECT_EQ(3, cell.connections.size());

    auto connection1 = cell.connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1.distance));
    EXPECT_TRUE(approxCompare(180.0f, connection1.angleFromPrevious));

    auto connection2 = cell.connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2.distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2.angleFromPrevious));

    auto connection3 = cell.connections.at(2);
    EXPECT_TRUE(approxCompare(1.0f, connection3.distance));
    EXPECT_TRUE(approxCompare(90.0f, connection3.angleFromPrevious));
}

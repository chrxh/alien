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
    bool areAngelsCorrect(CollectionDescription const& data) const
    {
        for (auto const& cell : data._cells) {
            if (!cell._connections.empty()) {
                float sumAngles = 0;
                for (auto const& connection : cell._connections) {
                    sumAngles += connection._angleFromPrevious;
                }
                if (std::abs(sumAngles - 360.0f) > NEAR_ZERO) {
                    return false;
                }
            }
        }
        return true;
    }
};


TEST_F(DescriptionEditTests, correctConnections)
{
    auto origData = DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().width(10).height(10).center({50.0f, 99.0f}));
    _simulationFacade->setSimulationData(origData);

    auto data = _simulationFacade->getSimulationData();

    DescriptionEditService::get().correctConnections(data, {100, 100});

    EXPECT_TRUE(areAngelsCorrect(data));
}


TEST_F(DescriptionEditTests, addThirdConnection1)
{
    auto data = CollectionDescription().addCells({
        CellDescription().id(1).pos({0, 0}),
        CellDescription().id(2).pos({1, 0}),
        CellDescription().id(3).pos({0, 1}),
        CellDescription().id(4).pos({0, -1}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(1, 4);

    auto cellById = getCellById(data);
    auto cell = cellById.at(1);

    EXPECT_EQ(3, cell._connections.size());

    auto connection1 = cell._connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection1._angleFromPrevious));

    auto connection2 = cell._connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2._angleFromPrevious));

    auto connection3 = cell._connections.at(2);
    EXPECT_TRUE(approxCompare(1.0f, connection3._distance));
    EXPECT_TRUE(approxCompare(180.0f, connection3._angleFromPrevious));
}

TEST_F(DescriptionEditTests, addThirdConnection2)
{
    auto data = CollectionDescription().addCells({
        CellDescription().id(1).pos({0, 0}),
        CellDescription().id(2).pos({1, 0}),
        CellDescription().id(3).pos({-1, 0}),
        CellDescription().id(4).pos({0, 1}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(1, 4);

    auto cellById = getCellById(data);
    auto cell = cellById.at(1);

    EXPECT_EQ(3, cell._connections.size());

    auto connection1 = cell._connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1._distance));
    EXPECT_TRUE(approxCompare(180.0f, connection1._angleFromPrevious));

    auto connection2 = cell._connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2._angleFromPrevious));

    auto connection3 = cell._connections.at(2);
    EXPECT_TRUE(approxCompare(1.0f, connection3._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection3._angleFromPrevious));
}

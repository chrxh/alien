#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Description.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescription.h"

#include "IntegrationTestFramework.h"

class EnergyFlowTests : public IntegrationTestFramework
{
public:
    EnergyFlowTests()
        : IntegrationTestFramework()
    {}

    ~EnergyFlowTests() = default;
};

TEST_F(EnergyFlowTests, energyFlowsLeadsEqualDistribution)
{
    Description data;
    for (int i = 0; i < 20; ++i) {
        auto cell = CellDescription().id(i + 1).pos({100.0f + toFloat(i), 100.0f});
        data._cells.emplace_back(cell);
        if (i > 0) {
            data.addConnection(i, i + 1);
        }
    }
    data._cells.at(0)._energy = 10000.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    for (int i = 0; i < 20; ++i) {
        EXPECT_TRUE(actualData.getCellRef(i + 1)._energy < 600.0f);
    }
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, energyFlowsToActiveConstructor)
{
    auto data = Description().creatures({
        CreatureDescription().genome(GenomeDescription().genes({
        GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
    }))});
    auto& creature = data._creatures.front();
    for (int i = 0; i < 20; ++i) {
        auto cell = CellDescription().id(i + 1).pos({100.0f + toFloat(i), 100.0f});
        if (i == 19) {
            cell.cellType(ConstructorDescription().geneIndex(0).autoTriggerInterval(0).currentBranch(0));
        }
        creature._cells.emplace_back(cell);
        if (i > 0) {
            data.addConnection(i, i + 1);
        }
    }
    creature._cells.at(0)._energy = 10000.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._creatures.size());

    auto const& actualCreature = actualData._creatures.front();
    ASSERT_EQ(20, actualCreature._cells.size());

    for (int i = 1; i < 21; ++i) {
        if (i == 20) {
            EXPECT_TRUE(actualData.getCellRef(i)._energy > 10000.0f - 500.0f);
        } else {
            EXPECT_TRUE(actualData.getCellRef(i)._energy < 200.0f);
        }
    }
}

TEST_F(EnergyFlowTests, energyFlowsToClosestActiveConstructor)
{
    auto constructorId1 = 10 + 1;
    auto constructorId2 = 20 + 19 + 1;

    auto data = Description().creatures({
        CreatureDescription().genome(GenomeDescription().genes({
            GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
        })),
    });
    auto& creature = data._creatures.front();

    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 20; ++i) {
            auto id = i + j * 20 + 1;
            auto cell = CellDescription().id(id).pos({100.0f + toFloat(i), 100.0f});
            if (id == constructorId1 || id == constructorId2) {
                cell.cellType(ConstructorDescription().geneIndex(0).autoTriggerInterval(0).currentBranch(0));
            }
            creature._cells.emplace_back(cell);
            if (i > 0) {
                data.addConnection(id - 1, id);
            }
            if (j == 1) {
                data.addConnection(id - 20, id);
            }
        }
    }
    creature._cells.at(0)._energy = 10000.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    for (int i = 1; i < 41; ++i) {
        if (i == constructorId1) {
            EXPECT_TRUE(actualData.getCellRef(i)._energy > 10000.0f - 400.0f);
        } else {
            EXPECT_TRUE(actualData.getCellRef(i)._energy < 200.0f);
        }
    }
}

TEST_F(EnergyFlowTests, energyFlowsNotToFinishedConstructor)
{
    auto data = Description().creatures({
        CreatureDescription().genome(GenomeDescription().genes({
            GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
        })),
    });
    auto& creature = data._creatures.front();

    for (int i = 0; i < 20; ++i) {
        auto cell = CellDescription().id(i + 1).pos({100.0f + toFloat(i), 100.0f});
        if (i == 19) {
            cell.cellType(ConstructorDescription().geneIndex(0).autoTriggerInterval(0).currentBranch(1));
        }
        creature._cells.emplace_back(cell);
        if (i > 0) {
            data.addConnection(i, i + 1);
        }
    }
    creature._cells.at(0)._energy = 10000.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    for (int i = 0; i < 20; ++i) {
        EXPECT_TRUE(actualData.getCellRef(i + 1)._energy < 600.0f);
    }
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, energyFlowsBranches)
{
    auto data = Description().cells({
        CellDescription().id(0).pos({100.0f, 99.0f}).energy(100.0f),
        CellDescription().id(1).pos({100.0f, 100.0f}).energy(2400.0f),
        CellDescription().id(2).pos({100.0f, 101.0f}).energy(100.0f),
        CellDescription().id(3).pos({99.0f, 100.0f}).energy(100.0f),
    });
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();

        EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

        for (auto const& cell : actualData._cells) {
            EXPECT_TRUE(cell._energy > 100.0f - NEAR_ZERO);
        }
    }
    {
        auto actualData = _simulationFacade->getSimulationData();
        for (auto const& cell : actualData._cells) {
            EXPECT_TRUE(cell._energy > 2700.0f / 4  - 50.0f);
            EXPECT_TRUE(cell._energy < 2700.0f / 4 + 50.0f);
        }
    }
}

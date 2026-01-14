#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class EnergyFlowTests : public IntegrationTestFramework
{
public:
    EnergyFlowTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
            _parameters.maxRawEnergyConversion.value[i] = 0;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~EnergyFlowTests() = default;
};

TEST_F(EnergyFlowTests, usableEnergyFlowsLeadsEqualDistribution)
{
    Description data;
    for (int i = 0; i < 20; ++i) {
        auto cell = CellDescription().id(i + 1).pos({100.0f + toFloat(i), 100.0f});
        data._cells.emplace_back(cell);
        if (i > 0) {
            data.addConnection(i, i + 1);
        }
    }
    data._cells.at(0)._usableEnergy = 1000.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();

    for (int i = 0; i < 20; ++i) {
        EXPECT_TRUE(actualData.getCellRef(i + 1)._usableEnergy < 150.0f);
    }
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, usableEnergyFlowsToActiveConstructor)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
    });
    auto creature = CreatureDescription();
    auto creatureId = creature._id;
    
    std::vector<CellDescription> cells;
    for (int i = 0; i < 20; ++i) {
        auto cell = CellDescription().id(i + 1).pos({100.0f + toFloat(i), 100.0f}).creatureId(creatureId);
        if (i == 19) {
            cell.cellType(ConstructorDescription().geneIndex(0).autoTriggerInterval(0).currentBranch(0));
        }
        cells.push_back(cell);
    }
    cells.at(0)._usableEnergy = 1000.0f;
    
    Description data;
    data.addCreature(creature, cells, genome);
    for (int i = 1; i < 20; ++i) {
        data.addConnection(i, i + 1);
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._creatures.size());

    auto const& actualCreature = actualData._creatures.front();
    ASSERT_EQ(20, actualData.getCellsForCreature(actualCreature._id).size());

    for (int i = 1; i < 21; ++i) {
        if (i == 20) {
            EXPECT_TRUE(actualData.getCellRef(i)._usableEnergy > 900.0f);
        } else {
            EXPECT_TRUE(actualData.getCellRef(i)._usableEnergy < 110.0f);
        }
    }
}

TEST_F(EnergyFlowTests, usableEnergyFlowsToClosestActiveConstructor)
{
    auto constructorId1 = 10 + 1;
    auto constructorId2 = 20 + 19 + 1;

    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
    });
    auto creature = CreatureDescription();
    auto creatureId = creature._id;

    std::vector<CellDescription> cells;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 20; ++i) {
            auto id = i + j * 20 + 1;
            auto cell = CellDescription().id(id).pos({100.0f + toFloat(i), 100.0f}).creatureId(creatureId);
            if (id == constructorId1 || id == constructorId2) {
                cell.cellType(ConstructorDescription().geneIndex(0).autoTriggerInterval(0).currentBranch(0));
            }
            cells.push_back(cell);
        }
    }
    cells.at(0)._usableEnergy = 1000.0f;

    Description data;
    data.addCreature(creature, cells, genome);
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 20; ++i) {
            auto id = i + j * 20 + 1;
            if (i > 0) {
                data.addConnection(id - 1, id);
            }
            if (j == 1) {
                data.addConnection(id - 20, id);
            }
        }
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();

    for (int i = 1; i < 41; ++i) {
        if (i == constructorId1) {
            EXPECT_TRUE(actualData.getCellRef(i)._usableEnergy > 900.0f);
        } else {
            EXPECT_TRUE(actualData.getCellRef(i)._usableEnergy < 110.0f);
        }
    }
}

TEST_F(EnergyFlowTests, usableEnergyFlowsNotToFinishedConstructor)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numBranches(1).nodes({NodeDescription()}),
    });
    auto creature = CreatureDescription();
    auto creatureId = creature._id;

    std::vector<CellDescription> cells;
    for (int i = 0; i < 20; ++i) {
        auto cell = CellDescription().id(i + 1).pos({100.0f + toFloat(i), 100.0f}).creatureId(creatureId);
        if (i == 19) {
            cell.cellType(ConstructorDescription().geneIndex(0).autoTriggerInterval(0).currentBranch(1));
        }
        cells.push_back(cell);
    }
    cells.at(0)._usableEnergy = 1000.0f;

    Description data;
    data.addCreature(creature, cells, genome);
    for (int i = 1; i < 20; ++i) {
        data.addConnection(i, i + 1);
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();

    for (int i = 0; i < 20; ++i) {
        EXPECT_TRUE(actualData.getCellRef(i + 1)._usableEnergy < 150.0f);
    }
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, usableEnergyFlowsBranches)
{
    auto data = Description().cells({
        CellDescription().id(0).pos({100.0f, 99.0f}).usableEnergy(100.0f),
        CellDescription().id(1).pos({100.0f, 100.0f}).usableEnergy(240.0f),
        CellDescription().id(2).pos({100.0f, 101.0f}).usableEnergy(100.0f),
        CellDescription().id(3).pos({99.0f, 100.0f}).usableEnergy(100.0f),
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
            EXPECT_TRUE(cell._usableEnergy > 100.0f - NEAR_ZERO);
        }
    }
    {
        auto actualData = _simulationFacade->getSimulationData();
        for (auto const& cell : actualData._cells) {
            EXPECT_TRUE(cell._usableEnergy > 540.0f / 4 - 5.0f);
            EXPECT_TRUE(cell._usableEnergy < 540.0f / 4 + 5.0f);
        }
    }
}

TEST_F(EnergyFlowTests, usableEnergyFlowsNotToConstructorUnderConstruction)
{
    auto genome = GenomeDescription().genes({GeneDescription().separation(false).nodes({NodeDescription(), NodeDescription()})});

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    Description data;
    data.addCreature(
        CreatureDescription(),
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellType(ConstructorDescription().autoTriggerInterval(0).currentNodeIndex(1))
             .usableEnergy(normalCellEnergy * 10)},
        genome);
    data.addCreature(
        CreatureDescription(),
        {CellDescription()
             .id(2)
             .cellState(CellState_Constructing)
             .pos({100.0f + 1.0f + _parameters.constructorAdditionalOffspringDistance, 100.0f})
             .cellType(ConstructorDescription().currentNodeIndex(1))
             .usableEnergy(normalCellEnergy)},
        genome);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualEnergy = getEnergy(actualData);
    EXPECT_TRUE(approxCompare(getEnergy(data), actualEnergy));

    EXPECT_EQ(2, actualData.getNumCells());
    auto cell1 = actualData.getCellRef(1);
    auto cell2 = actualData.getCellRef(2);
    EXPECT_TRUE(abs(cell1._usableEnergy - normalCellEnergy * 10) < 1.0f);
    EXPECT_TRUE(abs(cell2._usableEnergy - normalCellEnergy) < 1.0f);
}

TEST_F(EnergyFlowTests, usableEnergyFlowsEquallyToActiveConstructors)
{
    auto genome = GenomeDescription().genes({GeneDescription().separation(false).nodes({NodeDescription(), NodeDescription()})});

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    Description data;
    data.addCreature(
        CreatureDescription(),
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellType(ConstructorDescription().autoTriggerInterval(0).currentNodeIndex(1))
             .usableEnergy(normalCellEnergy * 10)},
        genome);
    data.addCreature(
        CreatureDescription(),
        {CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellType(ConstructorDescription().autoTriggerInterval(0).currentNodeIndex(1))
             .usableEnergy(normalCellEnergy)},
        genome);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualEnergy = getEnergy(actualData);
    EXPECT_TRUE(approxCompare(getEnergy(data), actualEnergy));

    EXPECT_EQ(2, actualData.getNumCells());
    auto cell1 = actualData.getCellRef(1);
    auto cell2 = actualData.getCellRef(2);
    EXPECT_TRUE(abs(cell1._usableEnergy - actualEnergy / 2) < 1.0f);
    EXPECT_TRUE(abs(cell2._usableEnergy - actualEnergy / 2) < 1.0f);
}

TEST_F(EnergyFlowTests, usableEnergyFlowsPrioritizeLowEnergyCell)
{
    // Test that energy flows preferentially to cells with low energy (below normal energy threshold)
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).usableEnergy(normalCellEnergy * 5.0f),    // High energy cell
        CellDescription().id(2).pos({101.0f, 100.0f}).usableEnergy(normalCellEnergy * 0.5f),  // Low energy cell (below normal)
        CellDescription().id(3).pos({102.0f, 100.0f}).usableEnergy(normalCellEnergy),        // Normal energy cell
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(100);

    auto actualData = _simulationFacade->getSimulationData();

    // Cell 2 (low energy) should receive energy preferentially from cell 1
    auto cell1 = actualData.getCellRef(1);
    auto cell2 = actualData.getCellRef(2);
    auto cell3 = actualData.getCellRef(3);

    // Cell 2 should have gained significant energy (approaching normal energy)
    EXPECT_TRUE(cell2._usableEnergy > normalCellEnergy * 0.5 + 10.0f);

    // Cell 1 should have lost energy to cell 2
    EXPECT_TRUE(cell1._usableEnergy < normalCellEnergy * 5);

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, usableEnergyFlowsEqualizeLowEnergyCells)
{
    // Test that when connected cell has low energy, energy flows to equalize
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).usableEnergy(normalCellEnergy * 0.8f),  // Below normal
        CellDescription().id(2).pos({101.0f, 100.0f}).usableEnergy(normalCellEnergy * 0.55f),  // Much lower
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(200);

    auto actualData = _simulationFacade->getSimulationData();

    auto cell1 = actualData.getCellRef(1);
    auto cell2 = actualData.getCellRef(2);

    // Both cells should approach equal energy levels (since both are below normal)
    auto avgEnergy = (normalCellEnergy * 0.8 + normalCellEnergy * 0.55) / 2.0f;
    EXPECT_TRUE(abs(cell1._usableEnergy - avgEnergy) < 10.0f);
    EXPECT_TRUE(abs(cell2._usableEnergy - avgEnergy) < 10.0f);

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, usableEnergyFlowsFromHighToLowEnergyInChain)
{
    // Test energy flow prioritization in a chain with one low energy cell
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).usableEnergy(normalCellEnergy * 2),    // High
        CellDescription().id(2).pos({101.0f, 100.0f}).usableEnergy(normalCellEnergy),        // Normal
        CellDescription().id(3).pos({102.0f, 100.0f}).usableEnergy(normalCellEnergy * 0.55f),  // Low
        CellDescription().id(4).pos({103.0f, 100.0f}).usableEnergy(normalCellEnergy),        // Normal
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(150);

    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getCellRef(3);

    // Cell 3 (low energy) should receive energy and approach normal energy
    EXPECT_TRUE(cell3._usableEnergy > normalCellEnergy * 0.4 + 10.0f);

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, usableEnergyFlowsMultipleLowEnergyCells)
{
    // Test that multiple low energy cells all receive energy
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).usableEnergy(normalCellEnergy * 10),   // Very high energy
        CellDescription().id(2).pos({101.0f, 100.0f}).usableEnergy(normalCellEnergy * 0.55f),  // Low
        CellDescription().id(3).pos({102.0f, 100.0f}).usableEnergy(normalCellEnergy * 0.6f),  // Low
        CellDescription().id(4).pos({103.0f, 100.0f}).usableEnergy(normalCellEnergy * 0.7f),  // Low
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(1, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(200);

    auto actualData = _simulationFacade->getSimulationData();

    auto cell2 = actualData.getCellRef(2);
    auto cell3 = actualData.getCellRef(3);
    auto cell4 = actualData.getCellRef(4);

    // All low energy cells should have gained energy
    EXPECT_TRUE(cell2._usableEnergy > normalCellEnergy * 0.3);
    EXPECT_TRUE(cell3._usableEnergy > normalCellEnergy * 0.4);
    EXPECT_TRUE(cell4._usableEnergy > normalCellEnergy * 0.5);

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, rawEnergyFlow_nonDigestor_nonDigestor)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).rawEnergy(100.0f),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(100);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(100.0f, actualData.getCellRef(1)._rawEnergy));
    EXPECT_TRUE(approxCompare(0.0f, actualData.getCellRef(2)._rawEnergy));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, rawEnergyFlow_nonDigestor_digestor)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).rawEnergy(10.0f),
        CellDescription().id(2).cellType(DigestorDescription()).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(100);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(0.0f, actualData.getCellRef(1)._rawEnergy));
    EXPECT_TRUE(approxCompare(10.0f, actualData.getCellRef(2)._rawEnergy));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, rawEnergyFlow_digestor_nonDigestor)
{
    auto data = Description().cells({
        CellDescription().id(1).cellType(DigestorDescription()).pos({100.0f, 100.0f}).rawEnergy(10.0f),
        CellDescription().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(100);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(10.0f, actualData.getCellRef(1)._rawEnergy));
    EXPECT_TRUE(approxCompare(0.0f, actualData.getCellRef(2)._rawEnergy));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, rawEnergyFlow_digestor_digestor)
{
    auto data = Description().cells({
        CellDescription().id(1).cellType(DigestorDescription().rawEnergyConductivity(0.7f)).pos({100.0f, 100.0f}).rawEnergy(10.0f),
        CellDescription().id(2).cellType(DigestorDescription().rawEnergyConductivity(0.3f)).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(3.0f, actualData.getCellRef(1)._rawEnergy));
    EXPECT_TRUE(approxCompare(7.0f, actualData.getCellRef(2)._rawEnergy));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, rawEnergyFlows_highConductivity)
{
    // Test that higher rawEnergyConductivity increases flow rate
    Description dataLowConductivity;
    Description dataHighConductivity;
    
    // Setup two identical scenarios with different conductivity
    for (int scenario = 0; scenario < 2; ++scenario) {
        auto& data = (scenario == 0) ? dataLowConductivity : dataHighConductivity;
        float conductivity = (scenario == 0) ? 0.1f : 0.9f;
        
        for (int i = 0; i < 10; ++i) {
            auto cell = CellDescription().id(i + 1).pos({100.0f + toFloat(i), 100.0f});
            // Make all cells Digestors with different conductivity
            cell.cellType(DigestorDescription().rawEnergyConductivity(conductivity));
            data._cells.emplace_back(cell);
            if (i > 0) {
                data.addConnection(i, i + 1);
            }
        }
        // Put all raw energy in the first cell
        data._cells.at(0)._rawEnergy = 100.0f;
    }

    // Run simulations for a shorter time to see difference in flow rate
    _simulationFacade->setSimulationData(dataLowConductivity);
    _simulationFacade->calcTimesteps(100);
    auto actualDataLow = _simulationFacade->getSimulationData();

    _simulationFacade->setSimulationData(dataHighConductivity);
    _simulationFacade->calcTimesteps(100);
    auto actualDataHigh = _simulationFacade->getSimulationData();

    // With high conductivity, energy should have spread further
    // Check the last cell - it should have more energy with high conductivity
    auto lastCellLow = actualDataLow.getCellRef(10);
    auto lastCellHigh = actualDataHigh.getCellRef(10);
    
    EXPECT_TRUE(lastCellHigh._rawEnergy > lastCellLow._rawEnergy + NEAR_ZERO);
    
    // Energy conservation
    EXPECT_TRUE(approxCompare(getEnergy(dataLowConductivity), getEnergy(actualDataLow)));
    EXPECT_TRUE(approxCompare(getEnergy(dataHighConductivity), getEnergy(actualDataHigh)));
}

TEST_F(EnergyFlowTests, rawEnergyFlow_exceedRawEnergyThreshold)
{
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .cellType(DigestorDescription().rawEnergyConductivity(0.7f))
            .pos({100.0f, 100.0f})
            .rawEnergy(SimulationParameters::maxRawEnergyThresholdForConduction * 2),
        CellDescription()
            .id(2)
            .cellType(DigestorDescription().rawEnergyConductivity(0.3f))
            .pos({101.0f, 100.0f})
            .rawEnergy(SimulationParameters::maxRawEnergyThresholdForConduction + NEAR_ZERO),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(SimulationParameters::maxRawEnergyThresholdForConduction * 2, actualData.getCellRef(1)._rawEnergy));
    EXPECT_TRUE(approxCompare(SimulationParameters::maxRawEnergyThresholdForConduction + NEAR_ZERO, actualData.getCellRef(2)._rawEnergy));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

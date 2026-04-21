#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/GenomeDesc.h>
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
    std::vector<ObjectDesc> objects;
    for (int i = 0; i < 20; ++i) {
        auto object = ObjectDesc().id(i + 1).pos({100.0f + toFloat(i), 100.0f}).type(CellDesc().headCell(true));
        objects.emplace_back(object);
    }
    objects.at(0).getCellRef()._usableEnergy = 1000.0f;

    Desc data;
    data.addCreature(objects);
    for (int i = 1; i < 20; ++i) {
        data.addConnection(i, i + 1);
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();

    for (int i = 0; i < 20; ++i) {
        EXPECT_TRUE(actualData.getObjectRef(i + 1).getCellRef()._usableEnergy < 150.0f);
    }
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, usableEnergyFlowsToActiveConstructor)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).numBranches(1).nodes({NodeDesc()}),
    });

    std::vector<ObjectDesc> cells;
    for (int i = 0; i < 20; ++i) {
        auto object = ObjectDesc().id(i + 1).pos({100.0f + toFloat(i), 100.0f}).type(CellDesc().headCell(true));
        if (i == 19) {
            object.getCellRef()._constructor = ConstructorDesc().geneIndex(0).autoTriggerInterval(0);
        }
        cells.push_back(object);
    }
    cells.at(0).getCellRef()._usableEnergy = 1000.0f;

    Desc data;
    data.addCreature(cells, CreatureDesc(), genome);
    for (int i = 1; i < 20; ++i) {
        data.addConnection(i, i + 1);
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._creatures.size());

    auto const& actualCreature = actualData._creatures.front();
    ASSERT_EQ(20, actualData.getObjectsForCreature(actualCreature._id).size());

    for (int i = 1; i < 21; ++i) {
        if (i == 20) {
            EXPECT_TRUE(actualData.getObjectRef(i).getCellRef()._usableEnergy > 900.0f);
        } else {
            EXPECT_TRUE(actualData.getObjectRef(i).getCellRef()._usableEnergy < 110.0f);
        }
    }
}

TEST_F(EnergyFlowTests, usableEnergyFlowsToClosestActiveConstructor)
{
    auto constructorId1 = 10 + 1;
    auto constructorId2 = 20 + 19 + 1;

    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).numBranches(1).nodes({NodeDesc()}),
    });

    std::vector<ObjectDesc> cells;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 20; ++i) {
            auto id = i + j * 20 + 1;
            auto object = ObjectDesc().id(id).pos({100.0f + toFloat(i), 100.0f}).type(CellDesc().headCell(true));
            if (id == constructorId1 || id == constructorId2) {
                object.getCellRef()._constructor = ConstructorDesc().geneIndex(0).autoTriggerInterval(0);
            }
            cells.push_back(object);
        }
    }
    cells.at(0).getCellRef()._usableEnergy = 1000.0f;

    Desc data;
    data.addCreature(cells, CreatureDesc(), genome);
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
            EXPECT_TRUE(actualData.getObjectRef(i).getCellRef()._usableEnergy > 900.0f);
        } else {
            EXPECT_TRUE(actualData.getObjectRef(i).getCellRef()._usableEnergy < 110.0f);
        }
    }
}

TEST_F(EnergyFlowTests, usableEnergyFlowsNotToFinishedConstructor)
{
    auto genome = GenomeDesc().genes({});

    std::vector<ObjectDesc> cells;
    for (int i = 0; i < 20; ++i) {
        auto object = ObjectDesc().id(i + 1).pos({100.0f + toFloat(i), 100.0f}).type(CellDesc().headCell(true));
        if (i == 19) {
            object.getCellRef()._constructor = ConstructorDesc();
        }
        cells.push_back(object);
    }
    cells.at(0).getCellRef()._usableEnergy = 1000.0f;

    Desc data;
    data.addCreature(cells, CreatureDesc(), genome);
    for (int i = 1; i < 20; ++i) {
        data.addConnection(i, i + 1);
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();

    for (int i = 0; i < 20; ++i) {
        EXPECT_TRUE(actualData.getObjectRef(i + 1).getCellRef()._usableEnergy < 150.0f);
    }
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, usableEnergyFlowsBranches)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(0).pos({100.0f, 99.0f}).type(CellDesc().headCell(true).usableEnergy(100.0f)),
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(240.0f)),
        ObjectDesc().id(2).pos({100.0f, 101.0f}).type(CellDesc().usableEnergy(100.0f)),
        ObjectDesc().id(3).pos({99.0f, 100.0f}).type(CellDesc().usableEnergy(100.0f)),
    });
    data.addConnection(0, 1);
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();

        EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

        for (auto const& object : actualData._objects) {
            EXPECT_TRUE(object.getCellRef()._usableEnergy > 100.0f - NEAR_ZERO);
        }
    }
    {
        auto actualData = _simulationFacade->getSimulationData();
        for (auto const& object : actualData._objects) {
            EXPECT_TRUE(object.getCellRef()._usableEnergy > 540.0f / 4 - 5.0f);
            EXPECT_TRUE(object.getCellRef()._usableEnergy < 540.0f / 4 + 5.0f);
        }
    }
}

TEST_F(EnergyFlowTests, usableEnergyFlowsNotToConstructorUnderConstruction)
{
    auto genome = GenomeDesc().genes({GeneDesc().separation(false).nodes({NodeDesc(), NodeDesc()})});

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    Desc data;
    data.addCreature(
        {ObjectDesc()
             .id(1)
             .pos({100.0f, 100.0f})
             .type(CellDesc().headCell(true).constructor(ConstructorDesc().autoTriggerInterval(0)).usableEnergy(normalCellEnergy * 10)),
         ObjectDesc()
             .id(2)
             .pos({100.0f + 1.0f, 100.0f})
             .type(CellDesc().headCell(true).cellState(CellState_Constructing).constructor(ConstructorDesc()).usableEnergy(normalCellEnergy))},
        CreatureDesc(),
        genome);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualEnergy = getEnergy(actualData);
    EXPECT_TRUE(approxCompare(getEnergy(data), actualEnergy));

    EXPECT_EQ(2, actualData.getNumObjects());
    auto object1 = actualData.getObjectRef(1);
    auto object2 = actualData.getObjectRef(2);
    EXPECT_TRUE(abs(object1.getCellRef()._usableEnergy - normalCellEnergy * 10) < 1.0f);
    EXPECT_TRUE(abs(object2.getCellRef()._usableEnergy - normalCellEnergy) < 1.0f);
}

TEST_F(EnergyFlowTests, usableEnergyFlowsEquallyToActiveConstructors)
{
    auto genome = GenomeDesc().genes({GeneDesc().separation(false).nodes({NodeDesc(), NodeDesc()})});

    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    Desc data;
    data.addCreature(
        {ObjectDesc()
             .id(1)
             .pos({100.0f, 100.0f})
             .type(CellDesc().headCell(true).constructor(ConstructorDesc().autoTriggerInterval(0)).usableEnergy(normalCellEnergy * 10)),
         ObjectDesc()
             .id(2)
             .pos({101.0f, 100.0f})
             .type(CellDesc().headCell(true).constructor(ConstructorDesc().autoTriggerInterval(0)).usableEnergy(normalCellEnergy))},
        CreatureDesc(),
        genome);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualEnergy = getEnergy(actualData);
    EXPECT_TRUE(approxCompare(getEnergy(data), actualEnergy));

    EXPECT_EQ(2, actualData.getNumObjects());
    auto object1 = actualData.getObjectRef(1);
    auto object2 = actualData.getObjectRef(2);
    EXPECT_TRUE(abs(object1.getCellRef()._usableEnergy - actualEnergy / 2) < 1.0f);
    EXPECT_TRUE(abs(object2.getCellRef()._usableEnergy - actualEnergy / 2) < 1.0f);
}

TEST_F(EnergyFlowTests, usableEnergyFlowsPrioritizeLowEnergyCell)
{
    // Test that energy flows preferentially to cells with low energy (below normal energy threshold)
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().usableEnergy(normalCellEnergy * 5.0f)),  // High energy cell
        ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().usableEnergy(normalCellEnergy * 0.5f)),  // Low energy cell (below normal)
        ObjectDesc().id(3).pos({102.0f, 100.0f}).type(CellDesc().usableEnergy(normalCellEnergy)),         // Normal energy cell
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(100);

    auto actualData = _simulationFacade->getSimulationData();

    // Cell 2 (low energy) should receive energy preferentially from cell 1
    auto object1 = actualData.getObjectRef(1);
    auto object2 = actualData.getObjectRef(2);
    auto cell3 = actualData.getObjectRef(3);

    // Cell 2 should have gained significant energy (approaching normal energy)
    EXPECT_TRUE(object2.getCellRef()._usableEnergy > normalCellEnergy * 0.5 + 10.0f);

    // Cell 1 should have lost energy to cell 2
    EXPECT_TRUE(object1.getCellRef()._usableEnergy < normalCellEnergy * 5);

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, usableEnergyFlowsEqualizeLowEnergyCells)
{
    // Test that when connected cell has low energy, energy flows to equalize
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().headCell(true).usableEnergy(normalCellEnergy * 0.8f)),   // Below normal
        ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().usableEnergy(normalCellEnergy * 0.55f)),  // Much lower
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(200);

    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    auto object2 = actualData.getObjectRef(2);

    // Both cells should approach equal energy levels (since both are below normal)
    auto avgEnergy = (normalCellEnergy * 0.8 + normalCellEnergy * 0.55) / 2.0f;
    EXPECT_TRUE(abs(object1.getCellRef()._usableEnergy - avgEnergy) < 10.0f);
    EXPECT_TRUE(abs(object2.getCellRef()._usableEnergy - avgEnergy) < 10.0f);

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, usableEnergyFlowsFromHighToLowEnergyInChain)
{
    // Test energy flow prioritization in a chain with one low energy cell
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().headCell(true).usableEnergy(normalCellEnergy * 2)),      // High
        ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().usableEnergy(normalCellEnergy)),          // Normal
        ObjectDesc().id(3).pos({102.0f, 100.0f}).type(CellDesc().usableEnergy(normalCellEnergy * 0.55f)),  // Low
        ObjectDesc().id(4).pos({103.0f, 100.0f}).type(CellDesc().usableEnergy(normalCellEnergy)),          // Normal
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(150);

    auto actualData = _simulationFacade->getSimulationData();

    auto cell3 = actualData.getObjectRef(3);

    // Cell 3 (low energy) should receive energy and approach normal energy
    EXPECT_TRUE(cell3.getCellRef()._usableEnergy > normalCellEnergy * 0.4 + 10.0f);

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, usableEnergyFlowsMultipleLowEnergyCells)
{
    // Test that multiple low energy cells all receive energy
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().headCell(true).usableEnergy(normalCellEnergy * 10)),     // Very high energy
        ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().usableEnergy(normalCellEnergy * 0.55f)),  // Low
        ObjectDesc().id(3).pos({102.0f, 100.0f}).type(CellDesc().usableEnergy(normalCellEnergy * 0.6f)),   // Low
        ObjectDesc().id(4).pos({103.0f, 100.0f}).type(CellDesc().usableEnergy(normalCellEnergy * 0.7f)),   // Low
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(1, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(200);

    auto actualData = _simulationFacade->getSimulationData();

    auto object2 = actualData.getObjectRef(2);
    auto cell3 = actualData.getObjectRef(3);
    auto cell4 = actualData.getObjectRef(4);

    // All low energy cells should have gained energy
    EXPECT_TRUE(object2.getCellRef()._usableEnergy > normalCellEnergy * 0.3);
    EXPECT_TRUE(cell3.getCellRef()._usableEnergy > normalCellEnergy * 0.4);
    EXPECT_TRUE(cell4.getCellRef()._usableEnergy > normalCellEnergy * 0.5);

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, rawEnergyFlow_nonDigestor_nonDigestor)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().rawEnergy(100.0f)),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(100);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(100.0f, actualData.getObjectRef(1).getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(0.0f, actualData.getObjectRef(2).getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, rawEnergyFlow_nonDigestor_digestor)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().rawEnergy(10.0f)),
        ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().cellType(DigestorDesc())),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(100);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(0.0f, actualData.getObjectRef(1).getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(10.0f, actualData.getObjectRef(2).getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, rawEnergyFlow_digestor_nonDigestor)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(DigestorDesc()).rawEnergy(10.0f)),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(100);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(10.0f, actualData.getObjectRef(1).getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(0.0f, actualData.getObjectRef(2).getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, rawEnergyFlow_digestor_digestor)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(DigestorDesc().rawEnergyConductivity(0.7f)).rawEnergy(10.0f)),
        ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().cellType(DigestorDesc().rawEnergyConductivity(0.3f))),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(3.0f, actualData.getObjectRef(1).getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(7.0f, actualData.getObjectRef(2).getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(EnergyFlowTests, rawEnergyFlows_highConductivity)
{
    // Test that higher rawEnergyConductivity increases flow rate
    Desc dataLowConductivity;
    Desc dataHighConductivity;

    // Setup two identical scenarios with different conductivity
    for (int scenario = 0; scenario < 2; ++scenario) {
        auto& data = (scenario == 0) ? dataLowConductivity : dataHighConductivity;
        float conductivity = (scenario == 0) ? 0.1f : 0.9f;

        std::vector<ObjectDesc> cells;
        for (int i = 0; i < 10; ++i) {
            auto object = ObjectDesc().id(i + 1).pos({100.0f + toFloat(i), 100.0f});
            // Make all cells Digestors with different conductivity
            object.getCellRef()._cellType = DigestorDesc().rawEnergyConductivity(conductivity);
            cells.emplace_back(object);
        }
        // Put all raw energy in the first cell
        cells.at(0).getCellRef()._rawEnergy = 100.0f;

        data.addCreature(cells);
        for (int i = 1; i < 10; ++i) {
            data.addConnection(i, i + 1);
        }
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
    auto lastObjectLow = actualDataLow.getObjectRef(10);
    auto lastObjectHigh = actualDataHigh.getObjectRef(10);

    EXPECT_TRUE(lastObjectHigh.getCellRef()._rawEnergy > lastObjectLow.getCellRef()._rawEnergy + NEAR_ZERO);

    // Energy conservation
    EXPECT_TRUE(approxCompare(getEnergy(dataLowConductivity), getEnergy(actualDataLow)));
    EXPECT_TRUE(approxCompare(getEnergy(dataHighConductivity), getEnergy(actualDataHigh)));
}

TEST_F(EnergyFlowTests, rawEnergyFlow_exceedRawEnergyThreshold)
{
    auto data = Desc().addCreature({
        ObjectDesc()
            .id(1)
            .pos({100.0f, 100.0f})
            .type(CellDesc().cellType(DigestorDesc().rawEnergyConductivity(0.7f)).rawEnergy(SimulationParameters::maxRawEnergyThresholdForConduction * 2)),
        ObjectDesc()
            .id(2)
            .pos({101.0f, 100.0f})
            .type(CellDesc()
                      .cellType(DigestorDesc().rawEnergyConductivity(0.3f))
                      .rawEnergy(SimulationParameters::maxRawEnergyThresholdForConduction + NEAR_ZERO)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(SimulationParameters::maxRawEnergyThresholdForConduction * 2, actualData.getObjectRef(1).getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(SimulationParameters::maxRawEnergyThresholdForConduction + NEAR_ZERO, actualData.getObjectRef(2).getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

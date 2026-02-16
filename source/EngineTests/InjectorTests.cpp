#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class InjectorTests : public IntegrationTestFramework
{
public:
    InjectorTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
            _parameters.injectorEnergyCost.value[i] = 0;
            _parameters.injectorRadius.value[i] = 3.5f;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~InjectorTests() = default;

protected:
    // Helper to create an injector creature with a generator that triggers it
    Desc createInjectorWithGenerator(RealVector2D const& injectorPos, int geneIndex = 0, int injectorColor = 0)
    {
        auto data = Desc().addCreature({
                ObjectDesc()
                    .id(1)
                    .pos(injectorPos)
                    .color(injectorColor)
                    .type(CellDesc().neuralNetwork(NeuralNetDesc().bias(0, 1.0f)).cellType(InjectorDesc().geneIndex(geneIndex))),
            ObjectDesc().id(2).pos({injectorPos.x + 1.0f, injectorPos.y}).color(injectorColor),
        }, CreatureDesc().id(1));
        data.addConnection(1, 2);
        return data;
    }

    // Helper to create a target creature with a constructor at a given position
    Desc createTargetCreatureWithConstructor(RealVector2D const& pos, uint64_t creatureId = 2, int color = 0, float usableEnergy = 100.0f)
    {
        auto data = Desc().addCreature({
            ObjectDesc().id(100).pos(pos).color(color).type(CellDesc().usableEnergy(usableEnergy).constructor(ConstructorDesc())),
            ObjectDesc().id(101).pos({pos.x + 1.0f, pos.y}).color(color).type(CellDesc().usableEnergy(usableEnergy)),
        }, CreatureDesc().id(creatureId));
        data.addConnection(100, 101);
        return data;
    }
};

/**
 * Test: No target found
 * The injector should not inject when there's no constructor cell in range
 */
TEST_F(InjectorTests, noTargetFound)
{
    auto data = createInjectorWithGenerator({100.0f, 100.0f});

    // Add target creature with constructor outside injection radius
    data.add(createTargetCreatureWithConstructor({100.0f, 104.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualInjector = actualData.getObjectRef(1);

    // Injector should have a signal with success value = 0
    EXPECT_TRUE(approxCompare(0.0f, actualInjector.getCellRef()._signal._channels[Channels::InjectorSuccess]));
}

/**
 * Test: Successful injection
 * The injector should inject its genome into a nearby constructor cell
 */
TEST_F(InjectorTests, successfulInjection)
{
    // Create injector with geneIndex=2
    auto data = createInjectorWithGenerator({100.0f, 100.0f}, 2);

    // Add target creature with constructor within injection radius
    data.add(createTargetCreatureWithConstructor({100.0f, 103.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualInjector = actualData.getObjectRef(1);
    auto actualTargetConstructor = actualData.getObjectRef(100).getCellRef()._constructor.value();

    // Injector should have a signal with success value > 0
    EXPECT_TRUE(actualInjector.getCellRef()._signal._channels[Channels::InjectorSuccess] > NEAR_ZERO);

    // Target constructor should have the injector's geneIndex
    EXPECT_EQ(2, actualTargetConstructor._geneIndex);
}

/**
 * Test: No injection of own creature cells
 * Cells belonging to the same creature should not be injected
 */
TEST_F(InjectorTests, noInjectionOnOwnCreatureCells)
{
    // Create a single creature with injector and constructor
    auto data = Desc().addCreature({
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().neuralNetwork(NeuralNetDesc().bias(0, 1.0f)).cellType(InjectorDesc().geneIndex(3))),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
        ObjectDesc().id(3).pos({100.0f, 103.0f}).type(CellDesc().constructor(ConstructorDesc().geneIndex(0))),  // Same creature
    }, CreatureDesc().id(1));
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    auto origConstructor = data.getObjectRef(3).getCellRef()._constructor.value();

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualConstructor = actualData.getObjectRef(3).getCellRef()._constructor.value();

    // Constructor's geneIndex should remain unchanged
    EXPECT_EQ(origConstructor._geneIndex, actualConstructor._geneIndex);
}

/**
 * Test: No injection on fixed cells
 * Cells with fixed=true should not be injected
 */
TEST_F(InjectorTests, noInjectionOnFixedCells)
{
    auto data = createInjectorWithGenerator({100.0f, 100.0f}, 3);

    // Add target creature with fixed constructor
    data.addCreature({
        ObjectDesc().id(100).pos({100.0f, 103.0f}).fixed(true).type(CellDesc().constructor(ConstructorDesc().geneIndex(0))),
        ObjectDesc().id(101).pos({101.0f, 103.0f}).fixed(true),
    }, CreatureDesc().id(2));
    data.addConnection(100, 101);

    auto origConstructor = data.getObjectRef(100).getCellRef()._constructor.value();

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualConstructor = actualData.getObjectRef(100).getCellRef()._constructor.value();

    // Constructor's geneIndex should remain unchanged
    EXPECT_EQ(origConstructor._geneIndex, actualConstructor._geneIndex);
}

/**
 * Test: Injection blocked by own connections
 * Injection should be blocked when same-creature cell connections cross the ray to the target
 */
TEST_F(InjectorTests, rayBlockedBySameCreatureConnections)
{
    // Create injector with connections that block the injection ray
    auto data = Desc().addCreature({
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().neuralNetwork(NeuralNetDesc().bias(0, 1.0f)).cellType(InjectorDesc().geneIndex(3))),
        ObjectDesc().id(2).pos({101.0f, 100.0f}),
        // Create a connection that crosses the ray path to target at (100, 97)
        ObjectDesc().id(3).pos({99.0f, 99.0f}),
        ObjectDesc().id(4).pos({101.0f, 99.0f}),
    }, CreatureDesc().id(1));
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(3, 4);
    data.addConnection(1, 4);

    // Add target creature below (ray to target is blocked by connection 3-4)
    data.addCreature({
        ObjectDesc().id(100).pos({100.0f, 97.0f}).type(CellDesc().constructor(ConstructorDesc().geneIndex(0))),
        ObjectDesc().id(101).pos({101.0f, 97.0f}),
    }, CreatureDesc().id(2));
    data.addConnection(100, 101);

    auto origConstructor = data.getObjectRef(100).getCellRef()._constructor.value();

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualConstructor = actualData.getObjectRef(100).getCellRef()._constructor.value();

    // Constructor's geneIndex should remain unchanged because ray is blocked
    EXPECT_EQ(origConstructor._geneIndex, actualConstructor._geneIndex);
}

/**
 * Test: Injection resets construction progress
 * After injection, the target constructor's currentNodeIndex and related fields should be reset
 */
TEST_F(InjectorTests, injectionResetsConstructionProgress)
{
    auto data = createInjectorWithGenerator({100.0f, 100.0f}, 2);

    // Add target creature with constructor that has some progress
    data.addCreature({
        ObjectDesc().id(100).pos({100.0f, 103.0f}).type(CellDesc().constructor(ConstructorDesc().geneIndex(5).currentNodeIndex(3).currentConcatenation(2))),
        ObjectDesc().id(101).pos({101.0f, 103.0f}),
    }, CreatureDesc().id(2));
    data.addConnection(100, 101);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualConstructor = actualData.getObjectRef(100).getCellRef()._constructor.value();

    // Constructor's geneIndex should be the injector's geneIndex
    EXPECT_EQ(2, actualConstructor._geneIndex);

    // Construction progress should be reset
    EXPECT_EQ(0, actualConstructor._currentNodeIndex);
    EXPECT_EQ(0, actualConstructor._currentConcatenation);
}

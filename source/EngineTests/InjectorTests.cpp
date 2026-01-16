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
    Description createInjectorWithGenerator(RealVector2D const& injectorPos, int geneIndex = 0, int injectorColor = 0)
    {
        auto data = Description().addCreature({
            ObjectDescription().id(1).pos(injectorPos).color(injectorColor).type(CellDescription().cellType(InjectorDescription().geneIndex(geneIndex))),
            ObjectDescription().id(2).pos({injectorPos.x + 1.0f, injectorPos.y}).color(injectorColor).type(CellDescription().cellType(GeneratorDescription().autoTriggerInterval(3))),
        }, CreatureDescription().id(1));
        data.addConnection(1, 2);
        return data;
    }

    // Helper to create a target creature with a constructor at a given position
    Description createTargetCreatureWithConstructor(RealVector2D const& pos, uint64_t creatureId = 2, int color = 0, float usableEnergy = 100.0f)
    {
        auto data = Description().addCreature({
            ObjectDescription().id(100).pos(pos).color(color).type(CellDescription().usableEnergy(usableEnergy).cellType(ConstructorDescription())),
            ObjectDescription().id(101).pos({pos.x + 1.0f, pos.y}).color(color).type(CellDescription().usableEnergy(usableEnergy)),
        }, CreatureDescription().id(creatureId));
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
    if (actualInjector.getCellRef()._signalState == SignalState_Active) {
        EXPECT_TRUE(approxCompare(0.0f, actualInjector.getCellRef()._signal._channels[Channels::InjectorSuccess]));
    }
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
    auto actualTargetConstructor = std::get<ConstructorDescription>(actualData.getObjectRef(100).getCellRef()._cellType);

    // Injector should have a signal with success value > 0
    ASSERT_TRUE(actualInjector.getCellRef()._signalState == SignalState_Active);
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
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(InjectorDescription().geneIndex(3))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().cellType(GeneratorDescription().autoTriggerInterval(3))),
        ObjectDescription().id(3).pos({100.0f, 103.0f}).type(CellDescription().cellType(ConstructorDescription().geneIndex(0))),  // Same creature
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    auto origConstructor = std::get<ConstructorDescription>(data.getObjectRef(3).getCellRef()._cellType);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualConstructor = std::get<ConstructorDescription>(actualData.getObjectRef(3).getCellRef()._cellType);

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
        ObjectDescription().id(100).pos({100.0f, 103.0f}).fixed(true).type(CellDescription().cellType(ConstructorDescription().geneIndex(0))),
        ObjectDescription().id(101).pos({101.0f, 103.0f}).fixed(true),
    }, CreatureDescription().id(2));
    data.addConnection(100, 101);

    auto origConstructor = std::get<ConstructorDescription>(data.getObjectRef(100).getCellRef()._cellType);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualConstructor = std::get<ConstructorDescription>(actualData.getObjectRef(100).getCellRef()._cellType);

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
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(InjectorDescription().geneIndex(3))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().cellType(GeneratorDescription().autoTriggerInterval(3))),
        // Create a connection that crosses the ray path to target at (100, 97)
        ObjectDescription().id(3).pos({99.0f, 99.0f}),
        ObjectDescription().id(4).pos({101.0f, 99.0f}),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(3, 4);
    data.addConnection(1, 4);

    // Add target creature below (ray to target is blocked by connection 3-4)
    data.addCreature({
        ObjectDescription().id(100).pos({100.0f, 97.0f}).type(CellDescription().cellType(ConstructorDescription().geneIndex(0))),
        ObjectDescription().id(101).pos({101.0f, 97.0f}),
    }, CreatureDescription().id(2));
    data.addConnection(100, 101);

    auto origConstructor = std::get<ConstructorDescription>(data.getObjectRef(100).getCellRef()._cellType);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualConstructor = std::get<ConstructorDescription>(actualData.getObjectRef(100).getCellRef()._cellType);

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
        ObjectDescription().id(100).pos({100.0f, 103.0f}).type(CellDescription().cellType(ConstructorDescription().geneIndex(5).currentNodeIndex(3).currentConcatenation(2))),
        ObjectDescription().id(101).pos({101.0f, 103.0f}),
    }, CreatureDescription().id(2));
    data.addConnection(100, 101);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualConstructor = std::get<ConstructorDescription>(actualData.getObjectRef(100).getCellRef()._cellType);

    // Constructor's geneIndex should be the injector's geneIndex
    EXPECT_EQ(2, actualConstructor._geneIndex);

    // Construction progress should be reset
    EXPECT_EQ(0, actualConstructor._currentNodeIndex);
    EXPECT_EQ(0, actualConstructor._currentConcatenation);
}

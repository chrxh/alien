#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class MemoryTests : public IntegrationTestFramework
{
public:
    MemoryTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~MemoryTests() = default;

protected:
    // Helper to create a memory cell with a signal source cell sending a signal
    Description createMemoryCellWithIncomingSignal(MemoryModeDescription const& mode, std::vector<float> const& signal)
    {
        auto data = Description().addCreature(CreatureDescription().id(1).cells({
            CellDescription().id(1).pos({100.0f, 100.0f}).cellType(MemoryDescription().mode(mode)),
            CellDescription().id(2).pos({101.0f, 100.0f}).signalAndState(signal),
        }));
        data.addConnection(1, 2);
        return data;
    }
};

//********************
//* SignalIntegrator *
//********************

TEST_F(MemoryTests, signalIntegrator_firstSignal_storesSignalInMemory)
{
    std::vector<float> signal = {1.0f, -0.5f, 0.25f, 0.0f, 0.75f, -1.0f, 0.5f, 0.0f};
    auto data = createMemoryCellWithIncomingSignal(SignalIntegratorDescription().newSignalWeight(0.5f), signal);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    // After first signal, the memory cell should store the incoming signal
    auto memoryCell = actualData.getCellRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    EXPECT_EQ(1u, memory._memoryEntries.size());
    EXPECT_TRUE(approxCompare(signal, memory._memoryEntries[0]._channels));
}

TEST_F(MemoryTests, signalIntegrator_secondSignal_integratesWithWeight)
{
    // Setup: First signal stored, then second signal integrates
    std::vector<float> firstSignal = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> secondSignal = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float newSignalWeight = 0.5f;

    // Create memory cell with first signal already stored
    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(MemoryDescription()
                          .mode(SignalIntegratorDescription().newSignalWeight(newSignalWeight))
                          .memoryEntries({MemoryEntryDescription().channels(firstSignal)})),
        CellDescription().id(2).pos({101.0f, 100.0f}).signalAndState(secondSignal),
    }));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    // Expected: (1-0.5)*firstSignal + 0.5*secondSignal = 0.5*1.0 + 0.5*0.0 = 0.5
    auto memoryCell = actualData.getCellRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    EXPECT_EQ(1u, memory._memoryEntries.size());

    std::vector<float> expectedSignal = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_TRUE(approxCompare(expectedSignal, memory._memoryEntries[0]._channels));
}

TEST_F(MemoryTests, signalIntegrator_outputSignalMatchesIntegratedValue)
{
    // After integration, the cell's output signal should match the integrated value
    std::vector<float> storedSignal = {0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> incomingSignal = {0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float newSignalWeight = 0.25f;

    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(MemoryDescription()
                          .mode(SignalIntegratorDescription().newSignalWeight(newSignalWeight))
                          .memoryEntries({MemoryEntryDescription().channels(storedSignal)})),
        CellDescription().id(2).pos({101.0f, 100.0f}).signalAndState(incomingSignal),
    }));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    // Expected: (1-0.25)*0.8 + 0.25*0.2 = 0.6 + 0.05 = 0.65
    auto memoryCell = actualData.getCellRef(1);
    std::vector<float> expectedSignal = {0.65f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_TRUE(approxCompare(expectedSignal, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalIntegrator_weightOfOne_replacesStoredSignal)
{
    // With newSignalWeight = 1.0, the stored signal should be completely replaced
    std::vector<float> storedSignal = {1.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> incomingSignal = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(
                MemoryDescription().mode(SignalIntegratorDescription().newSignalWeight(1.0f)).memoryEntries({MemoryEntryDescription().channels(storedSignal)})),
        CellDescription().id(2).pos({101.0f, 100.0f}).signalAndState(incomingSignal),
    }));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    // With weight 1.0, stored value = (1-1)*stored + 1*incoming = incoming
    auto memoryCell = actualData.getCellRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    EXPECT_TRUE(approxCompare(incomingSignal, memory._memoryEntries[0]._channels));
}

TEST_F(MemoryTests, signalIntegrator_weightOfZero_preservesStoredSignal)
{
    // With newSignalWeight = 0.0, the stored signal should be preserved
    std::vector<float> storedSignal = {1.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> incomingSignal = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(
                MemoryDescription().mode(SignalIntegratorDescription().newSignalWeight(0.0f)).memoryEntries({MemoryEntryDescription().channels(storedSignal)})),
        CellDescription().id(2).pos({101.0f, 100.0f}).signalAndState(incomingSignal),
    }));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    // With weight 0.0, stored value = (1-0)*stored + 0*incoming = stored
    auto memoryCell = actualData.getCellRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    EXPECT_TRUE(approxCompare(storedSignal, memory._memoryEntries[0]._channels));
}

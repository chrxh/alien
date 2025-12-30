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

    auto memoryCell = actualData.getCellRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    EXPECT_EQ(1u, memory._memoryEntries.size());

    // Expected: (1-0.25)*0.8 + 0.25*0.2 = 0.6 + 0.05 = 0.65
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

//**************
//* SignalDelay *
//**************

TEST_F(MemoryTests, signalDelay_firstSignal_storesSignalInMemory)
{
    std::vector<float> signal = {1.0f, -0.5f, 0.25f, 0.0f, 0.75f, -1.0f, 0.5f, 0.0f};
    auto data = createMemoryCellWithIncomingSignal(SignalDelayDescription().delay(5), signal);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    // After first signal, the memory cell should store the incoming signal
    auto memoryCell = actualData.getCellRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    auto& signalDelay = std::get<SignalDelayDescription>(memory._mode);

    // Signal should be stored in memory
    EXPECT_EQ(5u, memory._memoryEntries.size());
    EXPECT_EQ(1, signalDelay._numMemoryEntriesInitialized);

    // Verify the signal was stored at index 0
    EXPECT_TRUE(approxCompare(signal, memory._memoryEntries[0]._channels));

    // Verify the output signal (buffer not full yet, so signal should be unchanged)
    EXPECT_EQ(SignalState_Active, memoryCell._signalState);
    EXPECT_TRUE(approxCompare(signal, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalDelay_delayOf0_outputsSameCycleSignal)
{
    std::vector<float> signal = {0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto data = createMemoryCellWithIncomingSignal(SignalDelayDescription().delay(0), signal);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    // With delay of 1, after first cycle, the buffer is full
    auto memoryCell = actualData.getCellRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    auto& signalDelay = std::get<SignalDelayDescription>(memory._mode);

    EXPECT_EQ(0, signalDelay._numMemoryEntriesInitialized);

    // Verify the output signal
    EXPECT_EQ(SignalState_Active, memoryCell._signalState);
    EXPECT_TRUE(approxCompare(signal, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalDelay_delayOf1_outputsDelayedSignal)
{
    // First signal
    std::vector<float> signal1 = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto data = createMemoryCellWithIncomingSignal(SignalDelayDescription().delay(1), signal1);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3);

    // Second signal
    std::vector<float> signal2 = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto actualData = _simulationFacade->getSimulationData();
    actualData.getCellRef(2).signalAndState(signal2);
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(1);

    actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getCellRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    auto& signalDelay = std::get<SignalDelayDescription>(memory._mode);

    // Buffer should be fully initialized after 2 signals
    EXPECT_EQ(1, signalDelay._numMemoryEntriesInitialized);

    // The output signal should be signal1 (the first signal, delayed by 2)
    EXPECT_EQ(SignalState_Active, memoryCell._signalState);
    EXPECT_TRUE(approxCompare(signal1, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalDelay_delayOf2_outputsCorrectlyDelayedSignal)
{
    // Send 4 signals sequentially
    std::vector<float> signal1 = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> signal2 = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> signal3 = {0.25f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> signal4 = {0.125f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    auto data = createMemoryCellWithIncomingSignal(SignalDelayDescription().delay(2), signal1);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3);

    // Second signal
    auto actualData = _simulationFacade->getSimulationData();
    actualData.getCellRef(2).signalAndState(signal2);
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(3);

    // Third signal
    actualData = _simulationFacade->getSimulationData();
    actualData.getCellRef(2).signalAndState(signal3);
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(1);

    // After 3 signals, the buffer is full and output should be signal1
    actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getCellRef(1);
    auto& signalDelay = std::get<SignalDelayDescription>(std::get<MemoryDescription>(memoryCell._cellType)._mode);
    EXPECT_EQ(2, signalDelay._numMemoryEntriesInitialized);
    EXPECT_EQ(SignalState_Active, memoryCell._signalState);
    EXPECT_TRUE(approxCompare(signal1, memoryCell._signal._channels));

    // Waiting
    _simulationFacade->calcTimesteps(2);
    actualData = _simulationFacade->getSimulationData();

    // Fourth signal - should output signal2
    actualData.getCellRef(2).signalAndState(signal4);
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(1);

    actualData = _simulationFacade->getSimulationData();
    memoryCell = actualData.getCellRef(1);
    EXPECT_EQ(SignalState_Active, memoryCell._signalState);
    EXPECT_TRUE(approxCompare(signal2, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalDelay_delayOf2_noOutputBeforeBufferFull)
{
    // With delay of 2, first 2 signals should not produce a delayed output (buffer not filled)
    std::vector<float> signal1 = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> signal2 = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    auto data = createMemoryCellWithIncomingSignal(SignalDelayDescription().delay(2), signal1);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getCellRef(1);
    auto& signalDelay = std::get<SignalDelayDescription>(std::get<MemoryDescription>(memoryCell._cellType)._mode);

    // After first signal, buffer not full yet
    EXPECT_EQ(1, signalDelay._numMemoryEntriesInitialized);
    // Signal should still be the incoming signal1 (not modified by delay output)
    EXPECT_TRUE(approxCompare(signal1, memoryCell._signal._channels));

    // Waiting
    _simulationFacade->calcTimesteps(2);
    actualData = _simulationFacade->getSimulationData();

    // Second signal
    actualData.getCellRef(2).signalAndState(signal2);
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(1);

    actualData = _simulationFacade->getSimulationData();
    memoryCell = actualData.getCellRef(1);
    signalDelay = std::get<SignalDelayDescription>(std::get<MemoryDescription>(memoryCell._cellType)._mode);

    // After second signal, buffer is full
    EXPECT_EQ(2, signalDelay._numMemoryEntriesInitialized);

    // Signal should still be the incoming signal2 (not modified by delay output)
    EXPECT_EQ(SignalState_Active, memoryCell._signalState);
    EXPECT_TRUE(approxCompare(signal2, memoryCell._signal._channels));
}

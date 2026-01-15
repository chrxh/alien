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
    // Helper to create a memory cell with custom memory entries and settings
    Description createMemoryCellWithIncomingSignal(
        MemoryModeDescription const& mode,
        std::vector<float> const& signal,
        std::vector<SignalEntryDescription> const& signalEntries = {})
    {
        auto data = Description().addCreature({
            ObjectDescription().id(1).pos({100.0f, 100.0f}).cellType(MemoryDescription().mode(mode).signalEntries(signalEntries)),
            ObjectDescription().id(2).pos({101.0f, 100.0f}).signalAndState(signal),
        }, CreatureDescription().id(1));
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
    auto memoryCell = actualData.getObjectRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    EXPECT_EQ(1u, memory._signalEntries.size());
    EXPECT_TRUE(approxCompare(signal, memory._signalEntries[0]._channels));
}


TEST_F(MemoryTests, signalIntegrator_secondSignal_integratesWithWeight)
{
    // Setup: First signal stored, then second signal integrates
    std::vector<float> storedSignal = {0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> incomingSignal = {0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float newSignalWeight = 0.25f;

    auto data = Description().addCreature({
        ObjectDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(MemoryDescription()
                          .mode(SignalIntegratorDescription().newSignalWeight(newSignalWeight))
                          .signalEntries({SignalEntryDescription().channels(storedSignal)})),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).signalAndState(incomingSignal),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    EXPECT_EQ(1u, memory._signalEntries.size());

    // Expected: (1-0.25)*0.8 + 0.25*0.2 = 0.6 + 0.05 = 0.65
    std::vector<float> expectedSignal = {0.65f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_TRUE(approxCompare(expectedSignal, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalIntegrator_weightOfOne_replacesStoredSignal)
{
    // With newSignalWeight = 1.0, the stored signal should be completely replaced
    std::vector<float> storedSignal = {1.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> incomingSignal = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    auto data = Description().addCreature({
        ObjectDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(
                MemoryDescription().mode(SignalIntegratorDescription().newSignalWeight(1.0f)).signalEntries({SignalEntryDescription().channels(storedSignal)})),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).signalAndState(incomingSignal),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    // With weight 1.0, stored value = (1-1)*stored + 1*incoming = incoming
    auto memoryCell = actualData.getObjectRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    EXPECT_TRUE(approxCompare(incomingSignal, memory._signalEntries[0]._channels));
}

TEST_F(MemoryTests, signalIntegrator_weightOfZero_preservesStoredSignal)
{
    // With newSignalWeight = 0.0, the stored signal should be preserved
    std::vector<float> storedSignal = {1.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> incomingSignal = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    auto data = Description().addCreature({
        ObjectDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(
                MemoryDescription().mode(SignalIntegratorDescription().newSignalWeight(0.0f)).signalEntries({SignalEntryDescription().channels(storedSignal)})),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).signalAndState(incomingSignal),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    // With weight 0.0, stored value = (1-0)*stored + 0*incoming = stored
    auto memoryCell = actualData.getObjectRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    EXPECT_TRUE(approxCompare(storedSignal, memory._signalEntries[0]._channels));
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
    auto memoryCell = actualData.getObjectRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    auto& signalDelay = std::get<SignalDelayDescription>(memory._mode);

    // Signal should be stored in memory
    EXPECT_EQ(5u, memory._signalEntries.size());
    EXPECT_EQ(1, signalDelay._numSignalEntriesInitialized);

    // Verify the signal was stored at index 0
    EXPECT_TRUE(approxCompare(signal, memory._signalEntries[0]._channels));

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
    auto memoryCell = actualData.getObjectRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    auto& signalDelay = std::get<SignalDelayDescription>(memory._mode);

    EXPECT_EQ(0, signalDelay._numSignalEntriesInitialized);

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
    actualData.getObjectRef(2).signalAndState(signal2);
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(1);

    actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getObjectRef(1);
    auto& memory = std::get<MemoryDescription>(memoryCell._cellType);
    auto& signalDelay = std::get<SignalDelayDescription>(memory._mode);

    // Buffer should be fully initialized after 2 signals
    EXPECT_EQ(1, signalDelay._numSignalEntriesInitialized);

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
    actualData.getObjectRef(2).signalAndState(signal2);
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(3);

    // Third signal
    actualData = _simulationFacade->getSimulationData();
    actualData.getObjectRef(2).signalAndState(signal3);
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(1);

    // After 3 signals, the buffer is full and output should be signal1
    actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getObjectRef(1);
    auto& signalDelay = std::get<SignalDelayDescription>(std::get<MemoryDescription>(memoryCell._cellType)._mode);
    EXPECT_EQ(2, signalDelay._numSignalEntriesInitialized);
    EXPECT_EQ(SignalState_Active, memoryCell._signalState);
    EXPECT_TRUE(approxCompare(signal1, memoryCell._signal._channels));

    // Waiting
    _simulationFacade->calcTimesteps(2);
    actualData = _simulationFacade->getSimulationData();

    // Fourth signal - should output signal2
    actualData.getObjectRef(2).signalAndState(signal4);
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(1);

    actualData = _simulationFacade->getSimulationData();
    memoryCell = actualData.getObjectRef(1);
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
    auto memoryCell = actualData.getObjectRef(1);
    auto& signalDelay = std::get<SignalDelayDescription>(std::get<MemoryDescription>(memoryCell._cellType)._mode);

    // After first signal, buffer not full yet
    EXPECT_EQ(1, signalDelay._numSignalEntriesInitialized);
    // Signal should still be the incoming signal1 (not modified by delay output)
    EXPECT_TRUE(approxCompare(signal1, memoryCell._signal._channels));

    // Waiting
    _simulationFacade->calcTimesteps(2);
    actualData = _simulationFacade->getSimulationData();

    // Second signal
    actualData.getObjectRef(2).signalAndState(signal2);
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(1);

    actualData = _simulationFacade->getSimulationData();
    memoryCell = actualData.getObjectRef(1);
    signalDelay = std::get<SignalDelayDescription>(std::get<MemoryDescription>(memoryCell._cellType)._mode);

    // After second signal, buffer is full
    EXPECT_EQ(2, signalDelay._numSignalEntriesInitialized);

    // Signal should still be the incoming signal2 (not modified by delay output)
    EXPECT_EQ(SignalState_Active, memoryCell._signalState);
    EXPECT_TRUE(approxCompare(signal2, memoryCell._signal._channels));
}

//*****************
//* SignalRecorder *
//*****************

TEST_F(MemoryTests, signalRecorder_positiveChannel0_startsRecording)
{
    // Setup: memory with 5 entries, positive channel[0] should start recording
    std::vector<float> signal = {1.0f, 0.5f, -0.25f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<SignalEntryDescription> signalEntries(5);
    auto data = createMemoryCellWithIncomingSignal(SignalRecorderDescription().readOnly(false), signal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);
    auto& signalRecorder = std::get<SignalRecorderDescription>(memoryDesc._mode);

    // Should be in recording state with 1 entry recorded
    EXPECT_EQ(SignalRecorderState_Recording, signalRecorder._state);
    EXPECT_EQ(1, signalRecorder._numWrittenSignalEntries);
    EXPECT_TRUE(approxCompare(signal, memoryDesc._signalEntries[0]._channels));
}

TEST_F(MemoryTests, signalRecorder_recordingCompletes_whenMemoryFull)
{
    // Setup: memory with 2 entries
    std::vector<float> signal1 = {1.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> signal2 = {0.5f, 0.25f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<SignalEntryDescription> signalEntries(2);
    auto data = createMemoryCellWithIncomingSignal(SignalRecorderDescription().readOnly(false), signal1, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3);

    // Second signal - should record and complete
    auto actualData = _simulationFacade->getSimulationData();
    actualData.getObjectRef(2).signalAndState(signal2);
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(1);

    actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getObjectRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);
    auto& signalRecorder = std::get<SignalRecorderDescription>(memoryDesc._mode);

    // Recording should be complete, back to idle
    EXPECT_EQ(SignalRecorderState_Idle, signalRecorder._state);
    EXPECT_EQ(2, signalRecorder._numWrittenSignalEntries);
    EXPECT_TRUE(approxCompare(signal1, memoryDesc._signalEntries[0]._channels));
    EXPECT_TRUE(approxCompare(signal2, memoryDesc._signalEntries[1]._channels));
}

TEST_F(MemoryTests, signalRecorder_negativeChannel0_startsReading)
{
    // Setup: memory with pre-recorded entries
    std::vector<float> storedSignal = {0.8f, 0.4f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> triggerSignal = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<SignalEntryDescription> signalEntries = {
        SignalEntryDescription().channels(storedSignal),
        SignalEntryDescription().channels({0.1f, 0.2f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}),
    };
    auto data = createMemoryCellWithIncomingSignal(
        SignalRecorderDescription().readOnly(false).numWrittenSignalEntries(2), triggerSignal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);
    auto& signalRecorder = std::get<SignalRecorderDescription>(memoryDesc._mode);

    // Should be in reading state, output first stored signal
    EXPECT_EQ(SignalRecorderState_Reading, signalRecorder._state);
    EXPECT_EQ(1, signalRecorder._numReadSignalEntries);
    EXPECT_TRUE(approxCompare(storedSignal, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalRecorder_readingCompletes_resetsToIdle)
{
    // Setup: memory with 2 pre-recorded entries
    std::vector<float> storedSignal1 = {0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> storedSignal2 = {0.4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> triggerSignal = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<SignalEntryDescription> signalEntries = {
        SignalEntryDescription().channels(storedSignal1),
        SignalEntryDescription().channels(storedSignal2),
    };
    auto data = createMemoryCellWithIncomingSignal(
        SignalRecorderDescription().readOnly(false).numWrittenSignalEntries(2), triggerSignal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3);

    // Second read - should read second entry and complete
    auto actualData = _simulationFacade->getSimulationData();
    actualData.getObjectRef(2).signalAndState(triggerSignal);
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(1);

    actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getObjectRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);
    auto& signalRecorder = std::get<SignalRecorderDescription>(memoryDesc._mode);

    // Reading should be complete, back to idle
    EXPECT_EQ(SignalRecorderState_Idle, signalRecorder._state);
    EXPECT_EQ(0, signalRecorder._numReadSignalEntries);
    EXPECT_TRUE(approxCompare(storedSignal2, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalRecorder_initialRecordedEntries_canBeRead)
{
    // Test that memory cell with numWrittenSignalEntries > 0 can be read immediately
    std::vector<float> storedSignal = {0.75f, 0.5f, 0.25f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> triggerSignal = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<SignalEntryDescription> signalEntries = {SignalEntryDescription().channels(storedSignal)};
    auto data = createMemoryCellWithIncomingSignal(
        SignalRecorderDescription().readOnly(false).numWrittenSignalEntries(1), triggerSignal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    // Should output the pre-stored signal
    EXPECT_TRUE(approxCompare(storedSignal, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalRecorder_stateTransition_ignoresChannel0DuringProcess)
{
    // Start recording, then send negative channel[0] - should continue recording, not switch to reading
    std::vector<float> positiveSignal = {1.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> negativeSignal = {-1.0f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<SignalEntryDescription> signalEntries(3);
    auto data = createMemoryCellWithIncomingSignal(SignalRecorderDescription().readOnly(false), positiveSignal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3);

    // Send negative signal - should continue recording, not switch to reading
    auto actualData = _simulationFacade->getSimulationData();
    actualData.getObjectRef(2).signalAndState(negativeSignal);
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(1);

    actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getObjectRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);
    auto& signalRecorder = std::get<SignalRecorderDescription>(memoryDesc._mode);

    // Should still be recording, with 2 entries recorded (including the negative signal)
    EXPECT_EQ(SignalRecorderState_Recording, signalRecorder._state);
    EXPECT_EQ(2, signalRecorder._numWrittenSignalEntries);
}

TEST_F(MemoryTests, signalRecorder_readOnly_readingInsteadOfRecording)
{
    // Setup: memory with readOnly = true, positive channel[0] should NOT start recording
    std::vector<float> signal = {1.0f, 0.5f, -0.25f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<SignalEntryDescription> signalEntries(3);
    auto data = createMemoryCellWithIncomingSignal(SignalRecorderDescription().readOnly(true).numWrittenSignalEntries(3), signal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);
    auto& signalRecorder = std::get<SignalRecorderDescription>(memoryDesc._mode);

    // Should remain in idle state because readOnly prevents recording
    EXPECT_EQ(SignalRecorderState_Reading, signalRecorder._state);
    EXPECT_EQ(3, signalRecorder._numWrittenSignalEntries);
}

TEST_F(MemoryTests, signalRecorder_readOnly_allowsReading)
{
    // Setup: memory with readOnly = true and pre-recorded entries - reading should work
    std::vector<float> storedSignal = {0.75f, 0.5f, 0.25f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> triggerSignal = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<SignalEntryDescription> signalEntries = {SignalEntryDescription().channels(storedSignal)};
    auto data = createMemoryCellWithIncomingSignal(
        SignalRecorderDescription().readOnly(true).numWrittenSignalEntries(1), triggerSignal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);
    auto& signalRecorder = std::get<SignalRecorderDescription>(memoryDesc._mode);

    // Reading should work even with readOnly = true
    EXPECT_EQ(SignalRecorderState_Idle, signalRecorder._state);  // Reading completes with only 1 entry
    EXPECT_TRUE(approxCompare(storedSignal, memoryCell._signal._channels));
}

//******************
//* SignalStorage *
//******************

TEST_F(MemoryTests, signalStorage_readWithPositiveInput_readsFromIndex)
{
    // Setup: memory with 3 entries, positive channel[0] with value 0.5 should read from index 1
    std::vector<float> entry0 = {0.1f, 0.1f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> entry1 = {0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> entry2 = {0.9f, 0.9f, 0.9f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> inputSignal = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // Should read from index 1
    std::vector<SignalEntryDescription> signalEntries = {
        SignalEntryDescription().channels(entry0),
        SignalEntryDescription().channels(entry1),
        SignalEntryDescription().channels(entry2),
    };
    auto data = createMemoryCellWithIncomingSignal(SignalStorageDescription().readOnly(false), inputSignal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    // With channel[0] = 0.5, index = 0.5 * (3 - 1) = 1
    EXPECT_TRUE(approxCompare(entry1, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalStorage_readWithZeroInput_readsFromIndex0)
{
    // Setup: channel[0] = 0 should read from index 0
    std::vector<float> entry0 = {0.1f, 0.2f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> entry1 = {0.5f, 0.6f, 0.7f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> inputSignal = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // Should read from index 0
    std::vector<SignalEntryDescription> signalEntries = {
        SignalEntryDescription().channels(entry0),
        SignalEntryDescription().channels(entry1),
    };
    auto data = createMemoryCellWithIncomingSignal(SignalStorageDescription().readOnly(false), inputSignal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    EXPECT_TRUE(approxCompare(entry0, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalStorage_readWithMaxInput_readsFromLastIndex)
{
    // Setup: channel[0] = 1.0 should read from last index
    std::vector<float> entry0 = {0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> entry1 = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> entry2 = {0.9f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> inputSignal = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // Should read from index 2
    std::vector<SignalEntryDescription> signalEntries = {
        SignalEntryDescription().channels(entry0),
        SignalEntryDescription().channels(entry1),
        SignalEntryDescription().channels(entry2),
    };
    auto data = createMemoryCellWithIncomingSignal(SignalStorageDescription().readOnly(false), inputSignal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    // With channel[0] = 1.0, index = 1.0 * (3 - 1) = 2
    EXPECT_TRUE(approxCompare(entry2, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalStorage_writeWithNegativeInput_writesToIndex)
{
    // Setup: channel[0] = -0.5 should write to index 1
    std::vector<float> entry0 = {0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> entry1 = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> entry2 = {0.9f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> inputSignal = {-0.5f, 0.7f, 0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // Should write to index 1
    std::vector<SignalEntryDescription> signalEntries = {
        SignalEntryDescription().channels(entry0),
        SignalEntryDescription().channels(entry1),
        SignalEntryDescription().channels(entry2),
    };
    auto data = createMemoryCellWithIncomingSignal(SignalStorageDescription().readOnly(false), inputSignal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);

    // Entry at index 1 should now contain the input signal
    EXPECT_TRUE(approxCompare(inputSignal, memoryDesc._signalEntries[1]._channels));
    // Other entries should be unchanged
    EXPECT_TRUE(approxCompare(entry0, memoryDesc._signalEntries[0]._channels));
    EXPECT_TRUE(approxCompare(entry2, memoryDesc._signalEntries[2]._channels));
}

TEST_F(MemoryTests, signalStorage_readOnly_readsWithPositiveInput)
{
    // In read-only mode, positive input should still read
    std::vector<float> entry0 = {0.1f, 0.2f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> entry1 = {0.5f, 0.6f, 0.7f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> inputSignal = {0.4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<SignalEntryDescription> signalEntries = {
        SignalEntryDescription().channels(entry0),
        SignalEntryDescription().channels(entry1),
    };
    auto data = createMemoryCellWithIncomingSignal(SignalStorageDescription().readOnly(true), inputSignal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    // With 2 entries and input 0.4, index = 0.4 * (2 - 1)) = 0.4
    EXPECT_TRUE(approxCompare(entry0, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalStorage_readOnly_readsWithNegativeInput)
{
    // In read-only mode, negative input should also read (not write)
    std::vector<float> entry0 = {0.1f, 0.2f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> entry1 = {0.5f, 0.6f, 0.7f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> inputSignal = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // Negative, should still read
    std::vector<SignalEntryDescription> signalEntries = {
        SignalEntryDescription().channels(entry0),
        SignalEntryDescription().channels(entry1),
    };
    auto data = createMemoryCellWithIncomingSignal(SignalStorageDescription().readOnly(true), inputSignal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);

    // With input = -1.0 and readOnly = true, index = abs(-1.0) * (2 - 1) = 1
    EXPECT_TRUE(approxCompare(entry1, memoryCell._signal._channels));

    // Memory should be unchanged (no write occurred)
    EXPECT_TRUE(approxCompare(entry0, memoryDesc._signalEntries[0]._channels));
    EXPECT_TRUE(approxCompare(entry1, memoryDesc._signalEntries[1]._channels));
}

TEST_F(MemoryTests, signalStorage_singleEntry_alwaysAccessesIndex0)
{
    // With a single entry, any input should access index 0
    std::vector<float> entry0 = {0.5f, 0.6f, 0.7f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> inputSignal = {0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<SignalEntryDescription> signalEntries = {
        SignalEntryDescription().channels(entry0),
    };
    auto data = createMemoryCellWithIncomingSignal(SignalStorageDescription().readOnly(false), inputSignal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    // With 1 entry, index = 0.8 * (1 - 1) = 0
    EXPECT_TRUE(approxCompare(entry0, memoryCell._signal._channels));
}

TEST_F(MemoryTests, signalStorage_writeWithMaxNegativeInput_writesToLastIndex)
{
    // channel[0] = -1.0 should write to last index
    std::vector<float> entry0 = {0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> entry1 = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> inputSignal = {-1.0f, 0.9f, 0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // Should write to index 1
    std::vector<SignalEntryDescription> signalEntries = {
        SignalEntryDescription().channels(entry0),
        SignalEntryDescription().channels(entry1),
    };
    auto data = createMemoryCellWithIncomingSignal(SignalStorageDescription().readOnly(false), inputSignal, signalEntries);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    auto memoryCell = actualData.getObjectRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);

    // Entry at index 1 should now contain the input signal
    EXPECT_TRUE(approxCompare(inputSignal, memoryDesc._signalEntries[1]._channels));
    // Entry at index 0 should be unchanged
    EXPECT_TRUE(approxCompare(entry0, memoryDesc._signalEntries[0]._channels));
}

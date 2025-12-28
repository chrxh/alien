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
    // Helper to create a MemoryEntryDescription with specific channel values
    MemoryEntryDescription createMemoryEntry(std::vector<float> const& channels)
    {
        auto entry = MemoryEntryDescription();
        entry._channels = channels;
        entry._channels.resize(MAX_CHANNELS, 0);
        return entry;
    }
};

/**
 * Tests for MemoryMode_SignalIntegrator
 */

TEST_F(MemoryTests, signalIntegrator_noSignal_memoryUnchanged)
{
    // When no signal is received, memory should remain unchanged
    std::vector<float> initialMemory = {1.0f, 2.0f, 3.0f, 4.0f, 0, 0, 0, 0};
    Description data;
    data._cells = {
        CellDescription()
            .id(1)
            .cellType(MemoryDescription()
                          .mode(SignalIntegratorDescription().newSignalWeight(0.5f))
                          .memoryEntries({createMemoryEntry(initialMemory)})),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getCellRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);

    ASSERT_EQ(1, memoryDesc._memoryEntries.size());
    EXPECT_TRUE(approxCompare(initialMemory, memoryDesc._memoryEntries[0]._channels));
}

TEST_F(MemoryTests, signalIntegrator_withSignal_fullWeight)
{
    // With newSignalWeight = 1.0, memory should be completely replaced by incoming signal
    std::vector<float> initialMemory = {1.0f, 2.0f, 3.0f, 4.0f, 0, 0, 0, 0};
    std::vector<float> incomingSignal = {5.0f, 6.0f, 7.0f, 8.0f, 0, 0, 0, 0};
    Description data;
    data._cells = {
        CellDescription()
            .id(1)
            .pos({0, 0})
            .signalAndState(incomingSignal)
            .cellType(MemoryDescription()
                          .mode(SignalIntegratorDescription().newSignalWeight(1.0f))
                          .memoryEntries({createMemoryEntry(initialMemory)})),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getCellRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);

    ASSERT_EQ(1, memoryDesc._memoryEntries.size());
    EXPECT_TRUE(approxCompare(incomingSignal, memoryDesc._memoryEntries[0]._channels));
}

TEST_F(MemoryTests, signalIntegrator_withSignal_zeroWeight)
{
    // With newSignalWeight = 0.0, memory should remain unchanged
    std::vector<float> initialMemory = {1.0f, 2.0f, 3.0f, 4.0f, 0, 0, 0, 0};
    std::vector<float> incomingSignal = {5.0f, 6.0f, 7.0f, 8.0f, 0, 0, 0, 0};
    Description data;
    data._cells = {
        CellDescription()
            .id(1)
            .pos({0, 0})
            .signalAndState(incomingSignal)
            .cellType(MemoryDescription()
                          .mode(SignalIntegratorDescription().newSignalWeight(0.0f))
                          .memoryEntries({createMemoryEntry(initialMemory)})),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getCellRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);

    ASSERT_EQ(1, memoryDesc._memoryEntries.size());
    EXPECT_TRUE(approxCompare(initialMemory, memoryDesc._memoryEntries[0]._channels));
}

TEST_F(MemoryTests, signalIntegrator_withSignal_halfWeight)
{
    // With newSignalWeight = 0.5, memory should be the average of old and new
    std::vector<float> initialMemory = {0.0f, 4.0f, 0.0f, 0.0f, 0, 0, 0, 0};
    std::vector<float> incomingSignal = {2.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0, 0};
    std::vector<float> expectedMemory = {1.0f, 2.0f, 0.0f, 0.0f, 0, 0, 0, 0};
    Description data;
    data._cells = {
        CellDescription()
            .id(1)
            .pos({0, 0})
            .signalAndState(incomingSignal)
            .cellType(MemoryDescription()
                          .mode(SignalIntegratorDescription().newSignalWeight(0.5f))
                          .memoryEntries({createMemoryEntry(initialMemory)})),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getCellRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);

    ASSERT_EQ(1, memoryDesc._memoryEntries.size());
    EXPECT_TRUE(approxCompare(expectedMemory, memoryDesc._memoryEntries[0]._channels));
}

TEST_F(MemoryTests, signalIntegrator_withSignal_quarterWeight)
{
    // With newSignalWeight = 0.25, memory should be weighted towards old values
    std::vector<float> initialMemory = {4.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0, 0};
    std::vector<float> incomingSignal = {0.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0, 0};
    // Expected: 0.25 * 0 + 0.75 * 4 = 3
    std::vector<float> expectedMemory = {3.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0, 0};
    Description data;
    data._cells = {
        CellDescription()
            .id(1)
            .pos({0, 0})
            .signalAndState(incomingSignal)
            .cellType(MemoryDescription()
                          .mode(SignalIntegratorDescription().newSignalWeight(0.25f))
                          .memoryEntries({createMemoryEntry(initialMemory)})),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getCellRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);

    ASSERT_EQ(1, memoryDesc._memoryEntries.size());
    EXPECT_TRUE(approxCompare(expectedMemory, memoryDesc._memoryEntries[0]._channels));
}

TEST_F(MemoryTests, signalIntegrator_noMemoryEntries_noEffect)
{
    // When there are no memory entries, nothing should happen
    std::vector<float> incomingSignal = {1.0f, 2.0f, 3.0f, 4.0f, 0, 0, 0, 0};
    Description data;
    data._cells = {
        CellDescription()
            .id(1)
            .pos({0, 0})
            .signalAndState(incomingSignal)
            .cellType(MemoryDescription().mode(SignalIntegratorDescription().newSignalWeight(0.5f))),
    };

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getCellRef(1);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);

    // Should have no memory entries (or empty)
    EXPECT_EQ(0, memoryDesc._memoryEntries.size());
}

TEST_F(MemoryTests, signalIntegrator_propagatedSignal_integratesCorrectly)
{
    // Test that signal integrator works correctly with propagated signals
    std::vector<float> signal = {8.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0, 0};
    std::vector<float> initialMemory = {0.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0, 0};
    std::vector<float> expectedMemory = {4.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0, 0};  // 0.5 * 8 + 0.5 * 0 = 4
    Description data;
    data._cells = {
        CellDescription().id(1).pos({0, 0}).signalAndState(signal),
        CellDescription()
            .id(2)
            .pos({1, 0})
            .cellType(MemoryDescription()
                          .mode(SignalIntegratorDescription().newSignalWeight(0.5f))
                          .memoryEntries({createMemoryEntry(initialMemory)})),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto memoryCell = actualData.getCellRef(2);
    auto& memoryDesc = std::get<MemoryDescription>(memoryCell._cellType);

    ASSERT_EQ(1, memoryDesc._memoryEntries.size());
    EXPECT_TRUE(approxCompare(expectedMemory, memoryDesc._memoryEntries[0]._channels));
}

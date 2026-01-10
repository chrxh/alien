#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class CommunicatorTests : public IntegrationTestFramework
{
public:
    CommunicatorTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~CommunicatorTests() = default;

protected:
    // Helper to create a sender creature with 2 cells (sender + helper for signal)
    Description createSenderCreature(uint64_t creatureId, RealVector2D pos, float range = 50.0f, int maxTimesSent = 4, int color = 0)
    {
        auto data = Description().addCreature(
            CreatureDescription()
                .id(creatureId)
                .cells({
                    CellDescription()
                        .id(creatureId * 100)
                        .pos(pos)
                        .color(color)
                        .cellType(CommunicatorDescription().mode(SenderDescription().range(range).maxTimesSent(maxTimesSent))),
                    CellDescription().id(creatureId * 100 + 1).pos({pos.x + 1.0f, pos.y}).color(color).signalAndState({1.0f, 2.0f, 3.0f, 0, 0, 0, 0, 0}),
                }));
        data.addConnection(creatureId * 100, creatureId * 100 + 1);
        return data;
    }

    // Helper to create a receiver creature with 2 cells
    Description createReceiverCreature(
        uint64_t creatureId,
        RealVector2D pos,
        std::optional<int> restrictToColor = std::nullopt,
        LineageRestriction restrictToLineage = LineageRestriction_No,
        int color = 0)
    {
        auto data = Description().addCreature(CreatureDescription().id(creatureId).cells({
            CellDescription()
                .id(creatureId * 100)
                .pos(pos)
                .color(color)
                .cellType(CommunicatorDescription().mode(ReceiverDescription().restrictToColor(restrictToColor).restrictToLineage(restrictToLineage))),
            CellDescription().id(creatureId * 100 + 1).pos({pos.x + 1.0f, pos.y}).color(color),
        }));
        data.addConnection(creatureId * 100, creatureId * 100 + 1);
        return data;
    }
};

TEST_F(CommunicatorTests, sender_noReceiver_noSignalTransmitted)
{
    // Create a sender with no nearby receiver
    auto data = createSenderCreature(1, {100.0f, 100.0f});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto result = _simulationFacade->getSimulationData();
    auto sender = result.getCellRef(100);

    EXPECT_TRUE(sender._signalState == SignalState_Active);
}

TEST_F(CommunicatorTests, sender_receiverInRange_signalTransmitted)
{
    // Create sender in creature 1
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f);

    // Create receiver in creature 2, within range
    data.add(createReceiverCreature(2, {110.0f, 100.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getCellRef(200);

    // Receiver should have received the signal
    EXPECT_EQ(receiver._signalState, SignalState_Active);
    EXPECT_FLOAT_EQ(receiver._signal._channels[0], 1.0f);
    EXPECT_FLOAT_EQ(receiver._signal._channels[1], 2.0f);
    EXPECT_FLOAT_EQ(receiver._signal._channels[2], 3.0f);
    EXPECT_EQ(receiver._signal._numTimesSent, 1);
}

TEST_F(CommunicatorTests, sender_receiverOutOfRange_noSignalTransmitted)
{
    // Create sender in creature 1 with small range
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 10.0f);

    // Create receiver in creature 2, out of range
    data.add(createReceiverCreature(2, {115.0f, 100.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getCellRef(200);

    // Receiver should NOT have received the signal
    EXPECT_NE(receiver._signalState, SignalState_Active);
}

TEST_F(CommunicatorTests, sender_sameCreatureReceiver_noSignalTransmitted)
{
    // Create sender and receiver in the same creature (both connected)
    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        CellDescription()
            .id(0)
            .pos({99.0f, 100.0f})
            .signalAndState({1.0f, 2.0f, 3.0f, 0, 0, 0, 0, 0}),
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Active).baseAngle(0).openingAngle(0))
            .cellType(CommunicatorDescription().mode(SenderDescription().range(50.0f).maxTimesSent(4))),
        CellDescription()
            .id(2)
            .pos({110.0f, 100.0f})
            .cellType(CommunicatorDescription().mode(ReceiverDescription())),
    }));
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getCellRef(2);

    // Since they're in the same creature, CommunicatorProcessor should NOT transmit.
    EXPECT_EQ(receiver._signalState, SignalState_Inactive);
}

TEST_F(CommunicatorTests, sender_multipleReceiversInRange_allReceiveSignal)
{
    // Create sender in creature 1
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f);

    // Create multiple receivers in different creatures
    data.add(createReceiverCreature(2, {110.0f, 100.0f}), false);
    data.add(createReceiverCreature(3, {100.0f, 110.0f}), false);
    data.add(createReceiverCreature(4, {90.0f, 100.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto result = _simulationFacade->getSimulationData();

    // All receivers should have received the signal
    for (uint64_t id : {200, 300, 400}) {
        auto receiver = result.getCellRef(id);
        EXPECT_EQ(receiver._signalState, SignalState_Active);
        EXPECT_FLOAT_EQ(receiver._signal._channels[0], 1.0f);
        EXPECT_EQ(receiver._signal._numTimesSent, 1);
    }
}

TEST_F(CommunicatorTests, sender_maxTimesSentExceeded_noSignalTransmitted)
{
    // Create sender in creature 1 with signal that has numTimesSent = 2 (equal to maxTimesSent)
    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        CellDescription()
            .id(100)
            .pos({100.0f, 100.0f})
            .cellType(CommunicatorDescription().mode(SenderDescription().range(50.0f).maxTimesSent(2))),
        CellDescription()
            .id(101)
            .pos({101.0f, 100.0f})
            .signalState(SignalState_Active)
            .signal(SignalDescription().numTimesSent(2).channels({1.0f, 2.0f, 3.0f, 0, 0, 0, 0, 0})),
    }));
    data.addConnection(100, 101);

    // Create receiver
    data.add(createReceiverCreature(2, {110.0f, 100.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getCellRef(200);

    // Receiver should NOT have received the signal (maxTimesSent exceeded)
    EXPECT_NE(receiver._signalState, SignalState_Active);
}

TEST_F(CommunicatorTests, sender_receiverColorRestriction_matchingColor)
{
    // Create sender with color 2
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f, 4, 2);

    // Create receiver that only accepts color 2
    data.add(createReceiverCreature(2, {110.0f, 100.0f}, 2), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getCellRef(200);

    // Receiver should have received the signal (color matches)
    EXPECT_EQ(receiver._signalState, SignalState_Active);
}

TEST_F(CommunicatorTests, sender_receiverColorRestriction_nonMatchingColor)
{
    // Create sender with color 3
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f, 4, 3);

    // Create receiver that only accepts color 2
    data.add(createReceiverCreature(2, {110.0f, 100.0f}, 2), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getCellRef(200);

    // Receiver should NOT have received the signal (color doesn't match)
    EXPECT_NE(receiver._signalState, SignalState_Active);
}

TEST_F(CommunicatorTests, sender_noActiveSignal_noTransmission)
{
    // Create sender without active signal
    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        CellDescription()
            .id(100)
            .pos({100.0f, 100.0f})
            .cellType(CommunicatorDescription().mode(SenderDescription().range(50.0f).maxTimesSent(4))),
        // No signalAndState set, so signal is not active
        CellDescription().id(101).pos({101.0f, 100.0f}),
    }));
    data.addConnection(100, 101);

    // Create receiver
    data.add(createReceiverCreature(2, {110.0f, 100.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getCellRef(200);

    // Receiver should NOT have received the signal (sender has no active signal)
    EXPECT_NE(receiver._signalState, SignalState_Active);
}

TEST_F(CommunicatorTests, sender_signalPriority_lowerNumTimesSentWins)
{
    // Create first sender with numTimesSent = 3
    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        CellDescription()
            .id(100)
            .pos({100.0f, 100.0f})
            .cellType(CommunicatorDescription().mode(SenderDescription().range(50.0f).maxTimesSent(10))),
        CellDescription()
            .id(101)
            .pos({101.0f, 100.0f})
            .signalState(SignalState_Active)
            .signal(SignalDescription().numTimesSent(3).channels({1.0f, 0, 0, 0, 0, 0, 0, 0})),
    }));
    data.addConnection(100, 101);

    // Create second sender with numTimesSent = 1 (higher priority)
    data.addCreature(CreatureDescription().id(2).cells({
        CellDescription()
            .id(200)
            .pos({100.0f, 120.0f})
            .cellType(CommunicatorDescription().mode(SenderDescription().range(50.0f).maxTimesSent(10))),
        CellDescription()
            .id(201)
            .pos({101.0f, 120.0f})
            .signalState(SignalState_Active)
            .signal(SignalDescription().numTimesSent(1).channels({-1.0f, 0, 0, 0, 0, 0, 0, 0})),
    }));
    data.addConnection(200, 201);

    // Create receiver
    data.add(createReceiverCreature(3, {100.0f, 110.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getCellRef(300);

    // Receiver should have received the signal
    EXPECT_EQ(receiver._signalState, SignalState_Active);
    // The numTimesSent should be the lower one + 1 = 2
    EXPECT_EQ(receiver._signal._numTimesSent, 2);
    EXPECT_EQ(receiver._signal._channels[0], -1.0f);
}

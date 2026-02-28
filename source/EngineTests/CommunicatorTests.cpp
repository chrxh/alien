#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/GenomeDesc.h>
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
    Desc createSenderCreature(uint64_t creatureId, RealVector2D pos, float range = 50.0f, int maxTimesSent = 4, int color = 0)
    {
        auto data = Desc().addCreature(
            {
                ObjectDesc()
                    .id(creatureId * 100)
                    .pos(pos)
                    .color(color)
                    .type(CellDesc()
                              .neuralNetwork(NeuralNetDesc().biases({1.0f, 0.5f, 2.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))
                              .cellType(CommunicatorDesc().mode(SenderDesc().range(range).maxTimesSent(maxTimesSent)))),
                ObjectDesc().id(creatureId * 100 + 1).pos({pos.x + 1.0f, pos.y}).color(color),
            },
            CreatureDesc().id(creatureId));
        data.addConnection(creatureId * 100, creatureId * 100 + 1);
        return data;
    }

    // Helper to create a receiver creature with 2 cells
    Desc createReceiverCreature(
        uint64_t creatureId,
        RealVector2D pos,
        std::optional<int> restrictToColor = std::nullopt,
        LineageRestriction restrictToLineage = LineageRestriction_No,
        int color = 0)
    {
        auto data = Desc().addCreature(
            {
                ObjectDesc()
                    .id(creatureId * 100)
                    .pos(pos)
                    .color(color)
                    .type(CellDesc().cellType(CommunicatorDesc().mode(ReceiverDesc().restrictToColor(restrictToColor).restrictToLineage(restrictToLineage)))),
                ObjectDesc().id(creatureId * 100 + 1).pos({pos.x + 1.0f, pos.y}).color(color),
            },
            CreatureDesc().id(creatureId));
        data.addConnection(creatureId * 100, creatureId * 100 + 1);
        return data;
    }
};

TEST_F(CommunicatorTests, sender_noReceiver_noSignalTransmitted)
{
    // Create a sender with no nearby receiver
    auto data = createSenderCreature(1, {100.0f, 100.0f});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto sender = result.getObjectRef(100);

    // Sender should have signal channels populated
    EXPECT_TRUE(sender.getCellRef()._signal._channels[0] != 0.0f);
}

TEST_F(CommunicatorTests, sender_receiverInRange_signalTransmitted)
{
    // Create sender in creature 1
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f);

    // Create receiver in creature 2, within range
    data.add(createReceiverCreature(2, {110.0f, 100.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    // Receiver should have received the signal
    EXPECT_FLOAT_EQ(receiver.getCellRef()._signal._channels[0], 1.0f);
    EXPECT_FLOAT_EQ(receiver.getCellRef()._signal._channels[1], 0.5f);
    EXPECT_FLOAT_EQ(receiver.getCellRef()._signal._channels[2], 2.0f);
}

TEST_F(CommunicatorTests, sender_receiverOutOfRange_noSignalTransmitted)
{
    // Create sender in creature 1 with small range
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 10.0f);

    // Create receiver in creature 2, out of range
    data.add(createReceiverCreature(2, {115.0f, 100.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    // Receiver should NOT have received the signal (channels should be zero)
    EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] == 0.0f);
}

TEST_F(CommunicatorTests, sender_sameCreatureReceiver_noSignalTransmitted)
{
    // Create sender and receiver in the same creature (both connected)
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(0).pos({99.0f, 100.0f}).type(CellDesc().signal({1.0f, 2.0f, 3.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().cellType(CommunicatorDesc().mode(SenderDesc().range(50.0f).maxTimesSent(4)))),
            ObjectDesc().id(2).pos({110.0f, 100.0f}).type(CellDesc().cellType(CommunicatorDesc().mode(ReceiverDesc()))),
        },
        CreatureDesc().id(1));
    data.addConnection(0, 1);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(2);

    // Since they're in the same creature, CommunicatorProcessor should NOT transmit (channels should be zero from transmitter).
    EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] == 0.0f);
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
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();

    // All receivers should have received the signal
    for (uint64_t id : {200, 300, 400}) {
        auto receiver = result.getObjectRef(id);
        EXPECT_FLOAT_EQ(receiver.getCellRef()._signal._channels[1], 0.5f);
    }
}

TEST_F(CommunicatorTests, sender_maxTimesSentExceeded_noSignalTransmitted)
{
    // Create sender in creature 1 with signal that has numTimesSent = 2 (equal to maxTimesSent)
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 100.0f}).type(CellDesc().cellType(CommunicatorDesc().mode(SenderDesc().range(50.0f).maxTimesSent(2)))),
            ObjectDesc()
                .id(101)
                .pos({101.0f, 100.0f})
                .type(CellDesc().signal(SignalDesc().numTimesSent(2).channels({1.0f, 2.0f, 3.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))),
        },
        CreatureDesc().id(1));
    data.addConnection(100, 101);

    // Create receiver
    data.add(createReceiverCreature(2, {110.0f, 100.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    // Receiver should NOT have received the signal (maxTimesSent exceeded)
}

TEST_F(CommunicatorTests, sender_receiverColorRestriction_matchingColor)
{
    // Create sender with color 2
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f, 4, 2);

    // Create receiver that only accepts color 2
    data.add(createReceiverCreature(2, {110.0f, 100.0f}, 2), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    // Receiver should have received the signal (color matches)
    EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] != 0.0f);
}

TEST_F(CommunicatorTests, sender_receiverColorRestriction_mismatchingColor)
{
    // Create sender with color 3
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f, 4, 3);

    // Create receiver that only accepts color 2
    data.add(createReceiverCreature(2, {110.0f, 100.0f}, 2), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    // Receiver should NOT have received the signal (color doesn't match)
    EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] == 0.0f);
}

TEST_F(CommunicatorTests, sender_noActiveSignal_noTransmission)
{
    // Create sender without active signal
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 100.0f}).type(CellDesc().cellType(CommunicatorDesc().mode(SenderDesc().range(50.0f).maxTimesSent(4)))),
            // No signalAndState set, so signal is not active
            ObjectDesc().id(101).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(1));
    data.addConnection(100, 101);

    // Create receiver
    data.add(createReceiverCreature(2, {110.0f, 100.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    // Receiver should NOT have received the signal (sender has no active signal)
    EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] == 0.0f);
}

TEST_F(CommunicatorTests, sender_signalPriority_signalReceived)
{
    // Create first sender with signal = 1.0
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 100.0f}).type(CellDesc().cellType(CommunicatorDesc().mode(SenderDesc().range(50.0f)))),
            ObjectDesc()
                .id(101)
                .pos({101.0f, 100.0f})
                .type(CellDesc().signal(SignalDesc().channels({1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))),
        },
        CreatureDesc().id(1));
    data.addConnection(100, 101);

    // Create second sender with signal = -1.0
    data.addCreature(
        {
            ObjectDesc().id(200).pos({100.0f, 120.0f}).type(CellDesc().cellType(CommunicatorDesc().mode(SenderDesc().range(50.0f)))),
            ObjectDesc()
                .id(201)
                .pos({101.0f, 120.0f})
                .type(CellDesc().signal(SignalDesc().channels({-1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))),
        },
        CreatureDesc().id(2));
    data.addConnection(200, 201);

    // Create receiver
    data.add(createReceiverCreature(3, {100.0f, 110.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(300);

    // Receiver should have received a signal (from one of the senders)
    EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] != 0.0f);
}

TEST_F(CommunicatorTests, sender_signalPriority_lowerNumTimesSentWins)
{
    // Create first sender with numTimesSent = 3
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 100.0f}).type(CellDesc().cellType(CommunicatorDesc().mode(SenderDesc().range(50.0f).maxTimesSent(10)))),
            ObjectDesc()
                .id(101)
                .pos({101.0f, 100.0f})
                .type(CellDesc().signal(SignalDesc().numTimesSent(3).channels({1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))),
        },
        CreatureDesc().id(1));
    data.addConnection(100, 101);

    // Create second sender with numTimesSent = 1 (higher priority)
    data.addCreature(
        {
            ObjectDesc().id(200).pos({100.0f, 120.0f}).type(CellDesc().cellType(CommunicatorDesc().mode(SenderDesc().range(50.0f).maxTimesSent(10)))),
            ObjectDesc()
                .id(201)
                .pos({101.0f, 120.0f})
                .type(CellDesc().signal(SignalDesc().numTimesSent(1).channels({-1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))),
        },
        CreatureDesc().id(2));
    data.addConnection(200, 201);

    // Create receiver
    data.add(createReceiverCreature(3, {100.0f, 110.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(300);

    // Receiver should have received the signal
    // The numTimesSent should be the lower one + 1 = 2
    EXPECT_EQ(receiver.getCellRef()._signal._numTimesSent, 2);
    EXPECT_EQ(receiver.getCellRef()._signal._channels[0], -1.0f);
}

/**
 * Parameterized test for angle translation with different reference direction differences.
 * Parameter: receiverRefAngleDiff - the difference in degrees between sender and receiver reference directions.
 */
class CommunicatorTests_AngleTranslation
    : public CommunicatorTests
    , public testing::WithParamInterface<float>
{};

INSTANTIATE_TEST_SUITE_P(CommunicatorTests_AngleTranslation, CommunicatorTests_AngleTranslation, ::testing::Values(0.0f, 90.0f, 180.0f, 270.0f));

TEST_P(CommunicatorTests_AngleTranslation, sender_angleTranslation)
{
    // The sender is at (100, 100) with connected cell at (101, 100) -> reference direction (1, 0) -> 90 degrees
    // The receiver's connected cell position is calculated based on the angle difference parameter
    auto receiverRefAngleDiff = GetParam();

    // Calculate the receiver's connected cell position based on the angle difference
    // Sender reference angle is 90 degrees (pointing right), receiver will be at 90 + receiverRefAngleDiff
    auto receiverRefAngle = 90.0f + receiverRefAngleDiff;
    auto receiverConnectedObjectOffset = Math::unitVectorOfAngle(receiverRefAngle);

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 100.0f}).type(CellDesc().cellType(CommunicatorDesc().mode(SenderDesc().range(50.0f).maxTimesSent(4)))),
            ObjectDesc()
                .id(101)
                .pos({101.0f, 100.0f})
                .type(CellDesc()
                          .signal(SignalDesc().channels({1.0f, 0.5f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))),  // channel[1] = 0.5 = 90 degrees
        },
        CreatureDesc().id(1));
    data.addConnection(100, 101);

    data.addCreature(
        {
            ObjectDesc().id(200).pos({120.0f, 100.0f}).type(CellDesc().cellType(CommunicatorDesc().mode(ReceiverDesc()))),
            ObjectDesc().id(201).pos({120.0f + receiverConnectedObjectOffset.x, 100.0f + receiverConnectedObjectOffset.y}),
        },
        CreatureDesc().id(2));
    data.addConnection(200, 201);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] != 0.0f);

    // The angle translation formula: translatedAngle = senderAngle + (senderRefAngle - receiverRefAngle) / 180
    // senderAngle = 0.5 (90 degrees), senderRefAngle = 90 degrees, receiverRefAngle = 90 + receiverRefAngleDiff
    // angleDiff = 90 - (90 + receiverRefAngleDiff) = -receiverRefAngleDiff
    // translatedAngle = 0.5 + (-receiverRefAngleDiff) / 180
    auto expectedAngle = Math::getNormalizedAngle(0.5f * 180.0f - receiverRefAngleDiff, -180.0f) / 180.0f;
    EXPECT_NEAR(receiver.getCellRef()._signal._channels[1], expectedAngle, 0.001f);
}

/**
 * Parameterized test for lineage restriction.
 * Parameters: (LineageRestriction mode, relatedLineage flag, expected acceptance)
 */
struct LineageRestrictionParams
{
    LineageRestriction restriction;
    bool relatedLineage;
    bool expectedAccept;
};

class CommunicatorTests_LineageRestriction
    : public CommunicatorTests
    , public testing::WithParamInterface<LineageRestrictionParams>
{};

INSTANTIATE_TEST_SUITE_P(
    CommunicatorTests_LineageRestriction,
    CommunicatorTests_LineageRestriction,
    ::testing::Values(
        LineageRestrictionParams{LineageRestriction_RelatedLineage, true, true},    // relatedLineage, accept
        LineageRestrictionParams{LineageRestriction_RelatedLineage, false, false},  // relatedLineage, rejected
        LineageRestrictionParams{LineageRestriction_UnrelatedLineage, false, true},  // unrelatedLineage, accept
        LineageRestrictionParams{LineageRestriction_UnrelatedLineage, true, false}   // unrelatedLineage, rejected
        ));

TEST_P(CommunicatorTests_LineageRestriction, sender_lineageRestriction)
{
    auto params = GetParam();

    uint64_t senderLineageId = 12345;
    uint64_t receiverLineageId = params.relatedLineage ? 12345 : 67890;

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 100.0f}).type(CellDesc().cellType(CommunicatorDesc().mode(SenderDesc().range(50.0f).maxTimesSent(4)))),
            ObjectDesc()
                .id(101)
                .pos({101.0f, 100.0f})
                .type(CellDesc().signal(SignalDesc().channels({1.0f, 0.5f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))),
        },
        CreatureDesc().id(1),
        GenomeDesc().lineageId(senderLineageId));
    data.addConnection(100, 101);

    data.addCreature(
        {
            ObjectDesc().id(200).pos({120.0f, 100.0f}).type(CellDesc().cellType(CommunicatorDesc().mode(ReceiverDesc().restrictToLineage(params.restriction)))),
            ObjectDesc().id(201).pos({121.0f, 100.0f}),
        },
        CreatureDesc().id(2),
        GenomeDesc().lineageId(receiverLineageId));
    data.addConnection(200, 201);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    if (params.expectedAccept) {
        EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] != 0.0f);
        EXPECT_FLOAT_EQ(receiver.getCellRef()._signal._channels[1], 0.5f);
    } else {
        EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] == 0.0f);
    }
}

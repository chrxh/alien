#include <gtest/gtest.h>

#include <optional>

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
    // Helper to create a sender creature with 2 cells (sender + helper for reference direction).
    // The sender is connected to a helper cell to the east, so its reference direction is (1, 0).
    // With frontAngle and the encoded angle in signal.channels[CommunicatorAngle] (= angleChannel), the
    // absolute facing direction defaults to south (0, +1): refAngle(90) + frontAngle(0) + angleChannel(0.5) * 180 = 180.
    Desc createSenderCreature(
        uint64_t creatureId,
        RealVector2D pos,
        int range = 50,
        int color = 0,
        std::optional<float> frontAngle = 0.0f,
        float angleChannel = 0.5f)
    {
        auto data = Desc().addCreature(
            {
                ObjectDesc()
                    .id(creatureId * 100)
                    .pos(pos)
                    .color(color)
                    .type(CellDesc()
                              .frontAngle(frontAngle)
                              .neuralNetwork(NeuralNetDesc().biases({1.0f, angleChannel, 2.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))
                              .cellType(CommunicatorDesc().mode(SenderDesc().range(range)))),
                ObjectDesc().id(creatureId * 100 + 1).pos({pos.x + 1.0f, pos.y}).color(color),
            },
            CreatureDesc().id(creatureId));
        data.addConnection(creatureId * 100, creatureId * 100 + 1);
        return data;
    }

    // Helper to create a receiver creature with 2 cells (receiver + helper for reference direction).
    Desc createReceiverCreature(
        uint64_t creatureId,
        RealVector2D pos,
        int restrictToColors = 0x3ff,
        LineageRestriction restrictToLineage = LineageRestriction_No,
        int color = 0,
        std::optional<float> frontAngle = 0.0f)
    {
        auto data = Desc().addCreature(
            {
                ObjectDesc()
                    .id(creatureId * 100)
                    .pos(pos)
                    .color(color)
                    .type(CellDesc()
                              .frontAngle(frontAngle)
                              .cellType(CommunicatorDesc().mode(ReceiverDesc().restrictToColors(restrictToColors).restrictToLineage(restrictToLineage)))),
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
    // Create sender in creature 1 (facing south)
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f);

    // Create receiver in creature 2, within range and in the opposite half-plane (north of the sender)
    data.add(createReceiverCreature(2, {100.0f, 90.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    // Receiver should have received the signal. Sender and receiver share the same absolute front direction,
    // so the encoded angle is transmitted unchanged.
    EXPECT_FLOAT_EQ(receiver.getCellRef()._signal._channels[0], 1.0f);
    EXPECT_NEAR(receiver.getCellRef()._signal._channels[Channels::CommunicatorAngle], 0.5f, 1e-4f);
    EXPECT_FLOAT_EQ(receiver.getCellRef()._signal._channels[2], 2.0f);
}

TEST_F(CommunicatorTests, sender_receiverOutOfRange_noSignalTransmitted)
{
    // Create sender in creature 1 with small range
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 10.0f);

    // Create receiver in creature 2, out of range (north but too far away)
    data.add(createReceiverCreature(2, {100.0f, 85.0f}), false);

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
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().frontAngle(0.0f).cellType(CommunicatorDesc().mode(SenderDesc().range(50)))),
            ObjectDesc().id(2).pos({100.0f, 90.0f}).type(CellDesc().frontAngle(0.0f).cellType(CommunicatorDesc().mode(ReceiverDesc()))),
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
    // Create sender in creature 1 (facing south)
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f);

    // Create multiple receivers in the opposite half-plane (north, north-east, north-west)
    data.add(createReceiverCreature(2, {110.0f, 90.0f}), false);
    data.add(createReceiverCreature(3, {100.0f, 90.0f}), false);
    data.add(createReceiverCreature(4, {90.0f, 90.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();

    // All receivers should have received the signal
    for (uint64_t id : {200, 300, 400}) {
        auto receiver = result.getObjectRef(id);
        EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] != 0.0f);
    }
}

TEST_F(CommunicatorTests, sender_receiverColorRestriction_matchingColor)
{
    // Create sender with color 2
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f, 2);

    // Create receiver that only accepts color 2
    data.add(createReceiverCreature(2, {100.0f, 90.0f}, 1 << 2), false);

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
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f, 3);

    // Create receiver that only accepts color 2
    data.add(createReceiverCreature(2, {100.0f, 90.0f}, 1 << 2), false);

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
            ObjectDesc().id(100).pos({100.0f, 100.0f}).type(CellDesc().frontAngle(0.0f).cellType(CommunicatorDesc().mode(SenderDesc().range(50)))),
            // No signal set, so signal is not active
            ObjectDesc().id(101).pos({101.0f, 100.0f}),
        },
        CreatureDesc().id(1));
    data.addConnection(100, 101);

    // Create receiver
    data.add(createReceiverCreature(2, {100.0f, 90.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    // Receiver should NOT have received the signal (sender has no active signal)
    EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] == 0.0f);
}

TEST_F(CommunicatorTests, sender_signalPriority_signalReceived)
{
    // Create two senders (both facing south) and a receiver between them; only the sender for which the receiver
    // lies in the opposite half-plane should transmit.
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 100.0f}).type(CellDesc().frontAngle(0.0f).cellType(CommunicatorDesc().mode(SenderDesc().range(50)))),
            ObjectDesc().id(101).pos({101.0f, 100.0f}).type(CellDesc().signal(SignalDesc().channels({1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))),
        },
        CreatureDesc().id(1));
    data.addConnection(100, 101);

    data.addCreature(
        {
            ObjectDesc().id(200).pos({100.0f, 120.0f}).type(CellDesc().frontAngle(0.0f).cellType(CommunicatorDesc().mode(SenderDesc().range(50)))),
            ObjectDesc().id(201).pos({101.0f, 120.0f}).type(CellDesc().signal(SignalDesc().channels({-1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))),
        },
        CreatureDesc().id(2));
    data.addConnection(200, 201);

    // Receiver between both senders; it lies north of sender 2 (opposite half-plane) but south of sender 1.
    data.add(createReceiverCreature(3, {100.0f, 110.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(300);

    // Receiver should have received the signal from sender 2
    EXPECT_FLOAT_EQ(receiver.getCellRef()._signal._channels[0], -1.0f);
}

/**
 * Directional gating: the sender only transmits to receivers that lie in the half-plane opposite to the encoded
 * facing direction, i.e. where dot(receiverDirection, facing) <= 0.
 */
TEST_F(CommunicatorTests, sender_receiverInFacingHalfPlane_noSignalTransmitted)
{
    // Sender faces south; the receiver is placed south of the sender (same half-plane as the facing direction)
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f);
    data.add(createReceiverCreature(2, {100.0f, 110.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    // Receiver in the facing half-plane should NOT receive the signal
    EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] == 0.0f);
}

TEST_F(CommunicatorTests, sender_frontAngleFlipsDirection)
{
    // Rotating the sender's front by 180 degrees flips the facing direction to north, so now a southern receiver
    // (in the opposite half-plane) receives, while a northern receiver is blocked.
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f, 0, 180.0f);
    data.add(createReceiverCreature(2, {100.0f, 110.0f}), false);  // south -> should receive
    data.add(createReceiverCreature(3, {100.0f, 90.0f}), false);   // north -> should be blocked

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();

    EXPECT_TRUE(result.getObjectRef(200).getCellRef()._signal._channels[0] != 0.0f);
    EXPECT_TRUE(result.getObjectRef(300).getCellRef()._signal._channels[0] == 0.0f);
}

TEST_F(CommunicatorTests, sender_encodedAngleChangesDirection)
{
    // With an encoded angle of 0, the facing direction equals the absolute front (east). A western receiver is then
    // in the opposite half-plane and receives, while an eastern receiver is blocked.
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f, 0, 0.0f, 0.0f);
    data.add(createReceiverCreature(2, {90.0f, 100.0f}), false);   // west -> should receive
    data.add(createReceiverCreature(3, {110.0f, 100.0f}), false);  // east -> should be blocked

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();

    EXPECT_TRUE(result.getObjectRef(200).getCellRef()._signal._channels[0] != 0.0f);
    EXPECT_TRUE(result.getObjectRef(300).getCellRef()._signal._channels[0] == 0.0f);
}

TEST_F(CommunicatorTests, sender_frontAngleNotInitialized_noSignalTransmitted)
{
    // Sender without an initialized front direction must not transmit
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f, 0, std::nullopt);
    data.add(createReceiverCreature(2, {100.0f, 90.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] == 0.0f);
}

TEST_F(CommunicatorTests, receiver_frontAngleNotInitialized_noSignalTransmitted)
{
    // Receiver without an initialized front direction must not receive
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f);
    data.add(createReceiverCreature(2, {100.0f, 90.0f}, 0x3ff, LineageRestriction_No, 0, std::nullopt), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] == 0.0f);
}

/**
 * Parameterized test for angle translation. The encoded angle is relative to the absolute front direction
 * (reference direction rotated by frontAngle). The receiver's stored angle must be adjusted by the difference
 * of the sender's and receiver's absolute front angles.
 */
class CommunicatorTests_AngleTranslation
    : public CommunicatorTests
    , public testing::WithParamInterface<float>
{};

INSTANTIATE_TEST_SUITE_P(CommunicatorTests_AngleTranslation, CommunicatorTests_AngleTranslation, ::testing::Values(0.0f, 90.0f, -90.0f, 180.0f));

TEST_P(CommunicatorTests_AngleTranslation, sender_angleTranslation)
{
    auto receiverFrontAngle = GetParam();
    auto senderFrontAngle = 30.0f;

    // Sender: reference direction east (refAngle 90), non-trivial frontAngle, encoded angle 0.5.
    // Absolute encoded direction = 90 + senderFrontAngle + 0.5 * 180 = 210 + senderFrontAngle, still pointing into the
    // southern half-plane, so the northern receiver passes the directional gate.
    auto data = createSenderCreature(1, {100.0f, 100.0f}, 50.0f, 0, senderFrontAngle);

    // Receiver: north of the sender (opposite half-plane), reference direction east, variable non-trivial frontAngle.
    data.add(createReceiverCreature(2, {100.0f, 90.0f}, 0x3ff, LineageRestriction_No, 0, receiverFrontAngle), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] != 0.0f);

    // Absolute encoded angle = senderRefAngle(90) + senderFrontAngle + encodedAngle(0.5) * 180.
    // Receiver angle = absolute angle - receiverRefAngle(90) - receiverFrontAngle, normalized to [-1, 1].
    auto absoluteAngle = 90.0f + senderFrontAngle + 0.5f * 180.0f;
    auto expectedAngle = Math::getNormalizedAngle(absoluteAngle - 90.0f - receiverFrontAngle, -180.0f) / 180.0f;
    EXPECT_NEAR(receiver.getCellRef()._signal._channels[Channels::CommunicatorAngle], expectedAngle, 0.001f);
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
        LineageRestrictionParams{LineageRestriction_RelatedLineage, true, true},     // relatedLineage, accept
        LineageRestrictionParams{LineageRestriction_RelatedLineage, false, false},   // relatedLineage, rejected
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
            ObjectDesc().id(100).pos({100.0f, 100.0f}).type(CellDesc().frontAngle(0.0f).cellType(CommunicatorDesc().mode(SenderDesc().range(50)))),
            ObjectDesc().id(101).pos({101.0f, 100.0f}).type(CellDesc().signal(SignalDesc().channels({1.0f, 0.5f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))),
        },
        CreatureDesc().id(1).lineageId(senderLineageId),
        GenomeDesc());
    data.addConnection(100, 101);

    // Receiver north of the sender (opposite half-plane)
    data.addCreature(
        {
            ObjectDesc()
                .id(200)
                .pos({100.0f, 80.0f})
                .type(CellDesc().frontAngle(0.0f).cellType(CommunicatorDesc().mode(ReceiverDesc().restrictToLineage(params.restriction)))),
            ObjectDesc().id(201).pos({101.0f, 80.0f}),
        },
        CreatureDesc().id(2).lineageId(receiverLineageId),
        GenomeDesc());
    data.addConnection(200, 201);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto result = _simulationFacade->getSimulationData();
    auto receiver = result.getObjectRef(200);

    if (params.expectedAccept) {
        EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] != 0.0f);
        EXPECT_NEAR(receiver.getCellRef()._signal._channels[Channels::CommunicatorAngle], 0.5f, 1e-4f);
    } else {
        EXPECT_TRUE(receiver.getCellRef()._signal._channels[0] == 0.0f);
    }
}

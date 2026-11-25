#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <ranges>

#include <Base/Math.h>

#include <Data/DescEditService.h>
#include <Data/Descs.h>
#include <Data/DistanceSignal.h>
#include <Data/GenomeDesc.h>

#include <Engine/Interface/SimulationFacade.h>

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
    using Message = std::array<float, 4>;

    static auto constexpr PreyColor = 5;
    static auto constexpr OtherPreyColor = 6;
    static auto constexpr AllColors = 0x3ff;
    static auto constexpr TestMessage = Message{0.25f, 0.5f, 0.75f, -0.5f};
    static auto constexpr ReceiverNetworkOutput = 0.125f;

    CellDesc createSenderCell(int range = 50, Message const& message = {}, std::optional<float> frontAngle = 0.0f)
    {
        return CellDesc()
            .frontAngle(frontAngle)
            .neuralNetwork(NeuralNetDesc().biases({1.0f, 0, 0, 0, message.at(0), message.at(1), message.at(2), message.at(3), 0, 0, 0, 0}))
            .cellType(CommunicatorDesc().mode(SenderDesc().range(range).oneway(false)));
    }

    CellDesc
    createReceiverCell(int restrictToColors = AllColors, LineageRestriction restrictToLineage = LineageRestriction_No, std::optional<float> frontAngle = 0.0f)
    {
        auto neuralNetwork = NeuralNetDesc().connectionWeight(0, 0.0f);
        for (auto channel : std::views::iota(0, STANDARD_NEURONS_PER_CELL)) {
            neuralNetwork.bias(channel, ReceiverNetworkOutput);
        }
        return CellDesc()
            .frontAngle(frontAngle)
            .neuralNetwork(neuralNetwork)
            .cellType(CommunicatorDesc().mode(ReceiverDesc().restrictToColors(restrictToColors).restrictToLineage(restrictToLineage)));
    }

    CellDesc createSensorCell(SensorModeDesc const& sensorMode)
    {
        return CellDesc().frontAngle(0.0f).cellType(SensorDesc().autoTrigger(true).mode(sensorMode));
    }

    // Chain of cells to the east, the reference direction of the first cell is (1, 0)
    ContentDesc createCreature(uint64_t creatureId, RealVector2D pos, std::vector<CellDesc> const& cells, int lineageId = 0)
    {
        auto numCells = toInt(cells.size());
        std::vector<ObjectDesc> objects;
        for (auto i : std::views::iota(0, numCells)) {
            objects.emplace_back(ObjectDesc().id(creatureId * 100 + i).pos({pos.x + toFloat(i), pos.y}).type(cells.at(i)));
        }
        auto data = ContentDesc().addCreature(objects, CreatureDesc().id(creatureId).lineageId(lineageId));
        for (auto i : std::views::iota(0, numCells - 1)) {
            data.addConnection(creatureId * 100 + i, creatureId * 100 + i + 1);
        }
        return data;
    }

    ContentDesc createSenderCreature(uint64_t creatureId, RealVector2D pos, int range = 50, Message const& message = TestMessage)
    {
        return createCreature(creatureId, pos, {createSenderCell(range, message), CellDesc()});
    }

    // The sensor is located at a distance of 2 to the east of the sender
    ContentDesc createObserverCreature(
        uint64_t creatureId,
        RealVector2D pos,
        SensorModeDesc const& sensorMode = DetectCreatureDesc().restrictToColors(1 << PreyColor),
        int range = 50)
    {
        return createCreature(creatureId, pos, {createSenderCell(range), CellDesc(), createSensorCell(sensorMode)});
    }

    // The receiver is located at a distance of 2 to the east of the sender
    ContentDesc createRelayCreature(uint64_t creatureId, RealVector2D pos, int range)
    {
        return createCreature(creatureId, pos, {createSenderCell(range), CellDesc(), createReceiverCell()});
    }

    ContentDesc createReceiverCreature(uint64_t creatureId, RealVector2D pos, int restrictToColors = AllColors)
    {
        return createCreature(creatureId, pos, {createReceiverCell(restrictToColors), CellDesc()});
    }

    ContentDesc createPrey(uint64_t creatureId, RealVector2D pos, int color = PreyColor)
    {
        return ContentDesc().addCreature({ObjectDesc().id(creatureId * 100).pos(pos).color(color)}, CreatureDesc().id(creatureId));
    }

    // Cells without a head cell die after about 200 time steps
    ContentDesc letLivePermanently(ContentDesc data)
    {
        data._objects.front().getCellRef()._headCell = true;
        return data;
    }

    // The creature still acts in the first cycle
    ContentDesc letDieInFirstCycle(ContentDesc data)
    {
        for (auto& object : data._objects) {
            object.getCellRef()._usableEnergy = 1.0f;
        }
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.cellDeathProbability.baseValue[i] = 1.0f;
        }
        _simulationFacade->setSimulationParameters(_parameters);
        return data;
    }

    void calcCycles(int numCycles) { _simulationFacade->calcTimesteps(numCycles * TIMESTEPS_PER_CELL_FUNCTION); }

    std::vector<float> getSignals(uint64_t objectId)
    {
        return _simulationFacade->getSimulationData().getObjectRef(objectId).getCellRef()._neuralActivity._signals;
    }

    void expectMessage(uint64_t receiverId, Message const& message)
    {
        auto signals = getSignals(receiverId);
        for (auto const& [signal, expectedSignal] : std::views::zip(signals | std::views::drop(Channels::CommunicatorMessage), message)) {
            EXPECT_FLOAT_EQ(signal, expectedSignal);
        }
    }

    void expectNoMessage(uint64_t receiverId)
    {
        for (auto const& signal : getSignals(receiverId) | std::views::drop(Channels::CommunicatorMessage)) {
            EXPECT_FLOAT_EQ(signal, ReceiverNetworkOutput);
        }
    }

    void expectTargetReceived(uint64_t receiverId, RealVector2D const& toTarget, int age = 0, float frontAngle = 0.0f)
    {
        auto signals = getSignals(receiverId);
        EXPECT_FLOAT_EQ(signals.at(Channels::CommunicatorTargetReceived), 1.0f);
        EXPECT_NEAR(signals.at(Channels::CommunicatorAngle), encodeAngleForDefaultReceiver(toTarget, frontAngle), 1e-2f);
        EXPECT_NEAR(signals.at(Channels::CommunicatorFreshness), 1.0f - toFloat(age) / Const::CommunicatorTargetMaxAge, 1e-4f);
        EXPECT_NEAR(signals.at(Channels::CommunicatorDistance), DistanceSignal::encode(Math::length(toTarget)), 1e-3f);
    }

    void expectNoTarget(uint64_t receiverId)
    {
        for (auto const& signal : getSignals(receiverId) | std::views::take(Channels::CommunicatorMessage)) {
            EXPECT_FLOAT_EQ(signal, ReceiverNetworkOutput);
        }
    }

    float encodeAngleForDefaultReceiver(RealVector2D const& direction, float frontAngle = 0.0f) const
    {
        auto referenceAngleEast = 90.0f;
        return Math::getNormalizedAngle(Math::angleOfVector(direction) - referenceAngleEast - frontAngle, -180.0f) / 180.0f;
    }
};

TEST_F(CommunicatorTests, receiver_noSender_neuralNetworkIsKept)
{
    _simulationFacade->setSimulationData(createReceiverCreature(2, {100.5f, 90.5f}));
    calcCycles(1);

    expectNoMessage(200);
    expectNoTarget(200);
}

TEST_F(CommunicatorTests, receiver_senderInRange_messageReceived)
{
    auto data = createSenderCreature(1, {100.5f, 100.5f});
    data.add(createReceiverCreature(2, {100.5f, 90.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    expectMessage(200, TestMessage);
    expectNoTarget(200);
}

TEST_F(CommunicatorTests, receiver_senderOutOfRange_noMessage)
{
    auto data = createSenderCreature(1, {100.5f, 100.5f}, 10);
    data.add(createReceiverCreature(2, {100.5f, 85.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    expectNoMessage(200);
}

TEST_F(CommunicatorTests, receiver_senderOfSameCreature_noMessage)
{
    auto data = createCreature(1, {100.5f, 100.5f}, {createReceiverCell(), CellDesc(), createSenderCell(50, TestMessage)});

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    expectNoMessage(100);
}

TEST_F(CommunicatorTests, receiver_senderNotTriggered_noMessage)
{
    auto data = createSenderCreature(1, {100.5f, 100.5f});
    data.getObjectRef(100).getCellRef()._neuralNetwork.bias(Channels::CellTypeActivation, 0.0f);
    data.add(createReceiverCreature(2, {100.5f, 90.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    expectNoMessage(200);
}

TEST_F(CommunicatorTests, receiver_multipleReceivers_allReceiveMessage)
{
    auto data = createSenderCreature(1, {100.5f, 100.5f});
    data.add(createReceiverCreature(2, {110.5f, 90.5f}), false);
    data.add(createReceiverCreature(3, {100.5f, 90.5f}), false);
    data.add(createReceiverCreature(4, {90.5f, 110.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    for (auto receiverId : {200, 300, 400}) {
        expectMessage(receiverId, TestMessage);
    }
}

TEST_F(CommunicatorTests, receiver_multipleSenders_messageOfNearestSender)
{
    auto data = createSenderCreature(1, {100.5f, 100.5f}, 50, {1.0f, 1.0f, 1.0f, 1.0f});
    data.add(createSenderCreature(2, {100.5f, 80.5f}, 50, TestMessage), false);
    data.add(createSenderCreature(3, {115.5f, 85.5f}, 50, {1.0f, 1.0f, 1.0f, 1.0f}), false);
    data.add(createReceiverCreature(4, {100.5f, 85.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    expectMessage(400, TestMessage);
}

TEST_F(CommunicatorTests, receiver_senderWithoutFrontAngle_messageReceived)
{
    auto data = createCreature(1, {100.5f, 100.5f}, {createSenderCell(50, TestMessage, std::nullopt), CellDesc()});
    data.add(createReceiverCreature(2, {100.5f, 90.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    expectMessage(200, TestMessage);
}

TEST_F(CommunicatorTests, receiver_noFrontAngle_noMessage)
{
    auto data = createSenderCreature(1, {100.5f, 100.5f});
    data.add(createCreature(2, {100.5f, 90.5f}, {createReceiverCell(AllColors, LineageRestriction_No, std::nullopt), CellDesc()}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    expectNoMessage(200);
}

struct LineageRestrictionParams
{
    LineageRestriction restriction;
    bool isRelated;
    bool isAccepted;
};

class CommunicatorTests_LineageRestriction
    : public CommunicatorTests
    , public testing::WithParamInterface<LineageRestrictionParams>
{};

INSTANTIATE_TEST_SUITE_P(
    CommunicatorTests_LineageRestriction,
    CommunicatorTests_LineageRestriction,
    ::testing::Values(
        LineageRestrictionParams{LineageRestriction_RelatedLineage, true, true},
        LineageRestrictionParams{LineageRestriction_RelatedLineage, false, false},
        LineageRestrictionParams{LineageRestriction_UnrelatedLineage, true, false},
        LineageRestrictionParams{LineageRestriction_UnrelatedLineage, false, true}));

TEST_P(CommunicatorTests_LineageRestriction, receiver_lineageRestriction)
{
    auto [restriction, isRelated, isAccepted] = GetParam();

    auto data = createCreature(1, {100.5f, 100.5f}, {createSenderCell(50, TestMessage), CellDesc()}, 12345);
    data.add(createCreature(2, {100.5f, 90.5f}, {createReceiverCell(AllColors, restriction), CellDesc()}, isRelated ? 12345 : 67890), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    if (isAccepted) {
        expectMessage(200, TestMessage);
    } else {
        expectNoMessage(200);
    }
}

class CommunicatorTests_FrontAngle
    : public CommunicatorTests
    , public testing::WithParamInterface<float>
{};

INSTANTIATE_TEST_SUITE_P(CommunicatorTests_FrontAngle, CommunicatorTests_FrontAngle, ::testing::Values(0.0f, 90.0f, -90.0f, 180.0f));

TEST_P(CommunicatorTests_FrontAngle, receiver_targetReceived)
{
    auto frontAngle = GetParam();

    auto data = createObserverCreature(1, {100.5f, 100.5f});
    data.add(createPrey(9, {102.5f, 106.5f}), false);
    data.add(createCreature(2, {90.5f, 90.5f}, {createReceiverCell(AllColors, LineageRestriction_No, frontAngle), CellDesc()}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    expectTargetReceived(200, {12.0f, 16.0f}, 0, frontAngle);
}

TEST_F(CommunicatorTests, receiver_targetAcrossWorldBorder)
{
    auto data = createObserverCreature(1, {990.5f, 100.5f});
    data.add(createPrey(9, {992.5f, 106.5f}), false);
    data.add(createReceiverCreature(2, {8.5f, 100.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    expectTargetReceived(200, {-16.0f, 6.0f});
}

TEST_F(CommunicatorTests, receiver_targetIsKeptAndOnlyAges)
{
    auto data = letDieInFirstCycle(createObserverCreature(1, {100.5f, 100.5f}));
    data.add(createPrey(9, {102.5f, 106.5f}), false);
    data.add(createReceiverCreature(2, {90.5f, 90.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);
    auto signals = getSignals(200);
    calcCycles(3);
    auto signalsAfterwards = getSignals(200);

    EXPECT_FLOAT_EQ(signals.at(Channels::CommunicatorTargetReceived), 1.0f);
    EXPECT_FLOAT_EQ(signalsAfterwards.at(Channels::CommunicatorTargetReceived), 0.0f);
    EXPECT_FLOAT_EQ(signalsAfterwards.at(Channels::CommunicatorAngle), signals.at(Channels::CommunicatorAngle));
    EXPECT_FLOAT_EQ(signalsAfterwards.at(Channels::CommunicatorDistance), signals.at(Channels::CommunicatorDistance));
    auto aging = 3.0f * TIMESTEPS_PER_CELL_FUNCTION / Const::CommunicatorTargetMaxAge;
    EXPECT_NEAR(signalsAfterwards.at(Channels::CommunicatorFreshness), signals.at(Channels::CommunicatorFreshness) - aging, 1e-4f);
    expectNoMessage(200);
}

TEST_F(CommunicatorTests, receiver_targetIsForgottenAfterMaxAge)
{
    auto data = letDieInFirstCycle(createObserverCreature(1, {100.5f, 100.5f}));
    data.add(createPrey(9, {102.5f, 106.5f}), false);
    data.add(letLivePermanently(createReceiverCreature(2, {90.5f, 90.5f})), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1 + Const::CommunicatorTargetMaxAge / TIMESTEPS_PER_CELL_FUNCTION);
    auto freshness = getSignals(200).at(Channels::CommunicatorFreshness);
    calcCycles(1);

    EXPECT_NEAR(freshness, 0.004f, 1e-4f);
    expectNoTarget(200);
}

TEST_F(CommunicatorTests, sender_targetIsRememberedWithoutSensorMatch)
{
    auto data = createObserverCreature(1, {100.5f, 100.5f});
    data.add(letDieInFirstCycle(createPrey(9, {102.5f, 106.5f})), false);
    data.add(createReceiverCreature(2, {90.5f, 90.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);
    _simulationFacade->testOnly_cleanupAfterDataManipulation();
    EXPECT_TRUE(_simulationFacade->testOnly_isDataValid());
    calcCycles(2);

    expectTargetReceived(200, {12.0f, 16.0f}, 2 * TIMESTEPS_PER_CELL_FUNCTION);
}

TEST_F(CommunicatorTests, sender_targetIsForgottenAfterMaxAge)
{
    auto data = letLivePermanently(createObserverCreature(1, {100.5f, 100.5f}));
    data.add(letDieInFirstCycle(createPrey(9, {102.5f, 106.5f})), false);
    data.add(letLivePermanently(createReceiverCreature(2, {90.5f, 90.5f})), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1 + Const::CommunicatorTargetMaxAge / TIMESTEPS_PER_CELL_FUNCTION);
    auto received = getSignals(200).at(Channels::CommunicatorTargetReceived);
    calcCycles(1);

    EXPECT_FLOAT_EQ(received, 1.0f);
    expectNoTarget(200);
}

TEST_F(CommunicatorTests, receiver_olderTargetsAreIgnored)
{
    auto data = createObserverCreature(1, {100.5f, 70.5f}, DetectCreatureDesc().restrictToColors(1 << OtherPreyColor));
    data.add(createPrey(2, {102.5f, 64.5f}, OtherPreyColor), false);
    data.add(createReceiverCreature(3, {100.5f, 85.5f}), false);
    data.add(createObserverCreature(4, {100.5f, 100.5f}), false);
    data.add(createObserverCreature(5, {80.5f, 85.5f}), false);
    data.add(createObserverCreature(6, {120.5f, 85.5f}), false);
    data.add(letDieInFirstCycle(createPrey(7, {102.5f, 106.5f})), false);
    data.add(letDieInFirstCycle(createPrey(8, {82.5f, 91.5f})), false);
    data.add(letDieInFirstCycle(createPrey(9, {122.5f, 91.5f})), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);
    for ([[maybe_unused]] auto cycle : std::views::iota(0, 5)) {
        calcCycles(1);
        expectTargetReceived(300, {2.0f, -21.0f});
    }
}

TEST_F(CommunicatorTests, receiver_targetOlderThanOwnSensorMatch)
{
    auto data = createObserverCreature(1, {100.5f, 100.5f});
    data.add(letDieInFirstCycle(createPrey(2, {102.5f, 106.5f})), false);
    auto ownSensor = createSensorCell(DetectCreatureDesc().restrictToColors(1 << OtherPreyColor));
    data.add(createCreature(3, {90.5f, 90.5f}, {createReceiverCell(), CellDesc(), ownSensor}), false);
    data.add(createPrey(4, {92.5f, 84.5f}, OtherPreyColor), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(3);

    EXPECT_FLOAT_EQ(getSignals(302).at(Channels::SensorFoundResult), 1.0f);
    expectTargetReceived(300, {12.0f, 16.0f}, 2 * TIMESTEPS_PER_CELL_FUNCTION);
}

TEST_F(CommunicatorTests, sender_targetIsRelayed)
{
    auto data = createObserverCreature(1, {100.5f, 100.5f}, DetectCreatureDesc().restrictToColors(1 << PreyColor), 14);
    data.add(createPrey(9, {102.5f, 106.5f}), false);
    data.add(createRelayCreature(2, {100.5f, 88.5f}, 15), false);
    data.add(createReceiverCreature(3, {100.5f, 74.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(2);

    expectTargetReceived(300, {2.0f, 32.0f}, TIMESTEPS_PER_CELL_FUNCTION);
}

TEST_F(CommunicatorTests, swarm_targetReachesAllCreatures)
{
    auto constexpr NumColumns = 8;
    auto constexpr NumRows = 10;

    auto data = createObserverCreature(1, {90.5f, 100.5f}, DetectCreatureDesc().restrictToColors(1 << PreyColor), 20);
    data.add(createPrey(2, {92.5f, 106.5f}), false);
    for (auto i : std::views::iota(0, NumColumns * NumRows)) {
        data.add(createRelayCreature(3 + i, {100.5f + toFloat(i % NumColumns) * 5.0f, 90.5f + toFloat(i / NumColumns) * 2.0f}, 20), false);
    }

    _simulationFacade->setSimulationData(data);
    calcCycles(5);

    for (auto i : std::views::iota(0, NumColumns * NumRows)) {
        auto signals = getSignals((3 + i) * 100 + 2);
        EXPECT_FLOAT_EQ(signals.at(Channels::CommunicatorTargetReceived), 1.0f);
        EXPECT_GT(signals.at(Channels::CommunicatorFreshness), 0.95f);
    }
}

struct SearchedColorsParams
{
    int searchedColors;
    int receiverColors;
    bool isAccepted;
};

class CommunicatorTests_SearchedColors
    : public CommunicatorTests
    , public testing::WithParamInterface<SearchedColorsParams>
{};

INSTANTIATE_TEST_SUITE_P(
    CommunicatorTests_SearchedColors,
    CommunicatorTests_SearchedColors,
    ::testing::Values(
        SearchedColorsParams{1 << 5, 1 << 5, true},
        SearchedColorsParams{1 << 5, 0x3ff, true},
        SearchedColorsParams{1 << 5, 1 << 6, false},
        SearchedColorsParams{1 << 5 | 1 << 6, 1 << 5 | 1 << 6, true},
        SearchedColorsParams{1 << 5 | 1 << 6, 1 << 5, false},
        SearchedColorsParams{0x3ff, 1 << 5, false}));

TEST_P(CommunicatorTests_SearchedColors, receiver_restrictToColors)
{
    auto [searchedColors, receiverColors, isAccepted] = GetParam();

    auto data = createObserverCreature(1, {100.5f, 100.5f}, DetectCreatureDesc().restrictToColors(searchedColors));
    data.add(createPrey(9, {102.5f, 106.5f}), false);
    data.add(createReceiverCreature(2, {90.5f, 90.5f}, receiverColors), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    if (isAccepted) {
        expectTargetReceived(200, {12.0f, 16.0f});
    } else {
        expectNoTarget(200);
    }
    expectMessage(200, {});
}

TEST_F(CommunicatorTests, receiver_restrictToColors_freeCells)
{
    auto data = createObserverCreature(1, {100.5f, 100.5f}, DetectFreeCellDesc().minDensity(0.05f).restrictToColors(1 << PreyColor));
    for (auto i : std::views::iota(0, 8)) {
        data._objects.emplace_back(ObjectDesc().id(1000 + i).pos({98.5f + toFloat(i), 80.5f}).color(PreyColor).type(FreeCellDesc()));
    }
    data.add(createReceiverCreature(2, {90.5f, 110.5f}, 1 << PreyColor), false);
    data.add(createReceiverCreature(3, {110.5f, 110.5f}, 1 << OtherPreyColor), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    EXPECT_FLOAT_EQ(getSignals(200).at(Channels::CommunicatorTargetReceived), 1.0f);
    expectNoTarget(300);
}

TEST_F(CommunicatorTests, receiver_restrictToColors_energyIsAccepted)
{
    auto data = createObserverCreature(1, {100.5f, 100.5f}, DetectEnergyDesc().minDensity(0.05f));
    for (auto i : std::views::iota(0, 8)) {
        data._energies.emplace_back(EnergyDesc().id(1000 + i).pos({98.5f + toFloat(i), 80.5f}).energy(10.0f));
    }
    data.add(createReceiverCreature(2, {90.5f, 110.5f}, 1 << PreyColor), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    EXPECT_FLOAT_EQ(getSignals(200).at(Channels::CommunicatorTargetReceived), 1.0f);
}

TEST_F(CommunicatorTests, sender_solidIsNoTarget)
{
    auto data = createObserverCreature(1, {100.5f, 100.5f}, DetectSolidDesc());
    data._objects.emplace_back(ObjectDesc().id(1000).pos({102.5f, 106.5f}).type(SolidDesc()));
    data.add(createReceiverCreature(2, {90.5f, 90.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    EXPECT_FLOAT_EQ(getSignals(102).at(Channels::SensorFoundResult), 1.0f);
    expectNoTarget(200);
}

class CommunicatorTests_OnewayAngle
    : public CommunicatorTests
    , public testing::WithParamInterface<std::pair<float, bool>>
{};

INSTANTIATE_TEST_SUITE_P(
    CommunicatorTests_OnewayAngle,
    CommunicatorTests_OnewayAngle,
    ::testing::Values(std::make_pair(95.0f, false), std::make_pair(105.0f, true), std::make_pair(-95.0f, false), std::make_pair(-105.0f, true)));

TEST_P(CommunicatorTests_OnewayAngle, sender_oneway_deviationFromTarget)
{
    auto [deviation, isAddressed] = GetParam();

    RealVector2D senderPos{100.5f, 100.5f};
    RealVector2D preyPos{102.5f, 106.5f};
    auto receiverPos = senderPos + Math::unitVectorOfAngle(Math::angleOfVector(preyPos - senderPos) + deviation) * 10.0f;
    auto data = createObserverCreature(1, senderPos);
    data.getObjectRef(100).getCellRef()._cellType = CommunicatorDesc().mode(SenderDesc().range(50).oneway(true));
    data.add(createPrey(9, preyPos), false);
    data.add(createReceiverCreature(2, receiverPos), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    if (isAddressed) {
        expectTargetReceived(200, preyPos - receiverPos);
    } else {
        expectNoTarget(200);
        expectNoMessage(200);
    }
}

TEST_F(CommunicatorTests, sender_oneway_noTarget_messageReceived)
{
    auto data = createSenderCreature(1, {100.5f, 100.5f});
    data.getObjectRef(100).getCellRef()._cellType = CommunicatorDesc().mode(SenderDesc().range(50).oneway(true));
    data.add(createReceiverCreature(2, {100.5f, 90.5f}), false);
    data.add(createReceiverCreature(3, {100.5f, 110.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    expectMessage(200, TestMessage);
    expectMessage(300, TestMessage);
}

class CommunicatorTests_LargeCreature : public CommunicatorTests
{
protected:
    static auto constexpr NumCells = 20;
    static auto constexpr Range = 12;

    CellDesc createSensorCellForColor(int color) { return createSensorCell(DetectCreatureDesc().restrictToColors(1 << color)); }
};

TEST_F(CommunicatorTests_LargeCreature, sender_targetOfNearbySensorIsPreferred)
{
    std::vector<CellDesc> cells(NumCells);
    cells.at(0) = createSensorCellForColor(PreyColor);
    cells.at(1) = createSenderCell(Range);
    cells.at(NumCells - 2) = createSenderCell(Range);
    cells.at(NumCells - 1) = createSensorCellForColor(OtherPreyColor);
    auto data = createCreature(1, {100.5f, 100.5f}, cells);
    data.add(createPrey(8, {100.5f, 106.5f}, PreyColor), false);
    data.add(createPrey(9, {119.5f, 106.5f}, OtherPreyColor), false);
    data.add(createReceiverCreature(2, {101.5f, 90.5f}), false);
    data.add(createReceiverCreature(3, {118.5f, 90.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    expectTargetReceived(200, {-1.0f, 16.0f});
    expectTargetReceived(300, {1.0f, 16.0f});
}

TEST_F(CommunicatorTests_LargeCreature, sender_noSensorNearby_targetOfDistantSensor)
{
    std::vector<CellDesc> cells(NumCells);
    cells.at(0) = createSensorCellForColor(PreyColor);
    cells.at(NumCells - 1) = createSenderCell(Range);
    auto data = createCreature(1, {100.5f, 100.5f}, cells);
    data.add(createPrey(9, {100.5f, 106.5f}), false);
    data.add(createReceiverCreature(2, {118.5f, 90.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(1);

    expectTargetReceived(200, {-18.0f, 16.0f});
}

// The nearby sensors lose their prey, only the distant sensor still has a match
TEST_F(CommunicatorTests_LargeCreature, sender_olderTargetOfNearbySensorsIsPreferred)
{
    std::vector<CellDesc> cells(NumCells);
    cells.at(0) = createSensorCellForColor(PreyColor);
    for (auto& cell : cells | std::views::drop(12)) {
        cell = createSensorCellForColor(OtherPreyColor);
    }
    cells.at(NumCells - 1) = createSenderCell(Range);
    auto data = createCreature(1, {100.5f, 100.5f}, cells);
    data.add(createPrey(8, {100.5f, 106.5f}, PreyColor), false);
    data.add(letDieInFirstCycle(createPrey(9, {115.5f, 106.5f}, OtherPreyColor)), false);
    data.add(createReceiverCreature(2, {118.5f, 90.5f}), false);

    _simulationFacade->setSimulationData(data);
    calcCycles(2);

    expectTargetReceived(200, {-3.0f, 16.0f}, TIMESTEPS_PER_CELL_FUNCTION);
}

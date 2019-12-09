#include "Base/ServiceLocator.h"
#include "IntegrationGpuTestFramework.h"
#include "ModelBasic/DescriptionFactory.h"
#include "ModelBasic/QuantityConverter.h"

class CommunicatorGpuTests : public IntegrationGpuTestFramework
{
public:
    CommunicatorGpuTests()
        : IntegrationGpuTestFramework()
    {}

    virtual ~CommunicatorGpuTests() = default;

protected:
    virtual void SetUp();

    struct TestParameters
    {
        MEMBER_DECLARATION(TestParameters, Enums::CommunicatorIn::Type, command1, Enums::CommunicatorIn::DO_NOTHING);
        MEMBER_DECLARATION(TestParameters, Enums::CommunicatorIn::Type, command2, Enums::CommunicatorIn::DO_NOTHING);
        MEMBER_DECLARATION(TestParameters, Enums::CommunicatorIn::Type, command3, Enums::CommunicatorIn::DO_NOTHING);
    };
    struct Expectations
    {
        MEMBER_DECLARATION(Expectations, int, sendMessages1, 0);
        MEMBER_DECLARATION(Expectations, int, sendMessages2, 0);
        MEMBER_DECLARATION(Expectations, int, sendMessages3, 0);
        MEMBER_DECLARATION(Expectations, Enums::CommunicatorOutReceivedNewMessage::Type, messageReceived1, Enums::CommunicatorOutReceivedNewMessage::NO);
        MEMBER_DECLARATION(Expectations, Enums::CommunicatorOutReceivedNewMessage::Type, messageReceived2, Enums::CommunicatorOutReceivedNewMessage::NO);
        MEMBER_DECLARATION(Expectations, Enums::CommunicatorOutReceivedNewMessage::Type, messageReceived3, Enums::CommunicatorOutReceivedNewMessage::NO);
    };
    void runStandardTest(TestParameters const& testParameters, Expectations const& expectations) const;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

void CommunicatorGpuTests::SetUp()
{
    _parameters.radiationProb = 0;  //exclude radiation
    _context->setSimulationParameters(_parameters);
}

void CommunicatorGpuTests::runStandardTest(TestParameters const& testParameters, Expectations const& expectations)
    const
{
    auto const createComCluster = [this](QVector2D const pos, Enums::CommunicatorIn::Type command) {
        auto cluster = createHorizontalCluster(2, pos, QVector2D{}, 0);
        auto& firstCell = cluster.cells->at(0);
        firstCell.tokenBranchNumber = 0;
        auto& secondCell = cluster.cells->at(1);
        secondCell.tokenBranchNumber = 1;
        secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::COMMUNICATOR);
        auto token = createSimpleToken();
        auto& tokenData = *token.data;
        tokenData[Enums::Communicator::IN] = command;
        firstCell.addToken(token);
        return cluster;
    };
    auto const comRange = _parameters.cellFunctionCommunicatorRange / 2;
    auto const origCluster1 = createComCluster({0, 0}, testParameters._command1);
    auto const origCluster2 = createComCluster({comRange, 0}, testParameters._command2);
    auto const origCluster3 = createComCluster({0, comRange}, testParameters._command3);

    auto const origSecondCell1 = origCluster1.cells->at(1);
    auto const origSecondCell2 = origCluster2.cells->at(1);
    auto const origSecondCell3 = origCluster3.cells->at(1);

    DataDescription origData;
    origData.addCluster(origCluster1);
    origData.addCluster(origCluster2);
    origData.addCluster(origCluster3);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    auto const data = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
    check(origData, data);

    auto const cellByCellId = IntegrationTestHelper::getCellByCellId(data);
    auto const& secondCell1 = cellByCellId.at(origSecondCell1.id);
    auto const& secondCell2 = cellByCellId.at(origSecondCell2.id);
    auto const& secondCell3 = cellByCellId.at(origSecondCell3.id);

    EXPECT_EQ(1, secondCell1.tokens->size());
    EXPECT_EQ(1, secondCell2.tokens->size());
    EXPECT_EQ(1, secondCell3.tokens->size());

    auto const& token1 = secondCell1.tokens->at(0);
    auto const& token2 = secondCell2.tokens->at(0);
    auto const& token3 = secondCell3.tokens->at(0);
    EXPECT_EQ(expectations._sendMessages1, token1.data->at(Enums::Communicator::OUT_SENT_NUM_MESSAGE));
    EXPECT_EQ(expectations._sendMessages2, token2.data->at(Enums::Communicator::OUT_SENT_NUM_MESSAGE));
    EXPECT_EQ(expectations._sendMessages3, token3.data->at(Enums::Communicator::OUT_SENT_NUM_MESSAGE));
    EXPECT_EQ(expectations._messageReceived1, token1.data->at(Enums::Communicator::OUT_RECEIVED_NEW_MESSAGE));
    EXPECT_EQ(expectations._messageReceived2, token2.data->at(Enums::Communicator::OUT_RECEIVED_NEW_MESSAGE));
    EXPECT_EQ(expectations._messageReceived3, token3.data->at(Enums::Communicator::OUT_RECEIVED_NEW_MESSAGE));
}

TEST_F(CommunicatorGpuTests, testDoNothing)
{
    runStandardTest(
        TestParameters()
            .command1(Enums::CommunicatorIn::DO_NOTHING)
            .command2(Enums::CommunicatorIn::DO_NOTHING)
            .command3(Enums::CommunicatorIn::DO_NOTHING),
        Expectations()
            .sendMessages1(0)
            .sendMessages2(0)
            .sendMessages3(0)
            .messageReceived1(Enums::CommunicatorOutReceivedNewMessage::NO)
            .messageReceived2(Enums::CommunicatorOutReceivedNewMessage::NO)
            .messageReceived3(Enums::CommunicatorOutReceivedNewMessage::NO));
}

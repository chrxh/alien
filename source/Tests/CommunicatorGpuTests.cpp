#include "Base/ServiceLocator.h"
#include "IntegrationGpuTestFramework.h"
#include "ModelBasic/DescriptionFactory.h"
#include "ModelBasic/QuantityConverter.h"

class CommunicatorGpuTests : public IntegrationGpuTestFramework
{
public:
    CommunicatorGpuTests()
        : IntegrationGpuTestFramework({200, 200}, getModelData())
    {
    }

    virtual ~CommunicatorGpuTests() = default;

protected:
    ModelGpuData getModelData()
    {
        ModelGpuData data;
        data.setNumThreadsPerBlock(64 * 2);
        data.setNumBlocks(64);
        data.setNumClusterPointerArrays(1);
        data.setMaxClusters(100);
        data.setMaxCells(500);
        data.setMaxParticles(500);
        data.setMaxTokens(50);
        data.setMaxCellPointers(500 * 10);
        data.setMaxClusterPointers(100 * 10);
        data.setMaxParticlePointers(500 * 10);
        data.setMaxTokenPointers(50 * 10);
        data.setDynamicMemorySize(1000000);
        return data;
    }

    virtual void SetUp();

    struct TestParameters
    {
        MEMBER_DECLARATION(TestParameters, Enums::CommunicatorIn::Type, command1, Enums::CommunicatorIn::DO_NOTHING);
        MEMBER_DECLARATION(TestParameters, Enums::CommunicatorIn::Type, command2, Enums::CommunicatorIn::DO_NOTHING);
        MEMBER_DECLARATION(TestParameters, Enums::CommunicatorIn::Type, command3, Enums::CommunicatorIn::DO_NOTHING);
        MEMBER_DECLARATION(TestParameters, int, cellIndexOfToken1, 0);
        MEMBER_DECLARATION(TestParameters, int, cellIndexOfToken2, 0);
        MEMBER_DECLARATION(TestParameters, int, cellIndexOfToken3, 0);
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
    _parameters.cellFunctionCommunicatorRange = 50;
    _context->setSimulationParameters(_parameters);
}

void CommunicatorGpuTests::runStandardTest(TestParameters const& testParameters, Expectations const& expectations)
    const
{
    auto const createComCluster = [this](QVector2D const pos, Enums::CommunicatorIn::Type command, int cellIndexOfToken) {
        auto cluster = createHorizontalCluster(4, pos, QVector2D{}, 0);
        for (int i = 0; i < 4; ++i) {
            cluster.cells->at(i).tokenBranchNumber = i;
        }
        cluster.cells->at(2).cellFeature = CellFeatureDescription().setType(Enums::CellFunction::COMMUNICATOR);
        auto token = createSimpleToken();
        auto& tokenData = *token.data;
        tokenData[Enums::Communicator::IN] = command;
        tokenData[Enums::Branching::TOKEN_BRANCH_NUMBER] = cellIndexOfToken;
        cluster.cells->at(cellIndexOfToken).addToken(token);
        return cluster;
    };
    auto const comRange = _parameters.cellFunctionCommunicatorRange / 2;
    auto const origCluster1 = createComCluster({0, 0}, testParameters._command1, testParameters._cellIndexOfToken1);
    auto const origCluster2 = createComCluster({comRange, 0}, testParameters._command2, testParameters._cellIndexOfToken2);
    auto const origCluster3 = createComCluster({0, comRange}, testParameters._command3, testParameters._cellIndexOfToken3);

    DataDescription origData;
    origData.addCluster(origCluster1);
    origData.addCluster(origCluster2);
    origData.addCluster(origCluster3);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(2, _controller);

    auto const data = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
    check(origData, data);

    auto const cellByCellId = IntegrationTestHelper::getCellByCellId(data);

    auto const& origTokenCell1 = origCluster1.cells->at(testParameters._cellIndexOfToken1 + 2);
    auto const& origTokenCell2 = origCluster2.cells->at(testParameters._cellIndexOfToken2 + 2);
    auto const& origTokenCell3 = origCluster3.cells->at(testParameters._cellIndexOfToken3 + 2);

    auto const& tokenCell1 = cellByCellId.at(origTokenCell1.id);
    auto const& tokenCell2 = cellByCellId.at(origTokenCell2.id);
    auto const& tokenCell3 = cellByCellId.at(origTokenCell3.id);

    EXPECT_EQ(1, tokenCell1.tokens->size());
    EXPECT_EQ(1, tokenCell2.tokens->size());
    EXPECT_EQ(1, tokenCell3.tokens->size());

    auto const& token1 = tokenCell1.tokens->at(0);
    auto const& token2 = tokenCell2.tokens->at(0);
    auto const& token3 = tokenCell3.tokens->at(0);
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

TEST_F(CommunicatorGpuTests, testSendAndReceive)
{
    runStandardTest(
        TestParameters()
            .command1(Enums::CommunicatorIn::SEND_MESSAGE).cellIndexOfToken1(1)
            .command2(Enums::CommunicatorIn::RECEIVE_MESSAGE).cellIndexOfToken2(0)
            .command3(Enums::CommunicatorIn::RECEIVE_MESSAGE).cellIndexOfToken3(0),
        Expectations()
            .sendMessages1(2)
            .messageReceived2(Enums::CommunicatorOutReceivedNewMessage::YES)
            .messageReceived3(Enums::CommunicatorOutReceivedNewMessage::YES));
}

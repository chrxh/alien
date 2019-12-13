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

    struct Communicator
    {
        MEMBER_DECLARATION(Communicator, QVector2D, pos, QVector2D());
        MEMBER_DECLARATION(Communicator, Enums::CommunicatorIn::Type, command, Enums::CommunicatorIn::DO_NOTHING);
        MEMBER_DECLARATION(Communicator, int, cellIndexWithToken, 0);
    };
    struct CommunicatorResult
    {
        MEMBER_DECLARATION(CommunicatorResult, int, messagesSent, 0);
        MEMBER_DECLARATION(
            CommunicatorResult,
            Enums::CommunicatorOutReceivedNewMessage::Type,
            messageReceived,
            Enums::CommunicatorOutReceivedNewMessage::NO);
    };

    struct TestParameters
    {
        MEMBER_DECLARATION(TestParameters, vector<Communicator>, communicators, vector<Communicator>());
    };
    struct Expectations
    {
        MEMBER_DECLARATION(Expectations, vector<CommunicatorResult>, communicatorResult, vector<CommunicatorResult>());
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
    DataDescription origData;
    for (auto const& communicator : testParameters._communicators) {
        auto const origCommunicator = createComCluster(communicator._pos, communicator._command, communicator._cellIndexWithToken);
        origData.addCluster(origCommunicator);
    }

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(2, _controller);

    auto const data = IntegrationTestHelper::getContent(_access, { {0, 0}, {_universeSize.x, _universeSize.y} });
    check(origData, data);

    auto const cellByCellId = IntegrationTestHelper::getCellByCellId(data);

    int clusterIndex = 0;
    auto comIter = testParameters._communicators.begin();
    auto expectIter = expectations._communicatorResult.begin();
    for (; comIter != testParameters._communicators.end(); ++comIter, ++expectIter, ++clusterIndex) {
        auto const& communicator = *comIter;
        auto const& expectation = *expectIter;

        auto const& origCluster = origData.clusters->at(clusterIndex);
        auto const& origTokenCell = origCluster.cells->at(communicator._cellIndexWithToken + 2);
        auto const& tokenCell = cellByCellId.at(origTokenCell.id);
        EXPECT_EQ(1, tokenCell.tokens->size());
        auto const& token = tokenCell.tokens->at(0);
        EXPECT_EQ(expectation._messagesSent, token.data->at(Enums::Communicator::OUT_SENT_NUM_MESSAGE));
        EXPECT_EQ(expectation._messageReceived, token.data->at(Enums::Communicator::OUT_RECEIVED_NEW_MESSAGE));
    }
}

TEST_F(CommunicatorGpuTests, testDoNothing)
{
    runStandardTest(
        TestParameters().communicators(
            {Communicator().pos({0, 0}).command(Enums::CommunicatorIn::DO_NOTHING),
             Communicator().pos({25, 0}).command(Enums::CommunicatorIn::DO_NOTHING)}),
        Expectations().communicatorResult(
            {CommunicatorResult().messagesSent(0).messageReceived(Enums::CommunicatorOutReceivedNewMessage::NO),
             CommunicatorResult().messagesSent(0).messageReceived(Enums::CommunicatorOutReceivedNewMessage::NO)}));
}

TEST_F(CommunicatorGpuTests, testSendAndReceive)
{
    runStandardTest(
        TestParameters().communicators(
            {Communicator().pos({0, 0}).command(Enums::CommunicatorIn::SEND_MESSAGE).cellIndexWithToken(1),
             Communicator().pos({25, 0}).command(Enums::CommunicatorIn::RECEIVE_MESSAGE).cellIndexWithToken(0),
             Communicator().pos({-25, -25}).command(Enums::CommunicatorIn::RECEIVE_MESSAGE).cellIndexWithToken(0)}),
        Expectations().communicatorResult(
            {CommunicatorResult().messagesSent(2).messageReceived(Enums::CommunicatorOutReceivedNewMessage::NO),
             CommunicatorResult().messagesSent(0).messageReceived(Enums::CommunicatorOutReceivedNewMessage::YES),
             CommunicatorResult().messagesSent(0).messageReceived(Enums::CommunicatorOutReceivedNewMessage::YES)}));
}

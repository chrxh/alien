#include "Base/ServiceLocator.h"
#include "ModelBasic/QuantityConverter.h"

#include "IntegrationGpuTestFramework.h"

class ConstructorGpuTests
    : public IntegrationGpuTestFramework
{
public:
    ConstructorGpuTests() : IntegrationGpuTestFramework()
    {}

    virtual ~ConstructorGpuTests() = default;

protected:
    virtual void SetUp();

    struct TestResult
    {
        TokenDescription token;
        CellDescription constructorCell;
        optional<CellDescription> constructedCell;
    };
    TestResult runStandardConstructionTest(TokenDescription const& token) const;
    TestResult runConstructionOnWedgeClusterTest(TokenDescription const& token, float wedgeAngle, float clusterAngle) const;
    TestResult runConstructionOnTriangleClusterTest(TokenDescription const & token) const;

    TokenDescription createTokenForSimpleConstruction(
        Enums::ConstrIn::Type constructionIn,
        Enums::ConstrInOption::Type option,
        float angle,
        float distance) const;

    struct Expectations
    {
        Enums::ConstrOut::Type tokenOutput;
        optional<QVector2D> constructedCellPosRelativeToConstructorCell;
    };
    void checkResult(TestResult const& testResult, Expectations const& expectations) const;

protected:
    float _offspringDistance = 0.0f;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

void ConstructorGpuTests::SetUp()
{
    _offspringDistance = _parameters.cellFunctionConstructorOffspringCellDistance;
    _parameters.radiationProb = 0;    //exclude radiation
    _context->setSimulationParameters(_parameters);
}

auto ConstructorGpuTests::runStandardConstructionTest(TokenDescription const & token) const -> TestResult
{
    DataDescription origData;
    auto cluster = createHorizontalCluster(2, QVector2D{10, 10}, QVector2D{}, 0);

    auto& firstCell = cluster.cells->at(0);
    firstCell.tokenBranchNumber = 0;
    firstCell.addToken(token);

    auto& secondCell = cluster.cells->at(1);
    secondCell.tokenBranchNumber = 1;
    secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);

    //perform test
    IntegrationTestHelper::runSimulation(1, _controller);

    //check results
    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newSecondCell = newCellByCellId.at(secondCell.id);
    auto const& newToken = newSecondCell.tokens->at(0);
    auto const& newCluster = newData.clusters->at(0);
    EXPECT_TRUE(isCompatible(cluster.pos, newCluster.pos));

    TestResult result;
    result.token = newToken;
    result.constructorCell = newSecondCell;

    newCellByCellId.erase(firstCell.id);
    newCellByCellId.erase(secondCell.id);
    if (!newCellByCellId.empty()) {
        result.constructedCell = newCellByCellId.begin()->second;
    }

    checkEnergy(origData, newData);

    return result;
}

auto ConstructorGpuTests::runConstructionOnWedgeClusterTest(
    TokenDescription const& token,
    float wedgeAngle,
    float clusterAngle) const -> TestResult
{
    ClusterDescription cluster;
    cluster.setId(_numberGen->getId()).setVel(QVector2D{}).setAngle(0).setAngularVel(0);

    auto const center = QVector2D{10, 10};
    auto const cellEnergy = _parameters.cellFunctionConstructorOffspringCellEnergy;
    auto const relPos1 = Physics::unitVectorOfAngle(clusterAngle + 270 + wedgeAngle / 2) / 2;
    auto const relPos2 = QVector2D{0, 0};
    auto const relPos3 = Physics::unitVectorOfAngle(clusterAngle + 270 - wedgeAngle / 2) / 2;
    auto const cellId1 = _numberGen->getId();
    auto const cellId2 = _numberGen->getId();
    auto const cellId3 = _numberGen->getId();
    cluster.addCells({CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(center + relPos1)
                          .setMaxConnections(1)
                          .setConnectingCells({cellId2})
                          .setTokenBranchNumber(0)
                          .setId(cellId1)
                          .setCellFeature(CellFeatureDescription())
                          .addToken(token),
                      CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(center + relPos2)
                          .setMaxConnections(2)
                          .setConnectingCells({cellId1, cellId3})
                          .setTokenBranchNumber(1)
                          .setId(cellId2)
                          .setCellFeature(CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR)),
                      CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(center + relPos3)
                          .setMaxConnections(1)
                          .setConnectingCells({cellId2})
                          .setTokenBranchNumber(2)
                          .setId(cellId3)
                          .setCellFeature(CellFeatureDescription())});

    cluster.setPos(cluster.getClusterPosFromCells());

    DataDescription origData;
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);

    //perform test
    IntegrationTestHelper::runSimulation(1, _controller);

    //check results
    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
    auto const& newCluster = newData.clusters->at(0);
    EXPECT_TRUE(isCompatible(cluster.pos, newCluster.pos));

    auto newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newCell2 = newCellByCellId.at(cellId2);
    auto const& newToken = newCell2.tokens->at(0);

    TestResult result;
    result.token = newToken;
    result.constructorCell = newCell2;

    newCellByCellId.erase(cellId1);
    newCellByCellId.erase(cellId2);
    newCellByCellId.erase(cellId3);
    if (!newCellByCellId.empty()) {
        result.constructedCell = newCellByCellId.begin()->second;
    }

    checkEnergy(origData, newData);

    return result;
}

auto ConstructorGpuTests::runConstructionOnTriangleClusterTest(TokenDescription const & token) const -> TestResult
{
    ClusterDescription cluster;
    cluster.setId(_numberGen->getId()).setVel(QVector2D{}).setAngle(0).setAngularVel(0);

    auto const center = QVector2D{ 10, 10 };
    auto const cellEnergy = _parameters.cellFunctionConstructorOffspringCellEnergy;
    auto const relPos1 = QVector2D{ 0, -1 };
    auto const relPos2 = QVector2D{ -1, 0 };
    auto const relPos3 = QVector2D{ 0, 1 };
    auto const relPos4 = QVector2D{ 0, 0 };
    auto const cellId1 = _numberGen->getId();
    auto const cellId2 = _numberGen->getId();
    auto const cellId3 = _numberGen->getId();
    auto const cellId4 = _numberGen->getId();
    cluster.addCells({CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(center + relPos1)
                          .setMaxConnections(1)
                          .setConnectingCells({cellId4})
                          .setTokenBranchNumber(0)
                          .setId(cellId1)
                          .setCellFeature(CellFeatureDescription())
                          .addToken(token),
                      CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(center + relPos2)
                          .setMaxConnections(1)
                          .setConnectingCells({cellId4})
                          .setTokenBranchNumber(0)
                          .setId(cellId2)
                          .setCellFeature(CellFeatureDescription()),
                      CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(center + relPos3)
                          .setMaxConnections(1)
                          .setConnectingCells({cellId4})
                          .setTokenBranchNumber(0)
                          .setId(cellId3)
                          .setCellFeature(CellFeatureDescription()),
                      CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(center + relPos4)
                          .setMaxConnections(3)
                          .setConnectingCells({cellId1, cellId2, cellId3})
                          .setTokenBranchNumber(1)
                          .setId(cellId4)
                          .setCellFeature(CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR))});

    cluster.setPos(cluster.getClusterPosFromCells());

    DataDescription origData;
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);

    //perform test
    IntegrationTestHelper::runSimulation(1, _controller);

    //check results
    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& newCluster = newData.clusters->at(0);
    EXPECT_TRUE(isCompatible(cluster.pos, newCluster.pos));

    auto newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newCell4 = newCellByCellId.at(cellId4);
    auto const& newToken = newCell4.tokens->at(0);

    TestResult result;
    result.token = newToken;
    result.constructorCell = newCell4;

    newCellByCellId.erase(cellId1);
    newCellByCellId.erase(cellId2);
    newCellByCellId.erase(cellId3);
    newCellByCellId.erase(cellId4);
    if (!newCellByCellId.empty()) {
        result.constructedCell = newCellByCellId.begin()->second;
    }

    checkEnergy(origData, newData);

    return result;
}

TokenDescription ConstructorGpuTests::createTokenForSimpleConstruction(
    Enums::ConstrIn::Type constructionIn,
    Enums::ConstrInOption::Type option,
    float angle,
    float distance) const
{
    auto token = createSimpleToken();
    (*token.data)[Enums::Constr::IN] = constructionIn;
    (*token.data)[Enums::Constr::IN_OPTION] = option;
    (*token.data)[Enums::Constr::INOUT_ANGLE] = QuantityConverter::convertAngleToData(angle);
    (*token.data)[Enums::Constr::IN_DIST] = QuantityConverter::convertDistanceToData(distance);
    (*token.data)[Enums::Constr::IN_CELL_MAX_CONNECTIONS] = 2;
    token.energy = 2*_parameters.tokenMinEnergy + 2*_parameters.cellFunctionConstructorOffspringCellEnergy;
    return token;
}

void ConstructorGpuTests::checkResult(TestResult const& testResult, Expectations const& expectations) const
{
    auto const& token = testResult.token;
    EXPECT_EQ(expectations.tokenOutput, token.data->at(Enums::Constr::OUT));

    if (Enums::ConstrIn::DO_NOTHING == token.data->at(Enums::Constr::IN)) {
        EXPECT_FALSE(testResult.constructedCell);
        return;
    }

    if (Enums::ConstrOut::SUCCESS == expectations.tokenOutput) {
        EXPECT_TRUE(testResult.constructedCell);
        EXPECT_TRUE(isCompatible(
            _parameters.cellFunctionConstructorOffspringCellEnergy,
            static_cast<float>(*testResult.constructedCell->energy)));
        EXPECT_EQ(token.data->at(Enums::Constr::IN_CELL_FUNCTION), testResult.constructedCell->cellFeature->type);
        {
            auto const& connectingCells = *testResult.constructedCell->connectingCells;
            EXPECT_TRUE(
                std::find(connectingCells.begin(), connectingCells.end(), testResult.constructorCell.id)
                != connectingCells.end());
        }
        {
            auto const& connectingCells = *testResult.constructorCell.connectingCells;
            EXPECT_TRUE(
                std::find(connectingCells.begin(), connectingCells.end(), testResult.constructedCell->id)
                != connectingCells.end());
        }
        EXPECT_PRED3(
            predEqual, 0,
            (*testResult.constructorCell.pos + *expectations.constructedCellPosRelativeToConstructorCell
             - *testResult.constructedCell->pos)
                .length(),
            0.05);

        return;
    }

    EXPECT_FALSE(testResult.constructedCell);
}

TEST_F(ConstructorGpuTests, testConstructSimpleCell_standardParameters)
{
    auto const token =
        createTokenForSimpleConstruction(Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 0.0f, 0.0f);
    auto result = runStandardConstructionTest(token);
    auto const expectedCellPos = QVector2D{ _offspringDistance, 0 };
    checkResult(result, { Enums::ConstrOut::SUCCESS, expectedCellPos });
}

TEST_F(ConstructorGpuTests, testConstructSimpleCell_ignoreDistanceOnFirstConstructedCell1)
{
    auto const token = createTokenForSimpleConstruction(
        Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 0.0f, _offspringDistance / 2);
    auto const result = runStandardConstructionTest(token);

    auto const expectedCellPos = QVector2D{ _offspringDistance, 0 };
    checkResult(result, { Enums::ConstrOut::SUCCESS, expectedCellPos });
}

TEST_F(ConstructorGpuTests, testConstructSimpleCell_ignoreDistanceOnFirstConstructedCell2)
{
    auto const token = createTokenForSimpleConstruction(
        Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 0.0f, _offspringDistance * 2);
    auto const result = runStandardConstructionTest(token);

    auto const expectedCellPos = QVector2D{ _offspringDistance, 0 };
    checkResult(result, { Enums::ConstrOut::SUCCESS, expectedCellPos });
}

TEST_F(ConstructorGpuTests, testConstructSimpleCell_rightHandSide)
{
    auto const token = createTokenForSimpleConstruction(
        Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 90.0f, 0.0f);
    auto const result = runStandardConstructionTest(token);

    auto const expectedCellPos = QVector2D{ 0, _offspringDistance };
    checkResult(result, { Enums::ConstrOut::SUCCESS, expectedCellPos });
}

TEST_F(ConstructorGpuTests, testConstructSimpleCell_leftHandSide)
{
    auto const token = createTokenForSimpleConstruction(
        Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, -90.0f, 0.0f);
    auto const result = runStandardConstructionTest(token);

    auto const expectedCellPos = QVector2D{ 0, -_offspringDistance };
    checkResult(result, { Enums::ConstrOut::SUCCESS, expectedCellPos });
}

TEST_F(ConstructorGpuTests, testConstructSimpleCellOnWedgeCluster_rightHandSide)
{
    auto const token =
        createTokenForSimpleConstruction(Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 0.0f, 0.0f);
    auto const result = runConstructionOnWedgeClusterTest(token, 90, 0);

    auto const expectedCellPos = QVector2D{ _offspringDistance, 0 };
    checkResult(result, { Enums::ConstrOut::SUCCESS, expectedCellPos });
}

TEST_F(ConstructorGpuTests, testConstructSimpleCellOnWedgeCluster_leftHandSide)
{
    auto const token =
        createTokenForSimpleConstruction(Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 0.0f, 0.0f);
    auto const result = runConstructionOnWedgeClusterTest(token, 270, 0);

    auto const expectedCellPos = QVector2D{ -_offspringDistance, 0 };
    checkResult(result, { Enums::ConstrOut::SUCCESS, expectedCellPos });
}

TEST_F(ConstructorGpuTests, testConstructSimpleCellOnWedgeCluster_diagonal)
{
    auto const token =
        createTokenForSimpleConstruction(Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 0.0f, 0.0f);
    auto const result = runConstructionOnWedgeClusterTest(token, 90, 45);

    auto const expectedCellPos = QVector2D{ _offspringDistance / sqrtf(2), _offspringDistance / sqrtf(2) };
    checkResult(result, { Enums::ConstrOut::SUCCESS, expectedCellPos });
}

TEST_F(ConstructorGpuTests, testConstructSimpleCellOnTiangleCluster)
{
    auto const token =
        createTokenForSimpleConstruction(Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 0.0f, 0.0f);
    auto const result = runConstructionOnTriangleClusterTest(token);

    auto const expectedCellPos = QVector2D{ _offspringDistance, 0 };
    checkResult(result, { Enums::ConstrOut::SUCCESS, expectedCellPos });
}

TEST_F(ConstructorGpuTests, testConstructSimpleCellOnTiangleCluster_errorMaxConnectionsReached)
{
    _parameters.cellMaxBonds = 3;
    _context->setSimulationParameters(_parameters);

    auto const token =
        createTokenForSimpleConstruction(Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 0.0f, 0.0f);
    auto const result = runConstructionOnTriangleClusterTest(token);

    auto const expectedCellPos = QVector2D{ _offspringDistance, 0 };
    checkResult(result, { Enums::ConstrOut::ERROR_CONNECTION, boost::none});
}

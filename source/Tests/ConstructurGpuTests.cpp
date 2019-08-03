#include <boost/range/adaptors.hpp>

#include "Base/ServiceLocator.h"
#include "IntegrationGpuTestFramework.h"
#include "ModelBasic/QuantityConverter.h"

class ConstructorGpuTests : public IntegrationGpuTestFramework
{
public:
    ConstructorGpuTests()
        : IntegrationGpuTestFramework()
    {}

    virtual ~ConstructorGpuTests() = default;

protected:
    virtual void SetUp();

    struct TestResult
    {
        QVector2D movementOfCenter;
        int increaseNumberOfCells;
        TokenDescription token;
        CellDescription constructorCell;
        optional<CellDescription> constructedCell;
    };
    struct ConstructionOnLineClusterTestParameters
    {
        MEMBER_DECLARATION(ConstructionOnLineClusterTestParameters, bool, obstacle, false);
        MEMBER_DECLARATION(ConstructionOnLineClusterTestParameters, TokenDescription, token, TokenDescription());
    };
    TestResult runConstructionOnLineClusterTest(ConstructionOnLineClusterTestParameters const& parameters);
    TestResult runConstructionOnWedgeClusterTest(TokenDescription const& token, float wedgeAngle, float clusterAngle)
        const;
    TestResult runConstructionOnTriangleClusterTest(TokenDescription const& token) const;

    struct TokenForConstructionParameters
    {
        MEMBER_DECLARATION(TokenForConstructionParameters, optional<float>, energy, boost::none);
        MEMBER_DECLARATION(
            TokenForConstructionParameters,
            Enums::ConstrIn::Type,
            constructionInput,
            Enums::ConstrIn::DO_NOTHING);
        MEMBER_DECLARATION(
            TokenForConstructionParameters,
            Enums::ConstrInOption::Type,
            constructionOption,
            Enums::ConstrInOption::STANDARD);
        MEMBER_DECLARATION(TokenForConstructionParameters, float, angle, 0.0f);
        MEMBER_DECLARATION(TokenForConstructionParameters, float, distance, 0.0f);
    };
    TokenDescription createTokenForConstruction(TokenForConstructionParameters tokenParameters) const;

    struct Expectations
    {
        MEMBER_DECLARATION(Expectations, Enums::ConstrOut::Type, tokenOutput, Enums::ConstrOut::SUCCESS);
        MEMBER_DECLARATION(Expectations, optional<QVector2D>, constructedRelCellPos, boost::none);
        MEMBER_DECLARATION(Expectations, bool, destruction, false);
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
    _parameters.radiationProb = 0;  //exclude radiation
    _context->setSimulationParameters(_parameters);
}

auto ConstructorGpuTests::runConstructionOnLineClusterTest(ConstructionOnLineClusterTestParameters const& parameters)
    -> TestResult
{
    DataDescription origData;
    auto cluster = createHorizontalCluster(2, QVector2D{10.5, 10.5}, QVector2D{}, 0);

    auto& firstCell = cluster.cells->at(0);
    firstCell.tokenBranchNumber = 0;
    firstCell.addToken(parameters._token);

    auto& secondCell = cluster.cells->at(1);
    secondCell.tokenBranchNumber = 1;
    secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);

    origData.addCluster(cluster);

    if (parameters._obstacle) {
        auto obstacle =
            createHorizontalCluster(4, QVector2D{13.0f + _parameters.cellMinDistance / 2, 10.5}, QVector2D{}, 0);
        origData.addCluster(obstacle);
    }

    IntegrationTestHelper::updateData(_access, origData);

    //perform test
    IntegrationTestHelper::runSimulation(1, _controller);

    //check results
    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    checkEnergy(origData, newData);

    auto newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto newClusterByCellId = IntegrationTestHelper::getClusterByCellId(newData);
    auto newCluster = newClusterByCellId.at(firstCell.id);

    TestResult result;
    result.movementOfCenter = *newCluster.pos - *cluster.pos;
    result.increaseNumberOfCells = newCluster.cells->size() - cluster.cells->size();

    auto const& newSecondCell = newCellByCellId.at(secondCell.id);
    auto const& newToken = newSecondCell.tokens->at(0);

    result.token = newToken;
    result.constructorCell = newSecondCell;

    std::list<CellDescription> remainingCells;
    for (auto const& newCell : *newCluster.cells) {
        if (newCell.id != firstCell.id && newCell.id != secondCell.id) {
            remainingCells.push_back(newCell);
        }
    }
    EXPECT_GE(1, remainingCells.size());

    if (!remainingCells.empty()) {
        result.constructedCell = *remainingCells.begin();
    }

    return result;
}

auto ConstructorGpuTests::runConstructionOnWedgeClusterTest(
    TokenDescription const& token,
    float wedgeAngle,
    float clusterAngle) const -> TestResult
{
    ClusterDescription cluster;
    cluster.setId(_numberGen->getId()).setVel(QVector2D{}).setAngle(0).setAngularVel(0);

    auto const center = QVector2D{10.5f, 10.5f};
    auto const cellEnergy = _parameters.cellFunctionConstructorOffspringCellEnergy;
    auto const relPos1 = Physics::unitVectorOfAngle(clusterAngle + 270 + wedgeAngle / 2);
    auto const relPos2 = QVector2D{0, 0};
    auto const relPos3 = Physics::unitVectorOfAngle(clusterAngle + 270 - wedgeAngle / 2);
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
    checkEnergy(origData, newData);

    auto const& newCluster = newData.clusters->at(0);

    TestResult result;
    result.movementOfCenter = *newCluster.pos - *cluster.pos;
    result.increaseNumberOfCells = newCluster.cells->size() - cluster.cells->size();

    auto newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);

    auto const& newCell2 = newCellByCellId.at(cellId2);
    auto const& newToken = newCell2.tokens->at(0);

    result.token = newToken;
    result.constructorCell = newCell2;

    newCellByCellId.erase(cellId1);
    newCellByCellId.erase(cellId2);
    newCellByCellId.erase(cellId3);
    if (!newCellByCellId.empty()) {
        result.constructedCell = newCellByCellId.begin()->second;
    }

    return result;
}

auto ConstructorGpuTests::runConstructionOnTriangleClusterTest(TokenDescription const& token) const -> TestResult
{
    ClusterDescription cluster;
    cluster.setId(_numberGen->getId()).setVel(QVector2D{}).setAngle(0).setAngularVel(0);

    auto const center = QVector2D{10, 10};
    auto const cellEnergy = _parameters.cellFunctionConstructorOffspringCellEnergy;
    auto const relPos1 = QVector2D{0, -1};
    auto const relPos2 = QVector2D{-1, 0};
    auto const relPos3 = QVector2D{0, 1};
    auto const relPos4 = QVector2D{0, 0};
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
    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
    checkEnergy(origData, newData);

    auto const& newCluster = newData.clusters->at(0);

    TestResult result;
    result.movementOfCenter = *newCluster.pos - *cluster.pos;
    result.increaseNumberOfCells = newCluster.cells->size() - cluster.cells->size();

    auto newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);

    auto const& newCell4 = newCellByCellId.at(cellId4);
    auto const& newToken = newCell4.tokens->at(0);

    result.token = newToken;
    result.constructorCell = newCell4;

    newCellByCellId.erase(cellId1);
    newCellByCellId.erase(cellId2);
    newCellByCellId.erase(cellId3);
    newCellByCellId.erase(cellId4);
    if (!newCellByCellId.empty()) {
        result.constructedCell = newCellByCellId.begin()->second;
    }

    return result;
}

TokenDescription ConstructorGpuTests::createTokenForConstruction(TokenForConstructionParameters tokenParameters) const
{
    auto token = createSimpleToken();
    (*token.data)[Enums::Constr::IN] = tokenParameters._constructionInput;
    (*token.data)[Enums::Constr::IN_OPTION] = tokenParameters._constructionOption;
    (*token.data)[Enums::Constr::INOUT_ANGLE] = QuantityConverter::convertAngleToData(tokenParameters._angle);
    (*token.data)[Enums::Constr::IN_DIST] = QuantityConverter::convertDistanceToData(tokenParameters._distance);
    (*token.data)[Enums::Constr::IN_CELL_MAX_CONNECTIONS] = 2;
    token.energy = tokenParameters._energy.get_value_or(
        2 * _parameters.tokenMinEnergy + 2 * _parameters.cellFunctionConstructorOffspringCellEnergy);
    return token;
}

void ConstructorGpuTests::checkResult(TestResult const& testResult, Expectations const& expectations) const
{
    auto const& token = testResult.token;
    if (Enums::ConstrIn::DO_NOTHING == token.data->at(Enums::Constr::IN)) {
        EXPECT_FALSE(testResult.constructedCell);
        return;
    }

    EXPECT_EQ(expectations._tokenOutput, token.data->at(Enums::Constr::OUT));

    if (!expectations._destruction) {
        EXPECT_TRUE(isCompatible(testResult.movementOfCenter, QVector2D{}));

        if (Enums::ConstrOut::SUCCESS == expectations._tokenOutput) {
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
                predEqual,
                0,
                (*testResult.constructorCell.pos + *expectations._constructedRelCellPos
                 - *testResult.constructedCell->pos)
                    .length(),
                0.05);

            return;
        } else {
            EXPECT_FALSE(testResult.constructedCell);
        }
    }
}

TEST_F(ConstructorGpuTests, testDoNothing)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::DO_NOTHING));
    auto result = runConstructionOnLineClusterTest(ConstructionOnLineClusterTestParameters().token(token));
    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructCellOnLineCluster_standardParameters)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto result = runConstructionOnLineClusterTest(ConstructionOnLineClusterTestParameters().token(token));
    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedRelCellPos(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructCellOnLineCluster_ignoreDistanceOnFirstConstructedCell1)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).distance(_offspringDistance / 2));
    auto const result = runConstructionOnLineClusterTest(ConstructionOnLineClusterTestParameters().token(token));

    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedRelCellPos(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructCellOnLineCluster_ignoreDistanceOnFirstConstructedCell2)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).distance(_offspringDistance * 2));
    auto const result = runConstructionOnLineClusterTest(ConstructionOnLineClusterTestParameters().token(token));

    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedRelCellPos(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructCellOnLineCluster_rightHandSide)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(90.0f));
    auto const result = runConstructionOnLineClusterTest(ConstructionOnLineClusterTestParameters().token(token));

    auto const expectedCellPos = QVector2D{0, _offspringDistance};
    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedRelCellPos(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructCellOnLineCluster_leftHandSide)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(-90.0f));
    auto const result = runConstructionOnLineClusterTest(ConstructionOnLineClusterTestParameters().token(token));

    auto const expectedCellPos = QVector2D{0, -_offspringDistance};
    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedRelCellPos(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructCellOnLineCluster_errorMaxConnectionsReached)
{
    _parameters.cellMaxBonds = 1;
    _context->setSimulationParameters(_parameters);

    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runConstructionOnLineClusterTest(ConstructionOnLineClusterTestParameters().token(token));

    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_CONNECTION));
}

TEST_F(ConstructorGpuTests, testConstructCellOnLineCluster_errorNoEnergy)
{
    auto const lowTokenEnergy = _parameters.tokenMinEnergy + _parameters.cellFunctionConstructorOffspringCellEnergy / 2;
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).energy(lowTokenEnergy));
    auto const result = runConstructionOnLineClusterTest(ConstructionOnLineClusterTestParameters().token(token));

    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_NO_ENERGY));
}

TEST_F(ConstructorGpuTests, testConstructCellOnLineCluster_otherClusterObstacle_safeMode)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result =
        runConstructionOnLineClusterTest(ConstructionOnLineClusterTestParameters().token(token).obstacle(true));

    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructCellOnLineCluster_otherClusterObstacle_unsafeMode)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::UNSAFE));
    auto const result =
        runConstructionOnLineClusterTest(ConstructionOnLineClusterTestParameters().token(token).obstacle(true));

    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructCellOnLineCluster_otherClusterObstacle_brutforceMode)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::BRUTEFORCE));
    auto const result =
        runConstructionOnLineClusterTest(ConstructionOnLineClusterTestParameters().token(token).obstacle(true));

    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    checkResult(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedRelCellPos(expectedCellPos).destruction(true));
}

TEST_F(ConstructorGpuTests, testConstructCellOnLineCluster_sameClusterObstacle_safeMode)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(90));
    auto const result = runConstructionOnWedgeClusterTest(token, 180, 0);

    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructCellOnLineCluster_sameClusterObstacle_unsafeMode)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::UNSAFE).angle(90));
    auto const result = runConstructionOnWedgeClusterTest(token, 180, 0);

    auto const expectedCellPos = QVector2D{0, _offspringDistance};
    checkResult(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedRelCellPos(expectedCellPos).destruction(true));
}

TEST_F(ConstructorGpuTests, testConstructCellOnLineCluster_sameClusterObstacle_brutforceMode)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::BRUTEFORCE).angle(90));
    auto const result = runConstructionOnWedgeClusterTest(token, 180, 0);

    auto const expectedCellPos = QVector2D{0, _offspringDistance};
    checkResult(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedRelCellPos(expectedCellPos).destruction(true));
}

TEST_F(ConstructorGpuTests, testConstructCellOnWedgeCluster_rightHandSide)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runConstructionOnWedgeClusterTest(token, 90, 0);

    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedRelCellPos(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructCellOnWedgeCluster_leftHandSide)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runConstructionOnWedgeClusterTest(token, 270, 0);

    auto const expectedCellPos = QVector2D{-_offspringDistance, 0};
    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedRelCellPos(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructCellOnWedgeCluster_diagonal)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runConstructionOnWedgeClusterTest(token, 90, 45);

    auto const expectedCellPos = QVector2D{_offspringDistance / sqrtf(2), _offspringDistance / sqrtf(2)};
    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedRelCellPos(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructCellOnTiangleCluster)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runConstructionOnTriangleClusterTest(token);

    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    checkResult(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedRelCellPos(expectedCellPos));
}

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
        CellDescription constructedCell;
    };
    TestResult runStandardConstructionTest(TokenDescription const& token) const;
    TestResult runConstructionOnWedgeClusterTest(TokenDescription const& token, float wedgeAngle, float clusterAngle)
        const;

    TokenDescription createTokenForSimpleConstruction(
        Enums::ConstrIn::Type constructionIn,
        Enums::ConstrInOption::Type option,
        float angle,
        float distance) const;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

void ConstructorGpuTests::SetUp()
{
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
    EXPECT_EQ(3, newCluster.cells->size());
    EXPECT_TRUE(isCompatible(cluster.pos, newCluster.pos));

    TestResult result;
    result.token = newToken;
    result.constructorCell = newSecondCell;

    newCellByCellId.erase(firstCell.id);
    newCellByCellId.erase(secondCell.id);
    result.constructedCell = newCellByCellId.begin()->second;

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
    auto const relPos2 = Physics::unitVectorOfAngle(clusterAngle + 90) / 2;
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
    EXPECT_EQ(4, newCluster.cells->size());
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
    result.constructedCell = newCellByCellId.begin()->second;

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

TEST_F(ConstructorGpuTests, testConstructSimpleCell_standardParameters)
{
    auto const token =
        createTokenForSimpleConstruction(Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 0.0f, 0.0f);
    runStandardConstructionTest(token);
}

TEST_F(ConstructorGpuTests, testConstructSimpleCell_ignoreDistanceOnFirstConstructedCell1)
{
    auto const offspringDistance = _parameters.cellFunctionConstructorOffspringCellDistance;

    auto const token = createTokenForSimpleConstruction(
        Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 0.0f, offspringDistance / 2);
    auto const result = runStandardConstructionTest(token);

    ASSERT_PRED3(
        predEqual, 0,
        (*result.constructorCell.pos + QVector2D{offspringDistance, 0} - *result.constructedCell.pos).length(),
        0.05);
}

TEST_F(ConstructorGpuTests, testConstructSimpleCell_ignoreDistanceOnFirstConstructedCell2)
{
    auto const offspringDistance = _parameters.cellFunctionConstructorOffspringCellDistance;

    auto const token = createTokenForSimpleConstruction(
        Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 0.0f, offspringDistance * 2);
    auto const result = runStandardConstructionTest(token);

    ASSERT_PRED3(
        predEqual, 0,
        (*result.constructorCell.pos + QVector2D{offspringDistance, 0} - *result.constructedCell.pos).length(),
        0.05);
}

TEST_F(ConstructorGpuTests, testConstructSimpleCell_rightHandSide)
{
    auto const offspringDistance = _parameters.cellFunctionConstructorOffspringCellDistance;

    auto const token = createTokenForSimpleConstruction(
        Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 90.0f, 0.0f);
    auto const result = runStandardConstructionTest(token);

    ASSERT_PRED3(
        predEqual, 0, 
        (*result.constructorCell.pos + QVector2D{0, offspringDistance} - *result.constructedCell.pos).length(),
        0.05);
}

TEST_F(ConstructorGpuTests, testConstructSimpleCell_leftHandSide)
{
    auto const token = createTokenForSimpleConstruction(
        Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, -90.0f, 0.0f);
    auto const result = runStandardConstructionTest(token);

    auto const offspringDistance = _parameters.cellFunctionConstructorOffspringCellDistance;
    ASSERT_PRED3(
        predEqual, 0,
        (*result.constructorCell.pos + QVector2D{ 0, -offspringDistance } -*result.constructedCell.pos).length(),
        0.05);
}

TEST_F(ConstructorGpuTests, testConstructSimpleCellOnWedgeCluster_standardParameters)
{
    auto const token =
        createTokenForSimpleConstruction(Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 0.0f, 0.0f);
    auto const result = runConstructionOnWedgeClusterTest(token, 90, 0);

    auto const offspringDistance = _parameters.cellFunctionConstructorOffspringCellDistance;
    ASSERT_PRED3(
        predEqual, 0,
        (*result.constructorCell.pos + QVector2D{offspringDistance, 0} - *result.constructedCell.pos).length(),
        0.05);
}

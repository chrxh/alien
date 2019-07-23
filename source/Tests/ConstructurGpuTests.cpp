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

    struct SimpleTestResult
    {
        TokenDescription token;
        CellDescription constructorCell;
        CellDescription constructedCell;
    };
    SimpleTestResult runSimpleTest(TokenDescription const& token) const;

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

auto ConstructorGpuTests::runSimpleTest(TokenDescription const & token) const -> SimpleTestResult
{
    SimpleTestResult result;

    DataDescription origData;
    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);

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

    result.token = newToken;
    result.constructorCell = newSecondCell;

    newCellByCellId.erase(firstCell.id);
    newCellByCellId.erase(secondCell.id);
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

TEST_F(ConstructorGpuTests, testConstructSimpleCell)
{
    auto const token =
        createTokenForSimpleConstruction(Enums::ConstrIn::SAFE, Enums::ConstrInOption::STANDARD, 0.0f, 1.0f);
    runSimpleTest(token);
}

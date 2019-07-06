#include "Base/ServiceLocator.h"
#include "ModelBasic/QuantityConverter.h"

#include "IntegrationGpuTestFramework.h"

class ScannerGpuTests
    : public IntegrationGpuTestFramework
{
public:
    ScannerGpuTests() : IntegrationGpuTestFramework({ 10, 10 })
    {}

    virtual ~ScannerGpuTests() = default;

protected:
    virtual void SetUp();

    void checkedScannedCellWithToken(CellDescription const& cell, TokenDescription const& token) const;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

void ScannerGpuTests::SetUp()
{
    _parameters.radiationProb = 0;    //exclude radiation
    _context->setSimulationParameters(_parameters);
}

void ScannerGpuTests::checkedScannedCellWithToken(CellDescription const& cell, TokenDescription const& token) const
{
    EXPECT_EQ(*cell.maxConnections, token.data->at(Enums::Scanner::OUT_CELL_MAX_CONNECTIONS));
    EXPECT_EQ(*cell.tokenBranchNumber, token.data->at(Enums::Scanner::OUT_CELL_BRANCH_NO));
    EXPECT_EQ(cell.cellFeature->type, token.data->at(Enums::Scanner::OUT_CELL_FUNCTION));
}

TEST_F(ScannerGpuTests, testScanOriginCell)
{
    DataDescription origData;
    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    firstCell.tokenBranchNumber = 0;
    auto& secondCell = cluster.cells->at(1);
    secondCell.tokenBranchNumber = 1;
    secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);
    auto token = createSimpleToken();
    firstCell.addToken(token);
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 0;
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newCell = cellByCellId.at(secondCell.id);
    auto const& newToken = newCell.tokens->at(0);

    checkedScannedCellWithToken(secondCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::SUCCESS, newToken.data->at(Enums::Scanner::OUT));
    EXPECT_EQ(0, newToken.data->at(Enums::Scanner::OUT_DISTANCE));
    EXPECT_EQ(1, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
    EXPECT_EQ(2, newToken.data->at(Enums::Scanner::OUT_MASS));
}

TEST_F(ScannerGpuTests, testScanSecondCell)
{
    DataDescription origData;
    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    firstCell.tokenBranchNumber = 0;
    firstCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);
    auto& secondCell = cluster.cells->at(1);
    secondCell.tokenBranchNumber = 1;
    secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 1;
    firstCell.addToken(token);
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newCell = cellByCellId.at(secondCell.id);
    auto const& newToken = newCell.tokens->at(0);

    checkedScannedCellWithToken(firstCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::FINISHED, newToken.data->at(Enums::Scanner::OUT));
    EXPECT_EQ(QuantityConverter::convertShiftLenToData((*firstCell.pos - *secondCell.pos).length()), newToken.data->at(Enums::Scanner::OUT_DISTANCE));
    EXPECT_EQ(2, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
    EXPECT_EQ(2, newToken.data->at(Enums::Scanner::OUT_MASS));
}

#include "Base/ServiceLocator.h"
#include "ModelBasic/QuantityConverter.h"

#include "IntegrationGpuTestFramework.h"

class ScannerGpuTests
    : public IntegrationGpuTestFramework
{
public:
    ScannerGpuTests() : IntegrationGpuTestFramework()
    {}

    virtual ~ScannerGpuTests() = default;

protected:
    virtual void SetUp();

    void checkScannedCellWithToken(
        CellDescription const& cell,
        CellDescription const& prevCell,
        CellDescription const& prevPrevCell,
        TokenDescription const& token) const;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

void ScannerGpuTests::SetUp()
{
    _parameters.radiationProb = 0;    //exclude radiation
    _context->setSimulationParameters(_parameters);
}

void ScannerGpuTests::checkScannedCellWithToken(
    CellDescription const& cell,
    CellDescription const& prevCell,
    CellDescription const& prevPrevCell,
    TokenDescription const& token) const
{
    EXPECT_EQ(*cell.maxConnections, token.data->at(Enums::Scanner::OUT_CELL_MAX_CONNECTIONS));
    EXPECT_EQ(*cell.tokenBranchNumber, token.data->at(Enums::Scanner::OUT_CELL_BRANCH_NO));
    EXPECT_EQ(cell.cellFeature->type, token.data->at(Enums::Scanner::OUT_CELL_FUNCTION));
    EXPECT_EQ(static_cast<int>(*cell.energy), token.data->at(Enums::Scanner::OUT_ENERGY));
    EXPECT_EQ(
        QuantityConverter::convertShiftLenToData((*cell.pos - *prevCell.pos).length()),
        token.data->at(Enums::Scanner::OUT_DISTANCE));

    if (prevCell.pos != prevPrevCell.pos) {
        auto a1 = Physics::angleOfVector(*prevPrevCell.pos - *prevCell.pos);
        auto a2 = Physics::angleOfVector(*prevCell.pos - *cell.pos);
        auto angle = a1 - a2;
        EXPECT_EQ(QuantityConverter::convertAngleToData(angle), token.data->at(Enums::Scanner::OUT_ANGLE));
    }
    else {
        EXPECT_EQ(0, token.data->at(Enums::Scanner::OUT_ANGLE));
    }
}

TEST_F(ScannerGpuTests, testScanOriginCell)
{
    DataDescription origData;
    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);

    auto& firstCell = cluster.cells->at(0);
    firstCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 0;
    firstCell.addToken(token);
    firstCell.energy = _parameters.cellMinEnergy + 10.5f;

    auto& secondCell = cluster.cells->at(1);
    secondCell.tokenBranchNumber = 1;
    secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newSecondCell = cellByCellId.at(secondCell.id);
    auto const& newToken = newSecondCell.tokens->at(0);

    checkScannedCellWithToken(newSecondCell, newSecondCell, newSecondCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::SUCCESS, newToken.data->at(Enums::Scanner::OUT));
    EXPECT_EQ(0, newToken.data->at(Enums::Scanner::OUT_DISTANCE));
    EXPECT_EQ(1, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
    EXPECT_EQ(2, newToken.data->at(Enums::Scanner::OUT_MASS));
}

/**
* Situation: - 5x5 cluster with scanner function at middle cell
*            - token coming from left
*            - scanning cell number is 1
* Expected result: correct cell (at (1, 2)-position of the cluster) should be scanned
*/
TEST_F(ScannerGpuTests, testScanSecondCell)
{
    DataDescription origData;
    auto cluster = createRectangularCluster({ 5, 5 }, QVector2D{}, QVector2D{});

    auto& tokenSourceCell = cluster.cells->at(11);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 1;
    tokenSourceCell.addToken(token);
    tokenSourceCell.energy = _parameters.cellMinEnergy + 10.5f;

    auto& middleCell = cluster.cells->at(12);
    middleCell.tokenBranchNumber = 1;
    middleCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newCell = cellByCellId.at(middleCell.id);
    auto const& newTokenSourceCell = cellByCellId.at(tokenSourceCell.id);
    auto const& newToken = newCell.tokens->at(0);

    checkScannedCellWithToken(newTokenSourceCell, newCell, newCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::SUCCESS, newToken.data->at(Enums::Scanner::OUT));
    EXPECT_EQ(2, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
    EXPECT_EQ(25, newToken.data->at(Enums::Scanner::OUT_MASS));
}

/**
* Situation: - 5x5 cluster with scanner function at middle cell
*            - token coming from left
*            - scanning cell number is 2
* Expected result: correct cell (at (1, 3)-position of the cluster) should be scanned
*/
TEST_F(ScannerGpuTests, testScanThirdCell)
{
    DataDescription origData;
    auto cluster = createRectangularCluster({ 5, 5 }, QVector2D{}, QVector2D{});

    auto& tokenSourceCell = cluster.cells->at(11);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 2;
    tokenSourceCell.addToken(token);

    auto& middleCell = cluster.cells->at(12);
    middleCell.tokenBranchNumber = 1;
    middleCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);

    auto& scanCell = cluster.cells->at(16);
    scanCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);
    scanCell.tokenBranchNumber = 2;
    scanCell.energy = _parameters.cellMinEnergy + 10.5f;

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newMiddleCell = cellByCellId.at(middleCell.id);
    auto const& newScanCell = cellByCellId.at(scanCell.id);
    auto const& newTokenSourceCell = cellByCellId.at(tokenSourceCell.id);
    auto const& newToken = newMiddleCell.tokens->at(0);

    checkScannedCellWithToken(newScanCell, newTokenSourceCell, newMiddleCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::SUCCESS, newToken.data->at(Enums::Scanner::OUT));
    EXPECT_EQ(3, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
    EXPECT_EQ(25, newToken.data->at(Enums::Scanner::OUT_MASS));
}


/**
* Situation: - 5x5 cluster with scanner function at middle cell
*            - token coming from left
*            - scanning cell number is 9
* Expected result: correct cell (at (0, 1)-position of the cluster) should be scanned
*/
TEST_F(ScannerGpuTests, testScanDistantCell)
{
    DataDescription origData;
    auto cluster = createRectangularCluster({ 5, 5 }, QVector2D{}, QVector2D{});

    auto& tokenSourceCell = cluster.cells->at(11);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 9;
    tokenSourceCell.addToken(token);

    auto& middleCell = cluster.cells->at(12);
    middleCell.tokenBranchNumber = 1;
    middleCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);

    auto& scanCell = cluster.cells->at(5);
    scanCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);
    scanCell.tokenBranchNumber = 2;
    scanCell.energy = _parameters.cellMinEnergy + 10.5f;

    auto& prevScanCell = cluster.cells->at(6);
    auto& prevPrevScanCell = cluster.cells->at(7);

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newMiddleCell = cellByCellId.at(middleCell.id);
    auto const& newScanCell = cellByCellId.at(scanCell.id);
    auto const& newPrevScanCell = cellByCellId.at(prevScanCell.id);
    auto const& newPrevPrevScanCell = cellByCellId.at(prevPrevScanCell.id);
    auto const& newToken = newMiddleCell.tokens->at(0);

    checkScannedCellWithToken(newScanCell, newPrevScanCell, newPrevPrevScanCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::SUCCESS, newToken.data->at(Enums::Scanner::OUT));
    EXPECT_EQ(10, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
    EXPECT_EQ(25, newToken.data->at(Enums::Scanner::OUT_MASS));
}

/**
* Situation: - 5x5 cluster with scanner function at middle cell
*            - token coming from left
*            - scanning cell number is 24
* Expected result: correct cell (at (0, 0)-position of the cluster) should be scanned
*/
TEST_F(ScannerGpuTests, testScanFinished)
{
    DataDescription origData;
    auto cluster = createRectangularCluster({ 5, 5 }, QVector2D{}, QVector2D{});

    auto& tokenSourceCell = cluster.cells->at(11);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 24;
    tokenSourceCell.addToken(token);

    auto& middleCell = cluster.cells->at(12);
    middleCell.tokenBranchNumber = 1;
    middleCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);

    auto& scanCell = cluster.cells->at(0);
    scanCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);
    scanCell.tokenBranchNumber = 2;
    scanCell.energy = _parameters.cellMinEnergy + 10.5f;

    auto& prevScanCell = cluster.cells->at(1);
    auto& prevPrevScanCell = cluster.cells->at(2);

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newMiddleCell = cellByCellId.at(middleCell.id);
    auto const& newScanCell = cellByCellId.at(scanCell.id);
    auto const& newPrevScanCell = cellByCellId.at(prevScanCell.id);
    auto const& newPrevPrevScanCell = cellByCellId.at(prevPrevScanCell.id);
    auto const& newToken = newMiddleCell.tokens->at(0);

    checkScannedCellWithToken(newScanCell, newPrevScanCell, newPrevPrevScanCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::FINISHED, newToken.data->at(Enums::Scanner::OUT));
    EXPECT_EQ(25, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
    EXPECT_EQ(25, newToken.data->at(Enums::Scanner::OUT_MASS));
}

/**
* Situation: - 5x5 cluster with scanner function at middle cell
*            - token coming from left
*            - scanning cell number is 25
* Expected result: correct cell (at (2, 2)-position of the cluster) should be scanned
*/
TEST_F(ScannerGpuTests, testScanRestart1)
{
    DataDescription origData;
    auto cluster = createRectangularCluster({ 5, 5 }, QVector2D{}, QVector2D{});

    auto& tokenSourceCell = cluster.cells->at(11);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 25;
    tokenSourceCell.addToken(token);

    auto& middleCell = cluster.cells->at(12);
    middleCell.tokenBranchNumber = 1;
    middleCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);
    middleCell.energy = _parameters.cellMinEnergy + 10.5f;

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newMiddleCell = cellByCellId.at(middleCell.id);
    auto const& newToken = newMiddleCell.tokens->at(0);

    checkScannedCellWithToken(newMiddleCell, newMiddleCell, newMiddleCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::RESTART, newToken.data->at(Enums::Scanner::OUT));
    EXPECT_EQ(1, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
    EXPECT_EQ(25, newToken.data->at(Enums::Scanner::OUT_MASS));
}

/**
* Situation: - 5x5 cluster with scanner function at middle cell
*            - token coming from left
*            - scanning cell number is 180
* Expected result: correct cell (at (2, 2)-position of the cluster) should be scanned
*/
TEST_F(ScannerGpuTests, testScanRestart2)
{
    DataDescription origData;
    auto cluster = createRectangularCluster({ 5, 5 }, QVector2D{}, QVector2D{});

    auto& tokenSourceCell = cluster.cells->at(11);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 180;
    tokenSourceCell.addToken(token);

    auto& middleCell = cluster.cells->at(12);
    middleCell.tokenBranchNumber = 1;
    middleCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);
    middleCell.energy = _parameters.cellMinEnergy + 10.5f;

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newMiddleCell = cellByCellId.at(middleCell.id);
    auto const& newToken = newMiddleCell.tokens->at(0);

    checkScannedCellWithToken(newMiddleCell, newMiddleCell, newMiddleCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::RESTART, newToken.data->at(Enums::Scanner::OUT));
    EXPECT_EQ(1, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
    EXPECT_EQ(25, newToken.data->at(Enums::Scanner::OUT_MASS));
}

/**
* Situation: - 5x5 cluster with scanner function at middle cell
*            - token coming from left
*            - scanning cell number is 255
* Expected result: correct cell (at (2, 2)-position of the cluster) should be scanned
*/
TEST_F(ScannerGpuTests, testScanMaxCellNumber)
{
    DataDescription origData;
    auto cluster = createRectangularCluster({ 5, 5 }, QVector2D{}, QVector2D{});

    auto& tokenSourceCell = cluster.cells->at(11);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 255;
    tokenSourceCell.addToken(token);

    auto& middleCell = cluster.cells->at(12);
    middleCell.tokenBranchNumber = 1;
    middleCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);
    middleCell.energy = _parameters.cellMinEnergy + 10.5f;

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newMiddleCell = cellByCellId.at(middleCell.id);
    auto const& newToken = newMiddleCell.tokens->at(0);

    checkScannedCellWithToken(newMiddleCell, newMiddleCell, newMiddleCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::RESTART, newToken.data->at(Enums::Scanner::OUT));
    EXPECT_EQ(1, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
    EXPECT_EQ(25, newToken.data->at(Enums::Scanner::OUT_MASS));
}

/**
* Situation: - 260 cluster with scanner function at first cell
*            - token coming from second cell
*            - scanning cell number is 255
* Expected result: correct cell (at 256-position of the cluster) should be scanned
*/
TEST_F(ScannerGpuTests, testScanMaxCellNumber_largeCluster)
{
    DataDescription origData;
    auto cluster = createRectangularCluster({ 260, 1 }, QVector2D{ }, QVector2D{});

    auto& tokenSourceCell = cluster.cells->at(1);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 255;
    tokenSourceCell.addToken(token);

    auto& firstCell = cluster.cells->at(0);
    firstCell.tokenBranchNumber = 1;
    firstCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);

    auto& scanCell = cluster.cells->at(255);
    scanCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);
    scanCell.tokenBranchNumber = 2;
    scanCell.energy = _parameters.cellMinEnergy + 10.5f;

    auto& prevScanCell = cluster.cells->at(254);
    auto& prevPrevScanCell = cluster.cells->at(253);

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newFirstCell = cellByCellId.at(firstCell.id);
    auto const& newScanCell = cellByCellId.at(scanCell.id);
    auto const& newPrevScanCell = cellByCellId.at(prevScanCell.id);
    auto const& newPrevPrevScanCell = cellByCellId.at(prevPrevScanCell.id);
    auto const& newToken = newFirstCell.tokens->at(0);

    checkScannedCellWithToken(newScanCell, newPrevScanCell, newPrevPrevScanCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::SUCCESS, newToken.data->at(Enums::Scanner::OUT));
    EXPECT_EQ(0, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
    EXPECT_EQ(255, static_cast<unsigned char>(newToken.data->at(Enums::Scanner::OUT_MASS)));
}

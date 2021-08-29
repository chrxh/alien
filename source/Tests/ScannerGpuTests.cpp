#include "Base/ServiceLocator.h"
#include "EngineInterface/QuantityConverter.h"
#include "EngineInterface/DescriptionFactory.h"

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
    CellDescription const& nextCell,
    TokenDescription const& token) const
{
    EXPECT_EQ(*cell.maxConnections, token.data->at(Enums::Scanner::OUT_CELL_MAX_CONNECTIONS));
    EXPECT_EQ(*cell.tokenBranchNumber, token.data->at(Enums::Scanner::OUT_CELL_BRANCH_NO));
    EXPECT_EQ(cell.cellFeature->getType(), token.data->at(Enums::Scanner::OUT_CELL_FUNCTION));
    auto temp = token.data->at(Enums::Scanner::OUT_ENERGY);
    EXPECT_EQ(static_cast<int>(*cell.energy), temp);

    auto expectedDistanceData =
        QuantityConverter::convertDistanceToData(_spaceProp->distance(*cell.pos, *prevCell.pos));
    auto actualDistanceData = token.data->at(Enums::Scanner::OUT_DISTANCE);
    EXPECT_TRUE(std::abs(expectedDistanceData - actualDistanceData) < 2);

    if (prevCell.pos != nextCell.pos && prevCell.pos != cell.pos && nextCell.pos != cell.pos) {
        auto const a1 = Physics::angleOfVector(_spaceProp->displacement(*nextCell.pos, *cell.pos));
        auto const a2 = Physics::angleOfVector(_spaceProp->displacement(*cell.pos, *prevCell.pos));
        auto const angle = a2 - a1;
        auto const expectedAngleData = QuantityConverter::convertAngleToData(angle);
        auto const actualAngleData = token.data->at(Enums::Scanner::OUT_ANGLE);
        EXPECT_TRUE(std::abs(expectedAngleData - actualAngleData) < 2);
    }
}

TEST_F(ScannerGpuTests, testScanOriginCell)
{
    DataDescription origData;
    auto cluster =
        _factory->createRect(DescriptionFactory::CreateRectParameters().size({2, 1}), _context->getNumberGenerator());

    auto& firstCell = cluster.cells->at(0);
    firstCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 0;
    firstCell.addToken(token);

    auto& secondCell = cluster.cells->at(1);
    secondCell.tokenBranchNumber = 1;
    secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newSecondCell = cellByCellId.at(secondCell.id);
    auto const& newToken = newSecondCell.tokens->at(0);

    checkScannedCellWithToken(newSecondCell, newSecondCell, newSecondCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::SUCCESS, newToken.data->at(Enums::Scanner::OUTPUT));
    EXPECT_EQ(0, newToken.data->at(Enums::Scanner::OUT_DISTANCE));
    EXPECT_EQ(1, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
}

/**
* Situation: - 5x5 cluster with scanner function at middle cell
*            - token coming from left
*            - scanning cell number is 1
* Expected result: correct cell (at (2, 3)-position of the cluster) should be scanned
*/
TEST_F(ScannerGpuTests, testScanSecondCell)
{
    DataDescription origData;
    auto cluster =
        _factory->createRect(DescriptionFactory::CreateRectParameters().size({5, 5}), _context->getNumberGenerator());

    auto& tokenSourceCell = cluster.cells->at(7);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 1;
    tokenSourceCell.addToken(token);

    auto& middleCell = cluster.cells->at(12);
    middleCell.tokenBranchNumber = 1;
    middleCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newCell = cellByCellId.at(middleCell.id);
    auto const& newTokenSourceCell = cellByCellId.at(tokenSourceCell.id);
    auto const& newToken = newCell.tokens->at(0);
    auto const& nextCell = cellByCellId.at(cluster.cells->at(8).id);

    checkScannedCellWithToken(newTokenSourceCell, newCell, nextCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::SUCCESS, newToken.data->at(Enums::Scanner::OUTPUT));
    EXPECT_EQ(2, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
}

/**
* Situation: - 5x5 cluster with scanner function at middle cell
*            - token coming from left
*            - scanning cell number is 2
* Expected result: correct cell (at (2, 4)-position of the cluster) should be scanned
*/
TEST_F(ScannerGpuTests, testScanThirdCell)
{
    DataDescription origData;
    auto cluster =
        _factory->createRect(DescriptionFactory::CreateRectParameters().size({5, 5}), _context->getNumberGenerator());

    auto& tokenSourceCell = cluster.cells->at(7);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 2;
    tokenSourceCell.addToken(token);

    auto& middleCell = cluster.cells->at(12);
    middleCell.tokenBranchNumber = 1;
    middleCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);

    auto& scanCell = cluster.cells->at(8);
    scanCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);
    scanCell.tokenBranchNumber = 2;
    scanCell.energy = _parameters.cellMinEnergy + 10.5f;

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newMiddleCell = cellByCellId.at(middleCell.id);
    auto const& newScanCell = cellByCellId.at(scanCell.id);
    auto const& newTokenSourceCell = cellByCellId.at(tokenSourceCell.id);
    auto const& newToken = newMiddleCell.tokens->at(0);
    auto const& nextCell = cellByCellId.at(cluster.cells->at(13).id);

    checkScannedCellWithToken(newScanCell, newTokenSourceCell, nextCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::SUCCESS, newToken.data->at(Enums::Scanner::OUTPUT));
    EXPECT_EQ(3, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
}

/**
* Situation: - hexagon cluster with 3 layers (edge length) with scanner function at middle cell
*            - token coming from left
*            - scanning cell number is 3
* Expected result: correct cell (at (3, 4)-position of the hexagon) should be scanned
*/
TEST_F(ScannerGpuTests, testScanFourthCell_onHexagon)
{
    auto const factory = ServiceLocator::getInstance().getService<DescriptionFactory>();
    auto hexagon = factory->createHexagon(
        DescriptionFactory::CreateHexagonParameters().layers(3).cellEnergy(_parameters.cellMinEnergy * 2), _numberGen);

    auto& tokenSourceCell = hexagon.cells->at(7);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 3;
    tokenSourceCell.addToken(token);

    auto& middleCell = hexagon.cells->at(12);
    middleCell.tokenBranchNumber = 1;
    middleCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);

    auto& scanCell = hexagon.cells->at(13);
    scanCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);
    scanCell.tokenBranchNumber = 2;

    auto& prevScanCell = hexagon.cells->at(8);

    DataDescription origData;
    origData.addCluster(hexagon);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newMiddleCell = cellByCellId.at(middleCell.id);
    auto const& newScanCell = cellByCellId.at(scanCell.id);
    auto const& newPrevScanCell = cellByCellId.at(prevScanCell.id);
    auto const& newTokenSourceCell = cellByCellId.at(tokenSourceCell.id);
    auto const& newToken = newMiddleCell.tokens->at(0);
    auto const& nextCell = cellByCellId.at(hexagon.cells->at(16).id);

    checkScannedCellWithToken(newScanCell, newPrevScanCell, nextCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::SUCCESS, newToken.data->at(Enums::Scanner::OUTPUT));
    EXPECT_EQ(4, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
}

/**
 * Situation: - 5x5 cluster with scanner function at middle cell
 *            - token coming from left
 *            - scanning cell number is 9
 * Expected result: correct cell (at (1, 2)-position of the cluster) should be scanned
 */
TEST_F(ScannerGpuTests, testScanDistantCell)
{
    DataDescription origData;
    auto cluster =
        _factory->createRect(DescriptionFactory::CreateRectParameters().size({5, 5}), _context->getNumberGenerator());

    auto& tokenSourceCell = cluster.cells->at(7);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 9;
    tokenSourceCell.addToken(token);

    auto& middleCell = cluster.cells->at(12);
    middleCell.tokenBranchNumber = 1;
    middleCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);

    auto& scanCell = cluster.cells->at(1);
    scanCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);
    scanCell.tokenBranchNumber = 2;

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newMiddleCell = cellByCellId.at(middleCell.id);
    auto const& newScanCell = cellByCellId.at(scanCell.id);
    auto const& prevScanCell = cellByCellId.at(cluster.cells->at(6).id);
    auto const& nextScanCell = cellByCellId.at(cluster.cells->at(2).id);
    auto const& newToken = newMiddleCell.tokens->at(0);

    checkScannedCellWithToken(newScanCell, prevScanCell, nextScanCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::SUCCESS, newToken.data->at(Enums::Scanner::OUTPUT));
    EXPECT_EQ(10, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
}

/**
 * Situation: - 5x5 cluster with scanner function at middle cell
 *            - token coming from left
 *            - scanning cell number is 24
 * Expected result: correct cell (at (1, 1)-position of the cluster) should be scanned
 */
TEST_F(ScannerGpuTests, testScanRestart1)
{
    DataDescription origData;
    auto cluster =
        _factory->createRect(DescriptionFactory::CreateRectParameters().size({5, 5}), _context->getNumberGenerator());

    auto& tokenSourceCell = cluster.cells->at(7);
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

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newMiddleCell = cellByCellId.at(middleCell.id);
    auto const& newScanCell = cellByCellId.at(scanCell.id);
    auto const& newPrevScanCell = cellByCellId.at(cluster.cells->at(1).id);
    auto const& newToken = newMiddleCell.tokens->at(0);

    checkScannedCellWithToken(newScanCell, newPrevScanCell, newScanCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::RESTART, newToken.data->at(Enums::Scanner::OUTPUT));
    EXPECT_EQ(0, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
}

/**
 * Situation: - 5x5 cluster with scanner function at middle cell
 *            - token coming from left
 *            - scanning cell number is 25
 * Expected result: correct cell (at (3, 3)-position of the cluster) should be scanned
 */
TEST_F(ScannerGpuTests, testScanRestart2)
{
    DataDescription origData;
    auto cluster =
        _factory->createRect(DescriptionFactory::CreateRectParameters().size({5, 5}), _context->getNumberGenerator());

    auto& tokenSourceCell = cluster.cells->at(7);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 25;
    tokenSourceCell.addToken(token);

    auto& middleCell = cluster.cells->at(12);
    middleCell.tokenBranchNumber = 1;
    middleCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);
    middleCell.energy = _parameters.cellMinEnergy + 10.5f;

    auto& scanCell = cluster.cells->at(0);
    scanCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);
    scanCell.tokenBranchNumber = 2;
    scanCell.energy = _parameters.cellMinEnergy + 10.5f;

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newMiddleCell = cellByCellId.at(middleCell.id);
    auto const& newScanCell = cellByCellId.at(scanCell.id);
    auto const& newPrevScanCell = cellByCellId.at(cluster.cells->at(1).id);
    auto const& newToken = newMiddleCell.tokens->at(0);

    checkScannedCellWithToken(newScanCell, newPrevScanCell, newScanCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::RESTART, newToken.data->at(Enums::Scanner::OUTPUT));
    EXPECT_EQ(0, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
}

/**
 * Situation: - 5x5 cluster with scanner function at middle cell
 *            - token coming from left
 *            - scanning cell number is 255
 * Expected result: correct cell (at (3, 3)-position of the cluster) should be scanned
 */
TEST_F(ScannerGpuTests, testScanMaxCellNumber)
{
    DataDescription origData;
    auto cluster =
        _factory->createRect(DescriptionFactory::CreateRectParameters().size({5, 5}), _context->getNumberGenerator());

    auto& tokenSourceCell = cluster.cells->at(7);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 255;
    tokenSourceCell.addToken(token);

    auto& middleCell = cluster.cells->at(12);
    middleCell.tokenBranchNumber = 1;
    middleCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);
    middleCell.energy = _parameters.cellMinEnergy + 10.5f;

    auto& scanCell = cluster.cells->at(0);
    scanCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);
    scanCell.tokenBranchNumber = 2;
    scanCell.energy = _parameters.cellMinEnergy + 10.5f;

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newMiddleCell = cellByCellId.at(middleCell.id);
    auto const& newScanCell = cellByCellId.at(scanCell.id);
    auto const& newPrevScanCell = cellByCellId.at(cluster.cells->at(1).id);
    auto const& newToken = newMiddleCell.tokens->at(0);

    checkScannedCellWithToken(newScanCell, newPrevScanCell, newScanCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::RESTART, newToken.data->at(Enums::Scanner::OUTPUT));
    EXPECT_EQ(0, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
}

/**
 * Situation: - cluster with 260 cell and with scanner function at first cell
 *            - token coming from second cell
 *            - scanning cell number is 255
 * Expected result: correct cell (at 257-position of the cluster) should be scanned
 */
TEST_F(ScannerGpuTests, testScanMaxCellNumber_largeCluster)
{
    DataDescription origData;
    auto cluster =
        _factory->createRect(DescriptionFactory::CreateRectParameters().size({260, 1}), _context->getNumberGenerator());

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

    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newFirstCell = cellByCellId.at(firstCell.id);
    auto const& newScanCell = cellByCellId.at(scanCell.id);
    auto const& prevScanCell = cellByCellId.at(cluster.cells->at(254).id);
    auto const& nextScanCell = cellByCellId.at(cluster.cells->at(256).id);
    auto const& newToken = newFirstCell.tokens->at(0);

    checkScannedCellWithToken(newScanCell, prevScanCell, nextScanCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::SUCCESS, newToken.data->at(Enums::Scanner::OUTPUT));
    EXPECT_EQ(0, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
}

TEST_F(ScannerGpuTests, testScanCell_exceed)
{
    DataDescription origData;
    auto cluster =
        _factory->createRect(DescriptionFactory::CreateRectParameters().size({2, 1}), _context->getNumberGenerator());

    auto& tokenSourceCell = cluster.cells->at(0);
    tokenSourceCell.tokenBranchNumber = 0;
    auto token = createSimpleToken();
    (*token.data)[Enums::Scanner::INOUT_CELL_NUMBER] = 1;
    tokenSourceCell.addToken(token);
    tokenSourceCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);

    auto& firstCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 1;
    firstCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SCANNER);


    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newFirstCell = cellByCellId.at(firstCell.id);
    auto const& newScanCell = cellByCellId.at(tokenSourceCell.id);
    auto const& newToken = newFirstCell.tokens->at(0);

    checkScannedCellWithToken(newScanCell, newFirstCell, newScanCell, newToken);
    EXPECT_EQ(Enums::ScannerOut::RESTART, newToken.data->at(Enums::Scanner::OUTPUT));
    EXPECT_EQ(0, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
}

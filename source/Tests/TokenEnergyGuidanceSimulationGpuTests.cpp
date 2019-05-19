#include "SimulationGpuTest.h"

class TokenEnergyGuidanceSimulationGpuTest
    : public SimulationGpuTest
{
public:
    virtual ~TokenEnergyGuidanceSimulationGpuTest() = default;

protected:
    virtual void SetUp();

    const float tokenTransferEnergyAmount = 10.0;
};

void TokenEnergyGuidanceSimulationGpuTest::SetUp()
{
    _parameters.radiationProb = 0;    //exclude radiation
    _context->setSimulationParameters(_parameters);
}

/**
* Situation: - one cluster with 2 cells with fitting branch number
*			 - first cell has one token with balance cell command
*			 - both cells have same high energy
* Expected result: energy should be transferred from cell to token
*/
TEST_F(TokenEnergyGuidanceSimulationGpuTest, testBalanceCell_highCellEnergy)
{
    auto const valueCell = 100.0f;

    DataDescription origData;

    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    *firstCell.energy = _parameters.cellMinEnergy + valueCell + 1 + tokenTransferEnergyAmount;
    secondCell.energy = firstCell.energy;
    auto token = createSimpleToken();
    auto& tokenData = *token.data;
    tokenData[Enums::EnergyGuidance::IN] = Enums::EnergyGuidanceIn::BALANCE_CELL;
    tokenData[Enums::EnergyGuidance::IN_VALUE_CELL] = valueCell;
    firstCell.addToken(token);
    origData.addCluster(cluster);

    uint64_t secondCellId = secondCell.id;

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    ASSERT_EQ(1, newData.clusters->size());
    auto newCluster = newData.clusters->at(0);

    EXPECT_EQ(2, newCluster.cells->size());

    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto newCell = cellByCellId.at(secondCellId);
    ASSERT_EQ(1, newCell.tokens->size());

    auto const& newToken = newCell.tokens->at(0);
    EXPECT_EQ(*token.energy + tokenTransferEnergyAmount, *newToken.energy);
    EXPECT_EQ(*secondCell.energy - tokenTransferEnergyAmount, *newCell.energy);

    checkEnergy(origData, newData);
}

/**
* Situation: - one cluster with 2 cells with fitting branch number
*			 - first cell has one token with balance cell command
*			 - both cells have same low energy
*            - token has high energy
* Expected result: energy should be transferred from token to cell
*/
TEST_F(TokenEnergyGuidanceSimulationGpuTest, testBalanceCell_lowCellEnergy)
{
    auto const valueCell = 100.0f;
    auto const valueToken = 30.0f;

    DataDescription origData;

    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    *firstCell.energy = _parameters.cellMinEnergy + valueCell - 1 + tokenTransferEnergyAmount;
    secondCell.energy = firstCell.energy;
    auto token = createSimpleToken();
    token.energy = _parameters.tokenMinEnergy + valueToken + 1 + tokenTransferEnergyAmount;
    auto& tokenData = *token.data;
    tokenData[Enums::EnergyGuidance::IN] = Enums::EnergyGuidanceIn::BALANCE_CELL;
    tokenData[Enums::EnergyGuidance::IN_VALUE_CELL] = valueCell;
    tokenData[Enums::EnergyGuidance::IN_VALUE_TOKEN] = valueToken;
    firstCell.addToken(token);
    origData.addCluster(cluster);

    uint64_t secondCellId = secondCell.id;

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto newCell = cellByCellId.at(secondCellId);

    auto const& newToken = newCell.tokens->at(0);
    EXPECT_EQ(*token.energy - tokenTransferEnergyAmount, *newToken.energy);
    EXPECT_EQ(*secondCell.energy + tokenTransferEnergyAmount, *newCell.energy);

    checkEnergy(origData, newData);
}

/**
* Situation: - one cluster with 2 cells with fitting branch number
*			 - first cell has one token with balance token command
*			 - both cells have same energy
*            - token has high energy
* Expected result: energy should be transferred from token to cell
*/
TEST_F(TokenEnergyGuidanceSimulationGpuTest, testBalanceToken_highTokenEnergy)
{
    auto const valueToken = 30.0f;

    DataDescription origData;

    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    auto token = createSimpleToken();
    token.energy = _parameters.tokenMinEnergy + valueToken + 1 + tokenTransferEnergyAmount;
    auto& tokenData = *token.data;
    tokenData[Enums::EnergyGuidance::IN] = Enums::EnergyGuidanceIn::BALANCE_TOKEN;
    tokenData[Enums::EnergyGuidance::IN_VALUE_TOKEN] = valueToken;
    firstCell.addToken(token);
    origData.addCluster(cluster);

    uint64_t secondCellId = secondCell.id;

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto newCell = cellByCellId.at(secondCellId);

    auto const& newToken = newCell.tokens->at(0);
    EXPECT_EQ(*token.energy - tokenTransferEnergyAmount, *newToken.energy);
    EXPECT_EQ(*secondCell.energy + tokenTransferEnergyAmount, *newCell.energy);

    checkEnergy(origData, newData);
}

/**
* Situation: - one cluster with 2 cells with fitting branch number
*			 - first cell has one token with balance token command
*			 - both cells have same energy
*            - token has low energy
* Expected result: energy should be transferred from token to cell
*/
TEST_F(TokenEnergyGuidanceSimulationGpuTest, testBalanceToken_lowTokenEnergy)
{
    auto const valueCell = 100.0f;
    auto const valueToken = 30.0f;

    DataDescription origData;

    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    *firstCell.energy = _parameters.cellMinEnergy + valueCell + 1 + tokenTransferEnergyAmount;
    secondCell.energy = firstCell.energy;
    auto token = createSimpleToken();
    token.energy = _parameters.tokenMinEnergy + valueToken - 1 + tokenTransferEnergyAmount;
    auto& tokenData = *token.data;
    tokenData[Enums::EnergyGuidance::IN] = Enums::EnergyGuidanceIn::BALANCE_TOKEN;
    tokenData[Enums::EnergyGuidance::IN_VALUE_CELL] = valueCell;
    tokenData[Enums::EnergyGuidance::IN_VALUE_TOKEN] = valueToken;
    firstCell.addToken(token);
    origData.addCluster(cluster);

    uint64_t secondCellId = secondCell.id;

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto newCell = cellByCellId.at(secondCellId);

    auto const& newToken = newCell.tokens->at(0);
    EXPECT_EQ(*token.energy + tokenTransferEnergyAmount, *newToken.energy);
    EXPECT_EQ(*secondCell.energy - tokenTransferEnergyAmount, *newCell.energy);

    checkEnergy(origData, newData);
}

/**
* Situation: - one cluster with 2 cells with fitting branch number
*			 - first cell has one token with balance cell and token command
*			 - both cells have same low energy
*            - token has high energy
* Expected result: energy should be transferred from token to cell
*/
TEST_F(TokenEnergyGuidanceSimulationGpuTest, testBalanceCellAndToken_highTokenEnergy_lowCellEnergy)
{
    auto const valueCell = 100.0f;
    auto const valueToken = 30.0f;

    DataDescription origData;

    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    *firstCell.energy = _parameters.cellMinEnergy + valueCell - 1;
    secondCell.energy = firstCell.energy;
    auto token = createSimpleToken();
    token.energy = _parameters.tokenMinEnergy + valueToken + 1 + tokenTransferEnergyAmount;
    auto& tokenData = *token.data;
    tokenData[Enums::EnergyGuidance::IN] = Enums::EnergyGuidanceIn::BALANCE_BOTH;
    tokenData[Enums::EnergyGuidance::IN_VALUE_CELL] = valueCell;
    tokenData[Enums::EnergyGuidance::IN_VALUE_TOKEN] = valueToken;
    firstCell.addToken(token);
    origData.addCluster(cluster);

    uint64_t secondCellId = secondCell.id;

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto newCell = cellByCellId.at(secondCellId);

    auto const& newToken = newCell.tokens->at(0);
    EXPECT_EQ(*token.energy - tokenTransferEnergyAmount, *newToken.energy);
    EXPECT_EQ(*secondCell.energy + tokenTransferEnergyAmount, *newCell.energy);

    checkEnergy(origData, newData);
}

/**
* Situation: - one cluster with 2 cells with fitting branch number
*			 - first cell has one token with balance cell and token command
*			 - both cells have same high energy
*            - token has low energy
* Expected result: energy should be transferred from cell to token
*/
TEST_F(TokenEnergyGuidanceSimulationGpuTest, testBalanceCellAndToken_lowTokenEnergy_highCellEnergy)
{
    auto const valueCell = 100.0f;
    auto const valueToken = 30.0f;

    DataDescription origData;

    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    *firstCell.energy = _parameters.cellMinEnergy + valueCell + 1 + tokenTransferEnergyAmount;
    secondCell.energy = firstCell.energy;
    auto token = createSimpleToken();
    token.energy = _parameters.tokenMinEnergy + valueToken - 1;
    auto& tokenData = *token.data;
    tokenData[Enums::EnergyGuidance::IN] = Enums::EnergyGuidanceIn::BALANCE_BOTH;
    tokenData[Enums::EnergyGuidance::IN_VALUE_CELL] = valueCell;
    tokenData[Enums::EnergyGuidance::IN_VALUE_TOKEN] = valueToken;
    firstCell.addToken(token);
    origData.addCluster(cluster);

    uint64_t secondCellId = secondCell.id;

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto newCell = cellByCellId.at(secondCellId);

    auto const& newToken = newCell.tokens->at(0);
    EXPECT_EQ(*token.energy + tokenTransferEnergyAmount, *newToken.energy);
    EXPECT_EQ(*secondCell.energy - tokenTransferEnergyAmount, *newCell.energy);

    checkEnergy(origData, newData);
}

/**
* Situation: - one cluster with 2 cells with fitting branch number
*			 - first cell has one token with harvest cell command
*			 - both cells have same high energy
* Expected result: energy should be transferred from cell to token
*/
TEST_F(TokenEnergyGuidanceSimulationGpuTest, testHarvestCell)
{
    auto const valueCell = 100.0f;

    DataDescription origData;

    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    *firstCell.energy = _parameters.cellMinEnergy + valueCell + 1 + tokenTransferEnergyAmount;
    secondCell.energy = firstCell.energy;
    auto token = createSimpleToken();
    auto& tokenData = *token.data;
    tokenData[Enums::EnergyGuidance::IN] = Enums::EnergyGuidanceIn::HARVEST_CELL;
    tokenData[Enums::EnergyGuidance::IN_VALUE_CELL] = valueCell;
    firstCell.addToken(token);
    origData.addCluster(cluster);

    uint64_t secondCellId = secondCell.id;

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto newCell = cellByCellId.at(secondCellId);

    auto const& newToken = newCell.tokens->at(0);
    EXPECT_EQ(*token.energy + tokenTransferEnergyAmount, *newToken.energy);
    EXPECT_EQ(*secondCell.energy - tokenTransferEnergyAmount, *newCell.energy);

    checkEnergy(origData, newData);
}

/**
* Situation: - one cluster with 2 cells with fitting branch number
*			 - first cell has one token with harvest cell command
*			 - both cells have same energy
*            - token has high energy
* Expected result: energy should be transferred from token to cell
*/
TEST_F(TokenEnergyGuidanceSimulationGpuTest, testHarvestToken)
{
    auto const valueToken = 30.0f;

    DataDescription origData;

    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    auto token = createSimpleToken();
    token.energy = _parameters.tokenMinEnergy + valueToken + 1 + tokenTransferEnergyAmount;
    auto& tokenData = *token.data;
    tokenData[Enums::EnergyGuidance::IN] = Enums::EnergyGuidanceIn::HARVEST_TOKEN;
    tokenData[Enums::EnergyGuidance::IN_VALUE_TOKEN] = valueToken;
    firstCell.addToken(token);
    origData.addCluster(cluster);

    uint64_t secondCellId = secondCell.id;

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto newCell = cellByCellId.at(secondCellId);

    auto const& newToken = newCell.tokens->at(0);
    EXPECT_EQ(*token.energy - tokenTransferEnergyAmount, *newToken.energy);
    EXPECT_EQ(*secondCell.energy + tokenTransferEnergyAmount, *newCell.energy);

    checkEnergy(origData, newData);
}

/**
* Situation: - one cluster with 2 cells with fitting branch number
*			 - first cell has two tokens with harvest cell command
*			 - both cells have same energy
*            - both tokens has high energy
* Expected result: energy should be transferred from token to cell twice
*/
TEST_F(TokenEnergyGuidanceSimulationGpuTest, testParallelization)
{
    auto const valueToken = 30.0f;

    DataDescription origData;

    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    auto token = createSimpleToken();
    token.energy = _parameters.tokenMinEnergy + valueToken + 1 + tokenTransferEnergyAmount;
    auto& tokenData = *token.data;
    tokenData[Enums::EnergyGuidance::IN] = Enums::EnergyGuidanceIn::HARVEST_TOKEN;
    tokenData[Enums::EnergyGuidance::IN_VALUE_TOKEN] = valueToken;
    firstCell.addToken(token);
    firstCell.addToken(token);
    origData.addCluster(cluster);

    uint64_t secondCellId = secondCell.id;

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto newCell = cellByCellId.at(secondCellId);

    auto const& newToken1 = newCell.tokens->at(0);
    auto const& newToken2 = newCell.tokens->at(1);
    EXPECT_EQ(*token.energy - tokenTransferEnergyAmount, *newToken1.energy);
    EXPECT_EQ(*token.energy - tokenTransferEnergyAmount, *newToken2.energy);
    EXPECT_EQ(*secondCell.energy + 2*tokenTransferEnergyAmount, *newCell.energy);

    checkEnergy(origData, newData);
}

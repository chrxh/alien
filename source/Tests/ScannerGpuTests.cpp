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
    const float SmallVelocity = 0.005f;
    const float SmallAngularVelocity = 0.05f;
    const float NeglectableVelocity = 0.001f;
    const float NeglectableAngularVelocity = 0.01f;

protected:
    virtual void SetUp();
};


void ScannerGpuTests::SetUp()
{
    _parameters.radiationProb = 0;    //exclude radiation
    _context->setSimulationParameters(_parameters);
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

    EXPECT_EQ(Enums::CellFunction::SCANNER, newToken.data->at(Enums::Scanner::OUT_CELL_FUNCTION));
    EXPECT_EQ(Enums::ScannerOut::SUCCESS, newToken.data->at(Enums::Scanner::OUT));
    EXPECT_EQ(0, newToken.data->at(Enums::Scanner::OUT_DISTANCE));
    EXPECT_EQ(1, newToken.data->at(Enums::Scanner::INOUT_CELL_NUMBER));
    EXPECT_EQ(2, newToken.data->at(Enums::Scanner::OUT_MASS));
}
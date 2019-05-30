#include "Base/ServiceLocator.h"
#include "ModelBasic/CellComputerCompiler.h"

#include "IntegrationGpuTestFramework.h"

class CellComputerSimulationGpuTest
    : public IntegrationGpuTestFramework
{
public:
    virtual ~CellComputerSimulationGpuTest() = default;

protected:
    virtual void SetUp();
};


void CellComputerSimulationGpuTest::SetUp()
{
    _parameters.radiationProb = 0;    //exclude radiation
    _context->setSimulationParameters(_parameters);
}

TEST_F(CellComputerSimulationGpuTest, testSimpleProgramm)
{
    string simpleProgram = "mov [1], 3";

    auto basicFacade = ServiceLocator::getInstance().getService<ModelBasicBuilderFacade>();
    CellComputerCompiler* compiler = basicFacade->buildCellComputerCompiler(_context->getSymbolTable(), _context->getSimulationParameters());

    CompilationResult compiledProgram = compiler->compileSourceCode(simpleProgram);

    DataDescription origData;
    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    firstCell.tokenBranchNumber = 0;
    auto& secondCell = cluster.cells->at(1);
    secondCell.tokenBranchNumber = 1;
    secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::COMPUTER).setConstData(compiledProgram.compilation);
    auto token = createSimpleToken();
    auto& tokenData = *token.data;
    firstCell.addToken(token);
    origData.addCluster(cluster);

    uint64_t secondCellId = secondCell.id;

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    auto newCluster = newData.clusters->at(0);
    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto newCell = cellByCellId.at(secondCellId);
    auto const& newToken = newCell.tokens->at(0);
    EXPECT_EQ(3, newToken.data->at(1));
}

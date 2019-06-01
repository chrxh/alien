#include "Base/ServiceLocator.h"
#include "ModelBasic/CellComputerCompiler.h"

#include "IntegrationGpuTestFramework.h"

class CellComputerSimulationGpuTest
    : public IntegrationGpuTestFramework
{
public:
    CellComputerSimulationGpuTest() : IntegrationGpuTestFramework({ 10, 10 })
    {}

    virtual ~CellComputerSimulationGpuTest() = default;

protected:
    virtual void SetUp();

    QByteArray runSimpleCellComputer(string const& progam) const;
};


void CellComputerSimulationGpuTest::SetUp()
{
    _parameters.radiationProb = 0;    //exclude radiation
    _context->setSimulationParameters(_parameters);
}

QByteArray CellComputerSimulationGpuTest::runSimpleCellComputer(string const & program) const
{
    auto basicFacade = ServiceLocator::getInstance().getService<ModelBasicBuilderFacade>();
    CellComputerCompiler* compiler = basicFacade->buildCellComputerCompiler(_context->getSymbolTable(), _context->getSimulationParameters());

    CompilationResult compiledProgram = compiler->compileSourceCode(program);

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

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    auto const& cellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newCell = cellByCellId.at(secondCell.id);
    auto const& newToken = newCell.tokens->at(0);
    return *newToken.data;
}

TEST_F(CellComputerSimulationGpuTest, testDereferencing1)
{
    string program = "mov [1], 3";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(3, data.at(1));
}

TEST_F(CellComputerSimulationGpuTest, testDereferencing2)
{
    string program = "mov [1], 3\nmov [[1]], 5";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(5, data.at(3));
}

TEST_F(CellComputerSimulationGpuTest, testDereferencing3)
{
    string program = "mov [1], 3\nmov [2], 5\nmov [[1]], [2]";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(5, data.at(3));
}

TEST_F(CellComputerSimulationGpuTest, testDereferencing4)
{
    string program = "mov [1], 3\nmov [2], 5\nmov [5], 7\nmov [[1]], [[2]]";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(7, data.at(3));
}

TEST_F(CellComputerSimulationGpuTest, testArithmetic)
{
    string program = "mov [1], 1\nmov [2], 5\nadd [1], [2]\nsub [1], 2\nmul [1],3\ndiv [1],2";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(6, data.at(1));
}

TEST_F(CellComputerSimulationGpuTest, testBitwiseOperators)
{
    string program = "mov [1], 1\nmov [2], 5\nmov [3], 6\nxor [1], 3\nor [2], 3\nand [3], 3";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(1));
    EXPECT_EQ(7, data.at(2));
    EXPECT_EQ(2, data.at(3));
}

TEST_F(CellComputerSimulationGpuTest, testConditions1)
{
    string program = "mov [1], 2\nif [1] < 3\nmov [2], 1\nelse\nmov [2],2\nendif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditions2)
{
    string program = "mov [1], 3\nif [1] < 3\nmov [2], 1\nelse\nmov [2],2\nendif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}
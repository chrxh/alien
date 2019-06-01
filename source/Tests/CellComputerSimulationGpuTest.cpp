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
    string program = 
        "mov [1], 3\n"\
        "mov [[1]], 5"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(5, data.at(3));
}

TEST_F(CellComputerSimulationGpuTest, testDereferencing3)
{
    string program = 
        "mov [1], 3\n"\
        "mov [2], 5\n"\
        "mov [[1]], [2]"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(5, data.at(3));
}

TEST_F(CellComputerSimulationGpuTest, testDereferencing4)
{
    string program = 
        "mov [1], 3\n"\
        "mov [2], 5\n"\
        "mov [5], 7\n"\
        "mov [[1]], [[2]]"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(7, data.at(3));
}

TEST_F(CellComputerSimulationGpuTest, testArithmetic)
{
    string program = 
        "mov [1], 1\n"\
        "mov [2], 5\n"\
        "add [1], [2]\n"\
        "sub [1], 2\n"\
        "mul [1],3\n"\
        "div [1],2"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(6, data.at(1));
}

TEST_F(CellComputerSimulationGpuTest, testBitwiseOperators)
{
    string program = 
        "mov [1], 1\n"\
        "mov [2], 5\n"\
        "mov [3], 6\n"\
        "xor [1], 3\n"\
        "or [2], 3\n"\
        "and [3], 3"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(1));
    EXPECT_EQ(7, data.at(2));
    EXPECT_EQ(2, data.at(3));
}

TEST_F(CellComputerSimulationGpuTest, testConditionGT1)
{
    string program = 
        "mov [1], 2\n"\
        "if [1] < 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionGT2)
{
    string program = 
        "mov [1], 3\n"\
        "if [1] < 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionGT3)
{
    string program = 
        "mov [1], 4\n"\
        "if [1] < 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionGE1)
{
    string program = 
        "mov [1], 3\n"\
        "if [1] <= 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionGE2)
{
    string program = 
        "mov [1], 4\n"\
        "if [1] <= 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionGE3)
{
    string program = 
        "mov [1], 2\n"\
        "if [1] <= 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionEQ1)
{
    string program = 
        "mov [1], 3\n"\
        "if [1] = 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionEQ2)
{
    string program = 
        "mov [1], 4\n"\
        "if [1] = 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionEQ3)
{
    string program = 
        "mov [1], 2\n"\
        "if [1] = 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionNEQ1)
{
    string program = 
        "mov [1], 3\n"\
        "if [1] != 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionNEQ2)
{
    string program =
        "mov [1], 4\n"\
        "if [1] != 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionNEQ3)
{
    string program =
        "mov [1], 2\n"\
        "if [1] != 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionLE1)
{
    string program =
        "mov [1], 3\n"\
        "if [1] >= 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionLE2)
{
    string program =
        "mov [1], 4\n"\
        "if [1] >= 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionLE3)
{
    string program = 
        "mov [1], 2\n"\
        "if [1] >= 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionLT1)
{
    string program = 
        "mov [1], 3\n"\
        "if [1] > 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionLT2)
{
    string program = 
        "mov [1], 4\n"\
        "if [1] > 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testConditionLT3)
{
    string program = 
        "mov [1], 2\n"\
        "if [1] > 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputerSimulationGpuTest, testInstructionAfterConditionClause)
{
    string program = 
        "if [1] != 3\n"\
        "mov [2], 1\n"\
        "else\n"\
        "mov [2],2\n"\
        "endif\n"
        "mov [5], 6\n"\
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(6, data.at(5));
}

TEST_F(CellComputerSimulationGpuTest, testNegativeNumbers1)
{
    string program =
        "mov [1], 1\n"\
        "sub [1], 2"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(-1, static_cast<int>(data.at(1)));
}

TEST_F(CellComputerSimulationGpuTest, testNegativeNumbers2)
{
    string program =
        "mov [1], -1"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(-1, static_cast<int>(data.at(1)));
}

TEST_F(CellComputerSimulationGpuTest, testNegativeNumbers3)
{
    string program =
        "mov [1], 1\n"\
        "sub [1], -1"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, static_cast<int>(data.at(1)));
}

TEST_F(CellComputerSimulationGpuTest, testNegativeNumbers4)
{
    string program =
        "mov [-1], 1"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(255));
}

TEST_F(CellComputerSimulationGpuTest, testOverflow1)
{
    string program =
        "mov [1], 127\n"\
        "add [1], 1\n"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(static_cast<char>(-128), data.at(1));
}

TEST_F(CellComputerSimulationGpuTest, testOverflow2)
{
    string program =
        "mov [1], 255\n"\
        "add [1], 1\n"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(0, data.at(1));
}

TEST_F(CellComputerSimulationGpuTest, testOverflow3)
{
    string program =
        "mov [1], 55\n"\
        "mul [1], 43\n"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(61, data.at(1));  //55 * 43 = 2365 = 61 (mod 256)
}

TEST_F(CellComputerSimulationGpuTest, testOverflow4)
{
    string program =
        "mov [1], 55\n"\
        "mul [1], 45\n"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(static_cast<char>(-85), data.at(1));  //55 * 45 = 2475 = 171 (mod 256)
}

TEST_F(CellComputerSimulationGpuTest, testOverflow5)
{
    string program =
        "mov [1], 55\n"\
        "mul [1], -45\n"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(85, data.at(1));  //55 * (-45) = -2475 = -171 (mod 256)
}

TEST_F(CellComputerSimulationGpuTest, testDivisionByZero)
{
    string program =
        "mov [1], 55\n"\
        "div [1], 0\n"
        ;
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(0, data.at(1));  
}

#include <gtest/gtest.h>

#include "EngineInterface/CellComputationCompiler.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/DescriptionHelper.h"

#include "IntegrationTestFramework.h"

class CellComputationTests : public IntegrationTestFramework
{
public:
    CellComputationTests()
        : IntegrationTestFramework({10, 10})
    {}

    virtual ~CellComputationTests() = default;

    std::string runSimpleCellComputer(std::string const& program) const;

protected:
    virtual void SetUp();
};


void CellComputationTests::SetUp()
{
    auto parameters = _simController->getSimulationParameters();
    parameters.radiationProb = 0;  //exclude radiation
    _simController->setSimulationParameters_async(parameters);
}

std::string CellComputationTests::runSimpleCellComputer(std::string const& program) const
{
    auto symbols = _simController->getSymbolMap();
    auto parameters = _simController->getSimulationParameters();

    CompilationResult compiledProgram = CellComputationCompiler::compileSourceCode(program, symbols, parameters);

    DataDescription origData
        = DescriptionHelper::createRect(DescriptionHelper::CreateRectParameters().width(2).height(1));
    auto& origFirstCell = origData.cells.at(0);
    origFirstCell.tokenBranchNumber = 0;
    auto& origSecondCell = origData.cells.at(1);
    origSecondCell.tokenBranchNumber = 1;
    origSecondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction_Computation).setConstData(compiledProgram.compilation);
    origFirstCell.addToken(createSimpleToken());

    _simController->setSimulationData(origData);
    _simController->calcSingleTimestep();

    DataDescription data = _simController->getSimulationData({0, 0}, _simController->getWorldSize());

    auto cellById = getCellById(data);
    auto const& secondCell = cellById.at(origSecondCell.id);
    EXPECT_EQ(1, secondCell.tokens.size());
    auto const& token = secondCell.tokens.at(0);
    return token.data;
}

TEST_F(CellComputationTests, testDereferencing1)
{
    std::string program = "mov [1], 3";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(3, data.at(1));
}

TEST_F(CellComputationTests, testDereferencing2)
{
    std::string program = "mov [1], 3\n"
                     "mov [[1]], 5";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(5, data.at(3));
}

TEST_F(CellComputationTests, testDereferencing3)
{
    std::string program = "mov [1], 3\n"
                     "mov [2], 5\n"
                     "mov [[1]], [2]";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(5, data.at(3));
}

TEST_F(CellComputationTests, testDereferencing4)
{
    std::string program = "mov [1], 3\n"
                     "mov [2], 5\n"
                     "mov [5], 7\n"
                     "mov [[1]], [[2]]";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(7, data.at(3));
}

TEST_F(CellComputationTests, testArithmetic)
{
    std::string program = "mov [1], 1\n"
                     "mov [2], 5\n"
                     "add [1], [2]\n"
                     "sub [1], 2\n"
                     "mul [1],3\n"
                     "div [1],2";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(6, data.at(1));
}

TEST_F(CellComputationTests, testBitwiseOperators)
{
    std::string program = "mov [1], 1\n"
                     "mov [2], 5\n"
                     "mov [3], 6\n"
                     "xor [1], 3\n"
                     "or [2], 3\n"
                     "and [3], 3";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(1));
    EXPECT_EQ(7, data.at(2));
    EXPECT_EQ(2, data.at(3));
}

TEST_F(CellComputationTests, testConditionGT1)
{
    std::string program = "mov [1], 2\n"
                     "if [1] < 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputationTests, testConditionGT2)
{
    std::string program = "mov [1], 3\n"
                     "if [1] < 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputationTests, testConditionGT3)
{
    std::string program = "mov [1], 4\n"
                     "if [1] < 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputationTests, testConditionGE1)
{
    std::string program = "mov [1], 3\n"
                     "if [1] <= 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputationTests, testConditionGE2)
{
    std::string program = "mov [1], 4\n"
                     "if [1] <= 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputationTests, testConditionGE3)
{
    std::string program = "mov [1], 2\n"
                     "if [1] <= 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputationTests, testConditionEQ1)
{
    std::string program = "mov [1], 3\n"
                     "if [1] = 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputationTests, testConditionEQ2)
{
    std::string program = "mov [1], 4\n"
                     "if [1] = 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputationTests, testConditionEQ3)
{
    std::string program = "mov [1], 2\n"
                     "if [1] = 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputationTests, testConditionNEQ1)
{
    std::string program = "mov [1], 3\n"
                     "if [1] != 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputationTests, testConditionNEQ2)
{
    std::string program = "mov [1], 4\n"
                     "if [1] != 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputationTests, testConditionNEQ3)
{
    std::string program = "mov [1], 2\n"
                     "if [1] != 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputationTests, testConditionLE1)
{
    std::string program = "mov [1], 3\n"
                     "if [1] >= 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputationTests, testConditionLE2)
{
    std::string program = "mov [1], 4\n"
                     "if [1] >= 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputationTests, testConditionLE3)
{
    std::string program = "mov [1], 2\n"
                     "if [1] >= 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputationTests, testConditionLT1)
{
    std::string program = "mov [1], 3\n"
                     "if [1] > 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputationTests, testConditionLT2)
{
    std::string program = "mov [1], 4\n"
                     "if [1] > 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(2));
}

TEST_F(CellComputationTests, testConditionLT3)
{
    std::string program = "mov [1], 2\n"
                     "if [1] > 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, data.at(2));
}

TEST_F(CellComputationTests, testInstructionAfterConditionClause)
{
    std::string program = "if [1] != 3\n"
                     "mov [2], 1\n"
                     "else\n"
                     "mov [2],2\n"
                     "endif\n"
                     "mov [5], 6\n";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(6, data.at(5));
}

TEST_F(CellComputationTests, testNegativeNumbers1)
{
    std::string program = "mov [1], 1\n"
                     "sub [1], 2";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(-1, static_cast<int>(data.at(1)));
}

TEST_F(CellComputationTests, testNegativeNumbers2)
{
    std::string program = "mov [1], -1";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(-1, static_cast<int>(data.at(1)));
}

TEST_F(CellComputationTests, testNegativeNumbers3)
{
    std::string program = "mov [1], 1\n"
                     "sub [1], -1";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(2, static_cast<int>(data.at(1)));
}

TEST_F(CellComputationTests, testNegativeNumbers4)
{
    std::string program = "mov [-1], 1";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(1, data.at(255));
}

TEST_F(CellComputationTests, testOverflow1)
{
    std::string program = "mov [1], 127\n"
                     "add [1], 1\n";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(static_cast<char>(-128), data.at(1));
}

TEST_F(CellComputationTests, testOverflow2)
{
    std::string program = "mov [1], 255\n"
                     "add [1], 1\n";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(0, data.at(1));
}

TEST_F(CellComputationTests, testOverflow3)
{
    std::string program = "mov [1], 55\n"
                     "mul [1], 43\n";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(61, data.at(1));  //55 * 43 = 2365 = 61 (mod 256)
}

TEST_F(CellComputationTests, testOverflow4)
{
    std::string program = "mov [1], 55\n"
                     "mul [1], 45\n";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(static_cast<char>(-85), data.at(1));  //55 * 45 = 2475 = 171 (mod 256)
}

TEST_F(CellComputationTests, testOverflow5)
{
    std::string program = "mov [1], 55\n"
                     "mul [1], -45\n";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(85, data.at(1));  //55 * (-45) = -2475 = -171 (mod 256)
}

TEST_F(CellComputationTests, testDivisionByZero)
{
    std::string program = "mov [1], 55\n"
                     "div [1], 0\n";
    auto data = runSimpleCellComputer(program);
    EXPECT_EQ(0, data.at(1));
}

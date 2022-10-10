#include "CellComputationCompiler.h"

#include <sstream>
#include <boost/algorithm/string/case_conv.hpp>

#include "SymbolMap.h"
#include "SimulationParameters.h"
#include "Definitions.h"

namespace
{
    enum class CompilerState
    {
        LOOKING_FOR_INSTR_START,
        LOOKING_FOR_INSTR_END,
        LOOKING_FOR_OP1_START,
        LOOKING_FOR_OP1_END,
        LOOKING_FOR_SEPARATOR,
        LOOKING_FOR_COMPARATOR,
        LOOKING_FOR_OP2_START,
        LOOKING_FOR_OP2_END
    };

    struct InstructionUncoded
    {
        bool readingFinished = false;
        std::string name;
        std::string operand1;
        std::string operand2;
        std::string comp;
    };

    int const BytesPerInstruction = 3;

    std::string lastChar(std::string const& s)
    {
        if (!s.empty()) {
            return std::string(1, s.back());
        }
        return s;
    }

    std::string lastTwoChars(std::string const& s)
    {
        if (s.size() > 1) {
            return s.substr(s.size() - 2, 2);
        }
        return s;
    }

    bool isNameChar(char const& c) { return std::isalnum(c) || (c == ':'); }

    bool gotoNextStateAndReturnSuccess(
        CompilerState& state,
        char& currentSymbol,
        InstructionUncoded& instruction,
        int bytePos,
        int codeSize)
    {
        boost::algorithm::to_lower(instruction.name);

        switch (state) {
        case CompilerState::LOOKING_FOR_INSTR_START: {
            if (std::isalpha(currentSymbol)) {
                state = CompilerState::LOOKING_FOR_INSTR_END;
                instruction.name = currentSymbol;
            }
        } break;
        case CompilerState::LOOKING_FOR_INSTR_END: {
            if (!std::isalpha(currentSymbol)) {
                if (instruction.name == "else" || instruction.name == "endif")
                    instruction.readingFinished = true;
                else
                    state = CompilerState::LOOKING_FOR_OP1_START;
            } else {
                instruction.name += currentSymbol;
                if ((bytePos + 1) == codeSize
                    && ((instruction.name == "else") || (instruction.name == "endif")))
                    instruction.readingFinished = true;
            }
        } break;
        case CompilerState::LOOKING_FOR_OP1_START: {
            if (isNameChar(currentSymbol) || (currentSymbol == '-') || (currentSymbol == '_') || (currentSymbol == '[')
                || (currentSymbol == '(')) {
                state = CompilerState::LOOKING_FOR_OP1_END;
                instruction.operand1 = currentSymbol;
            }
        } break;
        case CompilerState::LOOKING_FOR_OP1_END: {
            if ((currentSymbol == '<') || (currentSymbol == '>') || (currentSymbol == '=') || (currentSymbol == '!')) {
                state = CompilerState::LOOKING_FOR_COMPARATOR;
                instruction.comp = currentSymbol;
            } else if (currentSymbol == ',')
                state = CompilerState::LOOKING_FOR_OP2_START;
            else if (
                !isNameChar(currentSymbol) && (currentSymbol != '-') && (currentSymbol != '_') && (currentSymbol != '[')
                && (currentSymbol != ']') && (currentSymbol != '(') && (currentSymbol != ')'))
                state = CompilerState::LOOKING_FOR_SEPARATOR;
            else
                instruction.operand1 += currentSymbol;
        } break;
        case CompilerState::LOOKING_FOR_SEPARATOR: {
            if (currentSymbol == ',')
                state = CompilerState::LOOKING_FOR_OP2_START;
            else if (
                (currentSymbol == '<') || (currentSymbol == '>') || (currentSymbol == '=') || (currentSymbol == '!')) {
                state = CompilerState::LOOKING_FOR_COMPARATOR;
                instruction.comp = currentSymbol;
            } else if (
                isNameChar(currentSymbol) || (currentSymbol == '-') || (currentSymbol == '_') || (currentSymbol == '[')
                || (currentSymbol == ']') || (currentSymbol == '(') || (currentSymbol == ')'))
                return false;
        } break;
        case CompilerState::LOOKING_FOR_COMPARATOR: {
            if ((currentSymbol == '<') || (currentSymbol == '>') || (currentSymbol == '=') || (currentSymbol == '!'))
                instruction.comp += currentSymbol;
            else if (
                !isNameChar(currentSymbol) && (currentSymbol != '-') && (currentSymbol != '_') && (currentSymbol != '[')
                && (currentSymbol != '('))
                state = CompilerState::LOOKING_FOR_OP2_START;
            else {
                state = CompilerState::LOOKING_FOR_OP2_END;
                instruction.operand2 = currentSymbol;
            }
        } break;
        case CompilerState::LOOKING_FOR_OP2_START: {
            if (isNameChar(currentSymbol) || (currentSymbol == '-') || (currentSymbol == '_') || (currentSymbol == '[')
                || (currentSymbol == '(')) {
                state = CompilerState::LOOKING_FOR_OP2_END;
                instruction.operand2 = currentSymbol;
                if (bytePos == (codeSize - 1))
                    instruction.readingFinished = true;
            }
        } break;
        case CompilerState::LOOKING_FOR_OP2_END: {
            if (!isNameChar(currentSymbol) && (currentSymbol != '-') && (currentSymbol != '_') && (currentSymbol != '[')
                && (currentSymbol != ']') && (currentSymbol != '(') && (currentSymbol != ')'))
                instruction.readingFinished = true;
            else {
                instruction.operand2 += currentSymbol;
                if ((bytePos + 1) == codeSize)
                    instruction.readingFinished = true;
            }
        } break;
        }
        if ((currentSymbol == '\n') || ((bytePos + 1) == codeSize)) {
            if (!instruction.name.empty()) {
                instruction.readingFinished = true;
            }
        }
        return true;
    }

    std::string applyTableToCode(SymbolMap const& symbols, std::string s)
    {
        std::string prefix;
        std::string postfix;
        for (int i = 0; i < 2; ++i) {
            auto first = s.substr(0, 1);
            if (first == "[" || first == "(") {
                prefix = prefix + first;
                s = s.substr(1);
            }
        }
        for (int i = 0; i < 2; ++i) {
            auto last = lastChar(s);
            if (last == "]" || last == ")") {
                postfix = last + postfix;
                s = s.substr(0, s.size() - 1);
            }
        }
        auto findResult = symbols.find(s);
        if (findResult != symbols.end()) {
            s = findResult->second;
        }
        return prefix + s + postfix;
    }

    bool resolveInstructionAndReturnSuccess(
        SymbolMap const& symbols,
        CellInstruction& instructionCoded,
        InstructionUncoded instructionUncoded)
    {
        try {
            boost::algorithm::to_lower(instructionUncoded.name);
            instructionUncoded.operand1 = applyTableToCode(symbols, instructionUncoded.operand1);
            instructionUncoded.operand2 = applyTableToCode(symbols, instructionUncoded.operand2);

            //prepare data for instruction coding
            if (instructionUncoded.name == "mov")
                instructionCoded.operation = Enums::ComputationOperation_Mov;
            else if (instructionUncoded.name == "add")
                instructionCoded.operation = Enums::ComputationOperation_Add;
            else if (instructionUncoded.name == "sub")
                instructionCoded.operation = Enums::ComputationOperation_Sub;
            else if (instructionUncoded.name == "mul")
                instructionCoded.operation = Enums::ComputationOperation_Mul;
            else if (instructionUncoded.name == "div")
                instructionCoded.operation = Enums::ComputationOperation_Div;
            else if (instructionUncoded.name == "xor")
                instructionCoded.operation = Enums::ComputationOperation_Xor;
            else if (instructionUncoded.name == "or")
                instructionCoded.operation = Enums::ComputationOperation_Or;
            else if (instructionUncoded.name == "and")
                instructionCoded.operation = Enums::ComputationOperation_And;
            else if (instructionUncoded.name == "if") {
                if (instructionUncoded.comp == ">")
                    instructionCoded.operation = Enums::ComputationOperation_Ifg;
                else if ((instructionUncoded.comp == ">=") || (instructionUncoded.comp == "=>"))
                    instructionCoded.operation = Enums::ComputationOperation_Ifge;
                else if ((instructionUncoded.comp == "=") || (instructionUncoded.comp == "=="))
                    instructionCoded.operation = Enums::ComputationOperation_Ife;
                else if (instructionUncoded.comp == "!=")
                    instructionCoded.operation = Enums::ComputationOperation_Ifne;
                else if ((instructionUncoded.comp == "<=") || (instructionUncoded.comp == "=<"))
                    instructionCoded.operation = Enums::ComputationOperation_Ifle;
                else if (instructionUncoded.comp == "<")
                    instructionCoded.operation = Enums::ComputationOperation_Ifl;
                else {
                    return false;
                }
            } else if (instructionUncoded.name == "else")
                instructionCoded.operation = Enums::ComputationOperation_Else;
            else if (instructionUncoded.name == "endif")
                instructionCoded.operation = Enums::ComputationOperation_Endif;
            else {
                return false;
            }

            if (instructionCoded.operation != Enums::ComputationOperation_Else
                && instructionCoded.operation != Enums::ComputationOperation_Endif) {
                {
                    auto left1 = instructionUncoded.operand1.substr(0, 1);
                    auto left2 = instructionUncoded.operand1.substr(0, 2);
                    auto right1 = lastChar(instructionUncoded.operand1);
                    auto right2 = lastTwoChars(instructionUncoded.operand1);
                    if (left2 == "[[" && right2 == "]]") {
                        instructionCoded.opType1 = Enums::ComputationOpType_MemMem;
                        instructionUncoded.operand1 =
                            instructionUncoded.operand1.substr(2, instructionUncoded.operand1.size() - 4);
                    } else if (left1 == "[" && right1 == "]") {
                        instructionCoded.opType1 = Enums::ComputationOpType_Mem;
                        instructionUncoded.operand1 =
                            instructionUncoded.operand1.substr(1, instructionUncoded.operand1.size() - 2);
                    } else if (left1 == "(" && right1 == ")") {
                        instructionCoded.opType1 = Enums::ComputationOpType_Cmem;
                        instructionUncoded.operand1 =
                            instructionUncoded.operand1.substr(1, instructionUncoded.operand1.size() - 2);
                    } else {
                        return false;
                    }
                }
                {
                    auto left1 = instructionUncoded.operand2.substr(0, 1);
                    auto left2 = instructionUncoded.operand2.substr(0, 2);
                    auto right1 = lastChar(instructionUncoded.operand2);
                    auto right2 = lastTwoChars(instructionUncoded.operand2);
                    if (left2 == "[[" && right2 == "]]") {
                        instructionCoded.opType2 = Enums::ComputationOpType_MemMem;
                        instructionUncoded.operand2 =
                            instructionUncoded.operand2.substr(2, instructionUncoded.operand2.size() - 4);
                    } else if (left1 == "[" && right1 == "]") {
                        instructionCoded.opType2 = Enums::ComputationOpType_Mem;
                        instructionUncoded.operand2 =
                            instructionUncoded.operand2.substr(1, instructionUncoded.operand2.size() - 2);
                    } else if (left1 == "(" && right1 == ")") {
                        instructionCoded.opType2 = Enums::ComputationOpType_Cmem;
                        instructionUncoded.operand2 =
                            instructionUncoded.operand2.substr(1, instructionUncoded.operand2.size() - 2);
                    } else {
                        instructionCoded.opType2 = Enums::ComputationOpType_Constant;
                    }
                }
                {
                    auto left2 = instructionUncoded.operand1.substr(0, 2);
                    if (left2 == "0x") {
                        instructionCoded.operand1 =
                            static_cast<uint8_t>(std::stoul(instructionUncoded.operand1.substr(2), nullptr, 16));
                    } else {
                        instructionCoded.operand1 =
                            static_cast<uint8_t>(std::stoul(instructionUncoded.operand1, nullptr, 10));
                    }
                }
                {
                    auto left2 = instructionUncoded.operand2.substr(0, 2);
                    if (left2 == "0x") {
                        instructionCoded.operand2 =
                            static_cast<uint8_t>(std::stoul(instructionUncoded.operand2.substr(2), nullptr, 16));
                    } else {
                        instructionCoded.operand2 =
                            static_cast<uint8_t>(std::stoul(instructionUncoded.operand2, nullptr, 10));
                    }
                }
            } else {
                instructionCoded.operand1 = 0;
                instructionCoded.operand2 = 0;
            }
        } catch (...) {
            return false;
        }
        return true;
    }
}


CompilationResult CellComputationCompiler::compileSourceCode(std::string const& code, SymbolMap const& symbols, SimulationParameters const& parameters)
{
    CompilerState state = CompilerState::LOOKING_FOR_INSTR_START;

    CompilationResult result;
    result.compilation = std::string(getMaxCompiledCodeSize(parameters), 0);
    int linePos = 0;
    InstructionUncoded instructionUncoded;
    CellInstruction instructionCoded;
    for (int bytePos = 0; bytePos < code.length(); ++bytePos) {
        auto currentSymbol = code[bytePos];

        if (!gotoNextStateAndReturnSuccess(state, currentSymbol, instructionUncoded, bytePos, code.length())) {
            result.compilationOk = false;
            result.lineOfFirstError = linePos;
            return result;
        }
        if (instructionUncoded.readingFinished) {
            linePos++;
            if (!resolveInstructionAndReturnSuccess(symbols, instructionCoded, instructionUncoded)) {
                result.compilationOk = false;
                result.lineOfFirstError = linePos;
                return result;
            }
            writeInstruction(result.compilation, instructionCoded);
            state = CompilerState::LOOKING_FOR_INSTR_START;
            instructionUncoded = InstructionUncoded();
        }
    }
    if (state == CompilerState::LOOKING_FOR_INSTR_START && linePos - 1 > parameters.cellFunctionComputerMaxInstructions) {
        result.compilationOk = true;
    } else {
        result.compilationOk = false;
        result.lineOfFirstError = linePos;
    }
    return result;
}

namespace
{
    template<typename T>
    std::string toHexString(T value)
    {
        std::stringstream stream;
        stream << "0x" << std::hex << static_cast<int>(value);
        return stream.str();
    }
}

std::string CellComputationCompiler::decompileSourceCode(
    std::string const& data,
    SymbolMap const& symbols,
    SimulationParameters const& parameters)
{
    std::string text;
    std::string textOp1, textOp2;
    int nestingLevel = 0;
    int dataSize = data[0];
    auto isNullInstruction = [&data](int address) {
        return data[address + 1] == 0 && data[address + 2] == 0 && data[address + 3] == 0;
    };
    while (dataSize >= BytesPerInstruction) {
        if (isNullInstruction(dataSize - BytesPerInstruction)) {
            dataSize -= BytesPerInstruction;
        } else {
            break;
        }
    }

    for (int instructionPointer = 0; instructionPointer < dataSize;) {

        //decode instruction data
        CellInstruction instruction;
        readInstruction(data, instructionPointer, instruction);

        //write spacing
        for (int j = 0; j < nestingLevel; ++j)
            text += "  ";

        //write operation
        if (instruction.operation == Enums::ComputationOperation_Mov)
            text += "mov";
        if (instruction.operation == Enums::ComputationOperation_Add)
            text += "add";
        if (instruction.operation == Enums::ComputationOperation_Sub)
            text += "sub";
        if (instruction.operation == Enums::ComputationOperation_Mul)
            text += "mul";
        if (instruction.operation == Enums::ComputationOperation_Div)
            text += "div";
        if (instruction.operation == Enums::ComputationOperation_Xor)
            text += "xor";
        if (instruction.operation == Enums::ComputationOperation_Or)
            text += "or";
        if (instruction.operation == Enums::ComputationOperation_And)
            text += "and";
        if ((instruction.operation >= Enums::ComputationOperation_Ifg)
            && (instruction.operation <= Enums::ComputationOperation_Ifl)) {
            text += "if";
            ++nestingLevel;
        }
        if (instruction.operation == Enums::ComputationOperation_Else) {
            if (nestingLevel > 0)
                text = text.substr(0, text.size() - 2);
            text += "else";
        }
        if (instruction.operation == Enums::ComputationOperation_Endif) {
            if (nestingLevel > 0) {
                text = text.substr(0, text.size() - 2);
                --nestingLevel;
            }
            text += "endif";
        }

        //write operands
        if (instruction.opType1 == Enums::ComputationOpType_Mem)
            textOp1 = "[" + toHexString(convertToAddress(instruction.operand1, parameters.tokenMemorySize)) + "]";
        if (instruction.opType1 == Enums::ComputationOpType_MemMem)
            textOp1 = "[[" + toHexString(convertToAddress(instruction.operand1, parameters.tokenMemorySize)) + "]]";
        if (instruction.opType1 == Enums::ComputationOpType_Cmem)
            textOp1 = "("
                + toHexString(convertToAddress(instruction.operand1, parameters.cellFunctionComputerCellMemorySize))
                + ")";
        if (instruction.opType2 == Enums::ComputationOpType_Mem)
            textOp2 = "[" + toHexString(convertToAddress(instruction.operand2, parameters.tokenMemorySize)) + "]";
        if (instruction.opType2 == Enums::ComputationOpType_MemMem)
            textOp2 = "[[" + toHexString(convertToAddress(instruction.operand2, parameters.tokenMemorySize)) + "]]";
        if (instruction.opType2 == Enums::ComputationOpType_Cmem)
            textOp2 = "("
                + toHexString(convertToAddress(instruction.operand2, parameters.cellFunctionComputerCellMemorySize))
                + ")";
        if (instruction.opType2 == Enums::ComputationOpType_Constant)
            textOp2 = toHexString(convertToAddress(instruction.operand2, parameters.tokenMemorySize));

        //write separation/comparator
        if (instruction.operation <= Enums::ComputationOperation_And) {
            text += " " + textOp1 + ", " + textOp2;
        }
        if (instruction.operation == Enums::ComputationOperation_Ifg)
            text += " " + textOp1 + " > " + textOp2;
        if (instruction.operation == Enums::ComputationOperation_Ifge)
            text += " " + textOp1 + " >= " + textOp2;
        if (instruction.operation == Enums::ComputationOperation_Ife)
            text += " " + textOp1 + " = " + textOp2;
        if (instruction.operation == Enums::ComputationOperation_Ifne)
            text += " " + textOp1 + " != " + textOp2;
        if (instruction.operation == Enums::ComputationOperation_Ifle)
            text += " " + textOp1 + " <= " + textOp2;
        if (instruction.operation == Enums::ComputationOperation_Ifl)
            text += " " + textOp1 + " < " + textOp2;
        if (instructionPointer < dataSize)
            text += "\n";
    }
    return text;
}

std::optional<int> CellComputationCompiler::extractAddress(std::string const& s)
{
    try {
        auto left1 = s.substr(0, 1);
        auto left2 = s.substr(0, 2);
        auto right1 = lastChar(s);
        auto right2 = lastTwoChars(s);
        if (left1 == "[" && left1 != "[[" && right1 == "]" && right2 != "]]") {
            auto address = s.substr(1, s.size() - 2);
            if (address.substr(0, 2) == "0x") {
                return static_cast<int>(std::stoul(address.substr(2), nullptr, 16));
            } else {
                return static_cast<int>(std::stoul(address, nullptr, 10));
            }
        }
        return std::nullopt;
    }
    catch(...) {
        return std::nullopt;
    }
}

int CellComputationCompiler::getMaxCompiledCodeSize(SimulationParameters const& parameters)
{
    return parameters.cellFunctionComputerMaxInstructions * 3 + 1;
}

void CellComputationCompiler::writeInstruction(std::string& data, CellInstruction const& instructionCoded)
{
    //machine code: [INSTR - 4 Bits][MEM/MEMMEM/CMEM - 2 Bit][MEM/MEMMEM/CMEM/CONST - 2 Bit]
    auto currentInstructions = data[0];
    data[currentInstructions * BytesPerInstruction + 1] = (static_cast<uint8_t>(instructionCoded.operation) << 4)
        | (static_cast<uint8_t>(instructionCoded.opType1) << 2)
        | static_cast<uint8_t>(instructionCoded.opType2);
    data[currentInstructions * BytesPerInstruction + 2] = instructionCoded.operand1;
    data[currentInstructions * BytesPerInstruction + 3] = instructionCoded.operand2;
    ++data[0];
}

void CellComputationCompiler::readInstruction(
    std::string const& data,
    int& instructionPointer,
    CellInstruction& instructionCoded)
{
    //machine code: [INSTR - 4 Bits][MEM/ADDR/CMEM - 2 Bit][MEM/ADDR/CMEM/CONST - 2 Bit]
    instructionCoded.operation = static_cast<Enums::ComputationOperation>((data[instructionPointer + 1] >> 4) & 0xF);
    instructionCoded.opType1 = static_cast<Enums::ComputationOpType>(((data[instructionPointer + 1] >> 2) & 0x3) % 3);
    instructionCoded.opType2 = static_cast<Enums::ComputationOpType>(data[instructionPointer + 1] & 0x3);
    instructionCoded.operand1 = data[instructionPointer + 2];  //readInteger(_code,instructionPointer + 1);
    instructionCoded.operand2 = data[instructionPointer + 3];  //readInteger(_code,instructionPointer + 2);

    //increment instruction pointer
    instructionPointer += BytesPerInstruction;
}

uint8_t CellComputationCompiler::convertToAddress(int8_t addr, uint32_t size)
{
    uint32_t t = static_cast<uint32_t>(static_cast<uint8_t>(addr));
    return ((t % size) + size) % size;
}

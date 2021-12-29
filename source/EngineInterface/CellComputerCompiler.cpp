#include "CellComputerCompiler.h"

#include <sstream>
#include <boost/algorithm/string/case_conv.hpp>

#include "SymbolMap.h"
#include "SimulationParameters.h"

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
            auto last = s.substr(s.size() - 1, 1);
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
        InstructionCoded& instructionCoded,
        InstructionUncoded instructionUncoded)
    {
        try {
            auto lastChar = [](std::string const& s) {
                if (!s.empty()) {
                    return std::string(1, s.back());
                }
                return s;
            };
            auto lastTwoChars = [](std::string const& s) {
                if (s.size() > 1) {
                    return s.substr(s.size() - 2, 2);
                }
                return s;
            };
            boost::algorithm::to_lower(instructionUncoded.name);
            instructionUncoded.operand1 = applyTableToCode(symbols, instructionUncoded.operand1);
            instructionUncoded.operand2 = applyTableToCode(symbols, instructionUncoded.operand2);

            //prepare data for instruction coding
            if (instructionUncoded.name == "mov")
                instructionCoded.operation = Enums::ComputerOperation::MOV;
            else if (instructionUncoded.name == "add")
                instructionCoded.operation = Enums::ComputerOperation::ADD;
            else if (instructionUncoded.name == "sub")
                instructionCoded.operation = Enums::ComputerOperation::SUB;
            else if (instructionUncoded.name == "mul")
                instructionCoded.operation = Enums::ComputerOperation::MUL;
            else if (instructionUncoded.name == "div")
                instructionCoded.operation = Enums::ComputerOperation::DIV;
            else if (instructionUncoded.name == "xor")
                instructionCoded.operation = Enums::ComputerOperation::XOR;
            else if (instructionUncoded.name == "or")
                instructionCoded.operation = Enums::ComputerOperation::OR;
            else if (instructionUncoded.name == "and")
                instructionCoded.operation = Enums::ComputerOperation::AND;
            else if (instructionUncoded.name == "if") {
                if (instructionUncoded.comp == ">")
                    instructionCoded.operation = Enums::ComputerOperation::IFG;
                else if ((instructionUncoded.comp == ">=") || (instructionUncoded.comp == "=>"))
                    instructionCoded.operation = Enums::ComputerOperation::IFGE;
                else if ((instructionUncoded.comp == "=") || (instructionUncoded.comp == "=="))
                    instructionCoded.operation = Enums::ComputerOperation::IFE;
                else if (instructionUncoded.comp == "!=")
                    instructionCoded.operation = Enums::ComputerOperation::IFNE;
                else if ((instructionUncoded.comp == "<=") || (instructionUncoded.comp == "=<"))
                    instructionCoded.operation = Enums::ComputerOperation::IFLE;
                else if (instructionUncoded.comp == "<")
                    instructionCoded.operation = Enums::ComputerOperation::IFL;
                else {
                    return false;
                }
            } else if (instructionUncoded.name == "else")
                instructionCoded.operation = Enums::ComputerOperation::ELSE;
            else if (instructionUncoded.name == "endif")
                instructionCoded.operation = Enums::ComputerOperation::ENDIF;
            else {
                return false;
            }

            if (instructionCoded.operation != Enums::ComputerOperation::ELSE
                && instructionCoded.operation != Enums::ComputerOperation::ENDIF) {
                {
                    auto left1 = instructionUncoded.operand1.substr(0, 1);
                    auto left2 = instructionUncoded.operand1.substr(0, 2);
                    auto right1 = lastChar(instructionUncoded.operand1);
                    auto right2 = lastTwoChars(instructionUncoded.operand1);
                    if (left2 == "[[" && right2 == "]]") {
                        instructionCoded.opType1 = Enums::ComputerOptype::MEMMEM;
                        instructionUncoded.operand1 =
                            instructionUncoded.operand1.substr(2, instructionUncoded.operand1.size() - 4);
                    } else if (left1 == "[" && right1 == "]") {
                        instructionCoded.opType1 = Enums::ComputerOptype::MEM;
                        instructionUncoded.operand1 =
                            instructionUncoded.operand1.substr(1, instructionUncoded.operand1.size() - 2);
                    } else if (left1 == "(" && right1 == ")") {
                        instructionCoded.opType1 = Enums::ComputerOptype::CMEM;
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
                        instructionCoded.opType2 = Enums::ComputerOptype::MEMMEM;
                        instructionUncoded.operand2 =
                            instructionUncoded.operand2.substr(2, instructionUncoded.operand2.size() - 4);
                    } else if (left1 == "[" && right1 == "]") {
                        instructionCoded.opType2 = Enums::ComputerOptype::MEM;
                        instructionUncoded.operand2 =
                            instructionUncoded.operand2.substr(1, instructionUncoded.operand2.size() - 2);
                    } else if (left1 == "(" && right1 == ")") {
                        instructionCoded.opType2 = Enums::ComputerOptype::CMEM;
                        instructionUncoded.operand2 =
                            instructionUncoded.operand2.substr(1, instructionUncoded.operand2.size() - 2);
                    } else {
                        instructionCoded.opType2 = Enums::ComputerOptype::CONSTANT;
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


CompilationResult CellComputerCompiler::compileSourceCode(std::string const& code, SymbolMap const& symbols)
{
    CompilerState state = CompilerState::LOOKING_FOR_INSTR_START;

    CompilationResult result;
    int linePos = 0;
    InstructionUncoded instructionUncoded;
    InstructionCoded instructionCoded;
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
    if (state == CompilerState::LOOKING_FOR_INSTR_START)
        result.compilationOk = true;
    else {
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

std::string CellComputerCompiler::decompileSourceCode(
    std::string const& data,
    SymbolMap const& symbols,
    SimulationParameters const& parameters)
{
    std::string text;
    std::string textOp1, textOp2;
    int conditionLevel = 0;
    auto const dataSize = (data.size() / 3) * 3;
    for (int instructionPointer = 0; instructionPointer < dataSize;) {

        //decode instruction data
        InstructionCoded instruction;
        readInstruction(data, instructionPointer, instruction);

        //write spacing
        for (int j = 0; j < conditionLevel; ++j)
            text += "  ";

        //write operation
        if (instruction.operation == Enums::ComputerOperation::MOV)
            text += "mov";
        if (instruction.operation == Enums::ComputerOperation::ADD)
            text += "add";
        if (instruction.operation == Enums::ComputerOperation::SUB)
            text += "sub";
        if (instruction.operation == Enums::ComputerOperation::MUL)
            text += "mul";
        if (instruction.operation == Enums::ComputerOperation::DIV)
            text += "div";
        if (instruction.operation == Enums::ComputerOperation::XOR)
            text += "xor";
        if (instruction.operation == Enums::ComputerOperation::OR)
            text += "or";
        if (instruction.operation == Enums::ComputerOperation::AND)
            text += "and";
        if ((instruction.operation >= Enums::ComputerOperation::IFG)
            && (instruction.operation <= Enums::ComputerOperation::IFL)) {
            text += "if";
            ++conditionLevel;
        }
        if (instruction.operation == Enums::ComputerOperation::ELSE) {
            if (conditionLevel > 0)
                text = text.substr(0, text.size() - 2);
            text += "else";
        }
        if (instruction.operation == Enums::ComputerOperation::ENDIF) {
            if (conditionLevel > 0) {
                text = text.substr(0, text.size() - 2);
                --conditionLevel;
            }
            text += "endif";
        }

        //write operands
        if (instruction.opType1 == Enums::ComputerOptype::MEM)
            textOp1 = "[" + toHexString(convertToAddress(instruction.operand1, parameters.tokenMemorySize)) + "]";
        if (instruction.opType1 == Enums::ComputerOptype::MEMMEM)
            textOp1 = "[[" + toHexString(convertToAddress(instruction.operand1, parameters.tokenMemorySize)) + "]]";
        if (instruction.opType1 == Enums::ComputerOptype::CMEM)
            textOp1 = "("
                + toHexString(convertToAddress(instruction.operand1, parameters.cellFunctionComputerCellMemorySize))
                + ")";
        if (instruction.opType2 == Enums::ComputerOptype::MEM)
            textOp2 = "[" + toHexString(convertToAddress(instruction.operand2, parameters.tokenMemorySize)) + "]";
        if (instruction.opType2 == Enums::ComputerOptype::MEMMEM)
            textOp2 = "[[" + toHexString(convertToAddress(instruction.operand2, parameters.tokenMemorySize)) + "]]";
        if (instruction.opType2 == Enums::ComputerOptype::CMEM)
            textOp2 = "("
                + toHexString(convertToAddress(instruction.operand2, parameters.cellFunctionComputerCellMemorySize))
                + ")";
        if (instruction.opType2 == Enums::ComputerOptype::CONSTANT)
            textOp2 = toHexString(convertToAddress(instruction.operand2, parameters.tokenMemorySize));

        //write separation/comparator
        if (instruction.operation <= Enums::ComputerOperation::AND) {
            text += " " + textOp1 + ", " + textOp2;
        }
        if (instruction.operation == Enums::ComputerOperation::IFG)
            text += " " + textOp1 + " > " + textOp2;
        if (instruction.operation == Enums::ComputerOperation::IFGE)
            text += " " + textOp1 + " >= " + textOp2;
        if (instruction.operation == Enums::ComputerOperation::IFE)
            text += " " + textOp1 + " = " + textOp2;
        if (instruction.operation == Enums::ComputerOperation::IFNE)
            text += " " + textOp1 + " != " + textOp2;
        if (instruction.operation == Enums::ComputerOperation::IFLE)
            text += " " + textOp1 + " <= " + textOp2;
        if (instruction.operation == Enums::ComputerOperation::IFL)
            text += " " + textOp1 + " < " + textOp2;
        if (instructionPointer < dataSize)
            text += "\n";
    }
    return text;
}

void CellComputerCompiler::writeInstruction(std::string& data, InstructionCoded const& instructionCoded)
{
    //machine code: [INSTR - 4 Bits][MEM/MEMMEM/CMEM - 2 Bit][MEM/MEMMEM/CMEM/CONST - 2 Bit]
    data.push_back(
        (static_cast<uint8_t>(instructionCoded.operation) << 4) | (static_cast<uint8_t>(instructionCoded.opType1) << 2)
        | static_cast<uint8_t>(instructionCoded.opType2));
    data.push_back(instructionCoded.operand1);
    data.push_back(instructionCoded.operand2);
}

void CellComputerCompiler::readInstruction(
    std::string const& data,
    int& instructionPointer,
    InstructionCoded& instructionCoded)
{
    //machine code: [INSTR - 4 Bits][MEM/ADDR/CMEM - 2 Bit][MEM/ADDR/CMEM/CONST - 2 Bit]
    instructionCoded.operation = static_cast<Enums::ComputerOperation::Type>((data[instructionPointer] >> 4) & 0xF);
    instructionCoded.opType1 = static_cast<Enums::ComputerOptype::Type>(((data[instructionPointer] >> 2) & 0x3) % 3);
    instructionCoded.opType2 = static_cast<Enums::ComputerOptype::Type>(data[instructionPointer] & 0x3);
    instructionCoded.operand1 = data[instructionPointer + 1];  //readInteger(_code,instructionPointer + 1);
    instructionCoded.operand2 = data[instructionPointer + 2];  //readInteger(_code,instructionPointer + 2);

    //increment instruction pointer
    instructionPointer += 3;
}

uint8_t CellComputerCompiler::convertToAddress(int8_t addr, uint32_t size)
{
    uint32_t t = static_cast<uint32_t>(static_cast<uint8_t>(addr));
    return ((t % size) + size) % size;
}

#include <QString>
#include <qdebug.h>

#include "global/numbergenerator.h"
#include "model/context/simulationunitcontext.h"
#include "model/modelsettings.h"
#include "model/metadata/symboltable.h"
#include "model/entities/cell.h"
#include "model/entities/token.h"
#include "model/context/simulationparameters.h"

#include "cellfunctioncomputerimpl.h"

CellFunctionComputerImpl::CellFunctionComputerImpl (SimulationUnitContext* context)
    : CellFunctionComputer(context)
    , _memory(context->getSimulationParameters()->cellFunctionComputerCellMemorySize, 0)
	, _symbolTable(context->getSymbolTable())
	, _parameters(context->getSimulationParameters())
{
}

CellFunctionComputerImpl::CellFunctionComputerImpl (QByteArray data, SimulationUnitContext* context)
	: CellFunctionComputerImpl(context)
{
	if (!data.isEmpty()) {
		int numInstructions = data[0];
		int minSize = 3 * std::min(numInstructions, context->getSimulationParameters()->cellFunctionComputerMaxInstructions);
		_code = data.mid(1, minSize);
		if (_code.size() != minSize) {
			_code.clear();
		}
	}
}

namespace {
    quint8 convertToAddress (qint8 addr, quint32 size)
    {
        quint32 t((quint32)((quint8)addr));
        return ((t % size) + size) % size;
    }

    bool isNameChar (const QChar& c)
    {
        return c.isLetterOrNumber() || (c == ':');
    }
}

QString CellFunctionComputerImpl::decompileInstructionCode () const
{
    QString text;
    QString textOp1, textOp2;
    int conditionLevel = 0;
	for(int instructionPointer = 0; instructionPointer < _code.size(); ) {

        //decode instruction data
		InstructionCoded instruction;
        readInstruction(instructionPointer, instruction);

        //write spacing
        for(int j = 0; j < conditionLevel; ++j )
            text += "  ";

        //write operation
        if( instruction.operation == Enums::ComputerOperation::MOV)
            text += "mov";
        if( instruction.operation == Enums::ComputerOperation::ADD)
            text += "add";
        if( instruction.operation == Enums::ComputerOperation::SUB)
            text += "sub";
        if( instruction.operation == Enums::ComputerOperation::MUL)
            text += "mul";
        if( instruction.operation == Enums::ComputerOperation::DIV)
            text += "div";
        if( instruction.operation == Enums::ComputerOperation::XOR)
            text += "xor";
        if( instruction.operation == Enums::ComputerOperation::OR)
            text += "or";
        if( instruction.operation == Enums::ComputerOperation::AND)
            text += "and";
        if( (instruction.operation >= Enums::ComputerOperation::IFG) && (instruction.operation <= Enums::ComputerOperation::IFL) ) {
            text += "if";
            ++conditionLevel;
        }
        if( instruction.operation == Enums::ComputerOperation::ELSE) {
            if( conditionLevel > 0)
                text.chop(2);
            text += "else";
        }
        if( instruction.operation == Enums::ComputerOperation::ENDIF) {
            if( conditionLevel > 0) {
                text.chop(2);
                --conditionLevel;
            }
            text += "endif";
        }

        //write operands
        if( instruction.opType1 == Enums::ComputerOptype::MEM)
            textOp1 = "["+ QString("0x%1").arg(convertToAddress(instruction.operand1, _parameters->tokenMemorySize),0, 16)+"]";
        if( instruction.opType1 == Enums::ComputerOptype::MEMMEM)
            textOp1 = "[["+ QString("0x%1").arg(convertToAddress(instruction.operand1, _parameters->tokenMemorySize),0, 16)+"]]";
        if( instruction.opType1 == Enums::ComputerOptype::CMEM)
            textOp1 = "("+ QString("0x%1").arg(convertToAddress(instruction.operand1, _parameters->cellFunctionComputerCellMemorySize),0, 16)+")";
        if( instruction.opType2 == Enums::ComputerOptype::MEM)
            textOp2 = "["+ QString("0x%1").arg(convertToAddress(instruction.operand2, _parameters->tokenMemorySize),0, 16)+"]";
        if( instruction.opType2 == Enums::ComputerOptype::MEMMEM)
            textOp2 = "[["+ QString("0x%1").arg(convertToAddress(instruction.operand2, _parameters->tokenMemorySize),0, 16)+"]]";
        if( instruction.opType2 == Enums::ComputerOptype::CMEM)
            textOp2 = "("+ QString("0x%1").arg(convertToAddress(instruction.operand2, _parameters->cellFunctionComputerCellMemorySize),0, 16)+")";
        if( instruction.opType2 == Enums::ComputerOptype::CONST)
            textOp2 = QString("0x%1").arg(convertToAddress(instruction.operand2, _parameters->tokenMemorySize),0, 16);

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
        if (instructionPointer < _code.size())
            text += "\n";
    }
    return text;
}

CellFunctionComputer::CompilationState CellFunctionComputerImpl::injectAndCompileInstructionCode (QString sourceCode)
{
    State state = State::LOOKING_FOR_INSTR_START;

	_code.clear();
    int linePos = 0;
	InstructionUncoded instructionUncoded;
	InstructionCoded instructionCoded;
	for (int bytePos = 0; bytePos < sourceCode.length(); ++bytePos) {
        QChar currentSymbol = sourceCode[bytePos];

		if (!gotoNextStateAndReturnSuccess(state, currentSymbol, instructionUncoded, bytePos, sourceCode.length())) {
			return{ false, linePos };
		}
        if( instructionUncoded.readingFinished ) {
			linePos++;
			if (!resolveInstructionAndReturnSuccess(instructionCoded, instructionUncoded)) {
				return{ false, linePos };
			}
            writeInstruction(instructionCoded);
            state = State::LOOKING_FOR_INSTR_START;
			instructionUncoded = InstructionUncoded();
        }
    }
    if( state == State::LOOKING_FOR_INSTR_START )
        return {true, linePos};
    else {
        return {false, linePos};
    }
}

bool CellFunctionComputerImpl::resolveInstructionAndReturnSuccess(InstructionCoded& instructionCoded, InstructionUncoded instructionUncoded)
{
	instructionUncoded.operand1 = applyTableToCode(instructionUncoded.operand1);
	instructionUncoded.operand2 = applyTableToCode(instructionUncoded.operand2);

	//prepare data for instruction coding
	if (instructionUncoded.name.toLower() == "mov")
		instructionCoded.operation = Enums::ComputerOperation::MOV;
	else if (instructionUncoded.name.toLower() == "add")
		instructionCoded.operation = Enums::ComputerOperation::ADD;
	else if (instructionUncoded.name.toLower() == "sub")
		instructionCoded.operation = Enums::ComputerOperation::SUB;
	else if (instructionUncoded.name.toLower() == "mul")
		instructionCoded.operation = Enums::ComputerOperation::MUL;
	else if (instructionUncoded.name.toLower() == "div")
		instructionCoded.operation = Enums::ComputerOperation::DIV;
	else if (instructionUncoded.name.toLower() == "xor")
		instructionCoded.operation = Enums::ComputerOperation::XOR;
	else if (instructionUncoded.name.toLower() == "or")
		instructionCoded.operation = Enums::ComputerOperation::OR;
	else if (instructionUncoded.name.toLower() == "and")
		instructionCoded.operation = Enums::ComputerOperation::AND;
	else if (instructionUncoded.name.toLower() == "if") {
		if (instructionUncoded.comp.toLower() == ">")
			instructionCoded.operation = Enums::ComputerOperation::IFG;
		else if ((instructionUncoded.comp.toLower() == ">=") || (instructionUncoded.comp.toLower() == "=>"))
			instructionCoded.operation = Enums::ComputerOperation::IFGE;
		else if ((instructionUncoded.comp.toLower() == "=") || (instructionUncoded.comp.toLower() == "=="))
			instructionCoded.operation = Enums::ComputerOperation::IFE;
		else if (instructionUncoded.comp.toLower() == "!=")
			instructionCoded.operation = Enums::ComputerOperation::IFNE;
		else if ((instructionUncoded.comp.toLower() == "<=") || (instructionUncoded.comp.toLower() == "=<"))
			instructionCoded.operation = Enums::ComputerOperation::IFLE;
		else if (instructionUncoded.comp.toLower() == "<")
			instructionCoded.operation = Enums::ComputerOperation::IFL;
		else {
			return false;
		}
	}
	else if (instructionUncoded.name.toLower() == "else")
		instructionCoded.operation = Enums::ComputerOperation::ELSE;
	else if (instructionUncoded.name.toLower() == "endif")
		instructionCoded.operation = Enums::ComputerOperation::ENDIF;
	else {
		return false;
	}

	if (instructionCoded.operation != Enums::ComputerOperation::ELSE && instructionCoded.operation != Enums::ComputerOperation::ENDIF) {
		if ((instructionUncoded.operand1.left(2) == "[[") && (instructionUncoded.operand1.right(2) == "]]")) {
			instructionCoded.opType1 =  Enums::ComputerOptype::MEMMEM;
			instructionUncoded.operand1 = instructionUncoded.operand1.remove(0, 2);
			instructionUncoded.operand1.chop(2);
		}
		else if ((instructionUncoded.operand1.left(1) == "[") && (instructionUncoded.operand1.right(1) == "]")) {
			instructionCoded.opType1 = Enums::ComputerOptype::MEM;
			instructionUncoded.operand1 = instructionUncoded.operand1.remove(0, 1);
			instructionUncoded.operand1.chop(1);
		}
		else if ((instructionUncoded.operand1.left(1) == "(") && (instructionUncoded.operand1.right(1) == ")")) {
			instructionCoded.opType1 = Enums::ComputerOptype::CMEM;
			instructionUncoded.operand1 = instructionUncoded.operand1.remove(0, 1);
			instructionUncoded.operand1.chop(1);
		}
		else {
			return false;
		}

		if ((instructionUncoded.operand2.left(2) == "[[") && (instructionUncoded.operand2.right(2) == "]]")) {
			instructionCoded.opType2 = Enums::ComputerOptype::MEMMEM;
			instructionUncoded.operand2 = instructionUncoded.operand2.remove(0, 2);
			instructionUncoded.operand2.chop(2);
		}
		else if ((instructionUncoded.operand2.left(1) == "[") && (instructionUncoded.operand2.right(1) == "]")) {
			instructionCoded.opType2 = Enums::ComputerOptype::MEM;
			instructionUncoded.operand2 = instructionUncoded.operand2.remove(0, 1);
			instructionUncoded.operand2.chop(1);
		}
		else if ((instructionUncoded.operand2.left(1) == "(") && (instructionUncoded.operand2.right(1) == ")")) {
			instructionCoded.opType2 = Enums::ComputerOptype::CMEM;
			instructionUncoded.operand2 = instructionUncoded.operand2.remove(0, 1);
			instructionUncoded.operand2.chop(1);
		}
		else
			instructionCoded.opType2 = Enums::ComputerOptype::CONST;


		if (instructionUncoded.operand1.left(2) == "0x") {
			bool ok(true);
			instructionCoded.operand1 = instructionUncoded.operand1.remove(0, 2).toInt(&ok, 16);
			if (!ok) {
				return false;
			}
		}
		else {
			bool ok(true);
			instructionCoded.operand1 = instructionUncoded.operand1.toInt(&ok, 10);
			if (!ok)
				return false;
		}
		if (instructionUncoded.operand2.left(2) == "0x") {
			bool ok(true);
			instructionCoded.operand2 = instructionUncoded.operand2.remove(0, 2).toInt(&ok, 16);
			if (!ok) {
				return false;
			}
		}
		else {
			bool ok(true);
			instructionCoded.operand2 = instructionUncoded.operand2.toInt(&ok, 10);
			if (!ok) {
				return false;
			}
		}
	}
	else {
		instructionCoded.operand1 = 0;
		instructionCoded.operand2 = 0;
	}
	return true;
}

bool CellFunctionComputerImpl::gotoNextStateAndReturnSuccess(State &state, QChar &currentSymbol, InstructionUncoded& instruction
	, int bytePos, int codeSize)
{
	switch (state) {
		case State::LOOKING_FOR_INSTR_START: {
			if (currentSymbol.isLetter()) {
				state = State::LOOKING_FOR_INSTR_END;
				instruction.name = currentSymbol;
			}
		}
		break;
		case State::LOOKING_FOR_INSTR_END: {
			if (!currentSymbol.isLetter()) {
				if ((instruction.name.toLower() == "else") || (instruction.name.toLower() == "endif"))
					instruction.readingFinished = true;
				else
					state = State::LOOKING_FOR_OP1_START;
			}
			else {
				instruction.name += currentSymbol;
				if ((bytePos + 1) == codeSize && ((instruction.name.toLower() == "else") || (instruction.name.toLower() == "endif")))
					instruction.readingFinished = true;
			}
		}
		break;
		case State::LOOKING_FOR_OP1_START: {
			if (isNameChar(currentSymbol) || (currentSymbol == '-') || (currentSymbol == '_') || (currentSymbol == '[') || (currentSymbol == '(')) {
				state = State::LOOKING_FOR_OP1_END;
				instruction.operand1 = currentSymbol;
			}
		}
		break;
		case State::LOOKING_FOR_OP1_END: {
			if ((currentSymbol == '<') || (currentSymbol == '>') || (currentSymbol == '=') || (currentSymbol == '!')) {
				state = State::LOOKING_FOR_COMPARATOR;
				instruction.comp = currentSymbol;
			}
			else if (currentSymbol == ',')
				state = State::LOOKING_FOR_OP2_START;
			else if (!isNameChar(currentSymbol) && (currentSymbol != '-') && (currentSymbol != '_') && (currentSymbol != '[') && (currentSymbol != ']') && (currentSymbol != '(') && (currentSymbol != ')'))
				state = State::LOOKING_FOR_SEPARATOR;
			else
				instruction.operand1 += currentSymbol;
		}
		break;
		case State::LOOKING_FOR_SEPARATOR: {
			if (currentSymbol == ',')
				state = State::LOOKING_FOR_OP2_START;
			else if ((currentSymbol == '<') || (currentSymbol == '>') || (currentSymbol == '=') || (currentSymbol == '!')) {
				state = State::LOOKING_FOR_COMPARATOR;
				instruction.comp = currentSymbol;
			}
			else if (isNameChar(currentSymbol) || (currentSymbol == '-') || (currentSymbol == '_') || (currentSymbol == '[') || (currentSymbol == ']') || (currentSymbol == '(') || (currentSymbol == ')'))
				return false;
		}
		break;
		case State::LOOKING_FOR_COMPARATOR: {
			if ((currentSymbol == '<') || (currentSymbol == '>') || (currentSymbol == '=') || (currentSymbol == '!'))
				instruction.comp += currentSymbol;
			else if (!isNameChar(currentSymbol) && (currentSymbol != '-') && (currentSymbol != '_') && (currentSymbol != '[') && (currentSymbol != '('))
				state = State::LOOKING_FOR_OP2_START;
			else {
				state = State::LOOKING_FOR_OP2_END;
				instruction.operand2 = currentSymbol;
			}
		}
		break;
		case State::LOOKING_FOR_OP2_START: {
			if (isNameChar(currentSymbol) || (currentSymbol == '-') || (currentSymbol == '_') || (currentSymbol == '[') || (currentSymbol == '(')) {
				state = State::LOOKING_FOR_OP2_END;
				instruction.operand2 = currentSymbol;
				if (bytePos == (codeSize - 1))
					instruction.readingFinished = true;
			}
		}
		break;
		case State::LOOKING_FOR_OP2_END: {
			if (!isNameChar(currentSymbol) && (currentSymbol != '-') && (currentSymbol != '_') && (currentSymbol != '[') && (currentSymbol != ']') && (currentSymbol != '(') && (currentSymbol != ')'))
				instruction.readingFinished = true;
			else {
				instruction.operand2 += currentSymbol;
				if ((bytePos + 1) == codeSize)
					instruction.readingFinished = true;
			}
		}
		break;
	}
	if ((currentSymbol == '\n') || ((bytePos + 1) == codeSize)) {
		if (!instruction.name.isEmpty()) {
			instruction.readingFinished = true;
		}
	}
	return true;
}

QByteArray& CellFunctionComputerImpl::getMemoryReference ()
{
    return _memory;
}

void CellFunctionComputerImpl::mutateImpl()
{
	auto& generator = NumberGenerator::getInstance();
	qint8 randomByte = static_cast<qint8>(generator.random(256));
	if (generator.random(2) == 0) {
		if (!_code.isEmpty()) {
			_code[generator.random(_code.size())] = randomByte;
		}
	}
	else {
		if (!_memory.isEmpty()) {
			_memory[generator.random(_memory.size())] = randomByte;
		}
	}
}

namespace
{
	enum class MemoryType {
		TOKEN, CELL
	};

	qint8 getMemoryByte(QByteArray const& tokenMemory, QByteArray const& cellMemory, quint8 pointer, MemoryType type)
	{
		if (type == MemoryType::TOKEN) {
			return tokenMemory[pointer];
		}
		if (type == MemoryType::CELL) {
			return cellMemory[pointer];
		}
		return tokenMemory[pointer];
	}

	void setMemoryByte(QByteArray& tokenMemory, QByteArray& cellMemory, quint8 pointer, qint8 value, MemoryType type)
	{
		if (type == MemoryType::TOKEN) {
			tokenMemory[pointer] = value;
		}
		if (type == MemoryType::CELL) {
			cellMemory[pointer] = value;
		}
	}
}

CellFeature::ProcessingResult CellFunctionComputerImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};

    std::vector<bool> condTable(_parameters->cellFunctionComputerMaxInstructions);
    int condPointer(0);
    int bytePos = 0;
    while( bytePos < _code.size() ) {

        //decode instruction
		InstructionCoded instruction;
        readInstruction(bytePos, instruction);

        //operand 1: pointer to mem
        quint8 opPointer1 = 0;
		MemoryType memType = MemoryType::TOKEN;
        if (instruction.opType1 == Enums::ComputerOptype::MEM)
            opPointer1 = convertToAddress(instruction.operand1, _parameters->tokenMemorySize);
        if (instruction.opType1 == Enums::ComputerOptype::MEMMEM) {
            instruction.operand1 = token->getMemoryRef()[convertToAddress(instruction.operand1, _parameters->tokenMemorySize)];
            opPointer1 = convertToAddress(instruction.operand1, _parameters->tokenMemorySize);
        }
		if (instruction.opType1 == Enums::ComputerOptype::CMEM) {
			opPointer1 = convertToAddress(instruction.operand1, _parameters->cellFunctionComputerCellMemorySize);
			memType = MemoryType::CELL;
		}

        //operand 2: loading value
        if (instruction.opType2 == Enums::ComputerOptype::MEM)
            instruction.operand2 = token->getMemoryRef()[convertToAddress(instruction.operand2, _parameters->tokenMemorySize)];
        if (instruction.opType2 == Enums::ComputerOptype::MEMMEM) {
            instruction.operand2 = token->getMemoryRef()[convertToAddress(instruction.operand2, _parameters->tokenMemorySize)];
            instruction.operand2 = token->getMemoryRef()[convertToAddress(instruction.operand2, _parameters->tokenMemorySize)];
        }
        if (instruction.opType2 == Enums::ComputerOptype::CMEM)
            instruction.operand2 = _memory[convertToAddress(instruction.operand2, _parameters->cellFunctionComputerCellMemorySize)];

        //execute instruction
        bool execute = true;
        for(int k = 0; k < condPointer; ++k)
            if( !condTable[k] )
                execute = false;
        if( execute ) {
			if (instruction.operation == Enums::ComputerOperation::MOV)
				setMemoryByte(token->getMemoryRef(), _memory, opPointer1, instruction.operand2, memType);
            if (instruction.operation == Enums::ComputerOperation::ADD)
				setMemoryByte(token->getMemoryRef(), _memory, opPointer1, getMemoryByte(token->getMemoryRef(), _memory, opPointer1, memType) + instruction.operand2, memType);
            if (instruction.operation == Enums::ComputerOperation::SUB)
				setMemoryByte(token->getMemoryRef(), _memory, opPointer1, getMemoryByte(token->getMemoryRef(), _memory, opPointer1, memType) - instruction.operand2, memType);
            if (instruction.operation == Enums::ComputerOperation::MUL)
				setMemoryByte(token->getMemoryRef(), _memory, opPointer1, getMemoryByte(token->getMemoryRef(), _memory, opPointer1, memType) * instruction.operand2, memType);
            if (instruction.operation == Enums::ComputerOperation::DIV) {
                if( instruction.operand2 > 0)
					setMemoryByte(token->getMemoryRef(), _memory, opPointer1, getMemoryByte(token->getMemoryRef(), _memory, opPointer1, memType) / instruction.operand2, memType);
                else
					setMemoryByte(token->getMemoryRef(), _memory, opPointer1, 0, memType);
            }
            if (instruction.operation == Enums::ComputerOperation::XOR)
				setMemoryByte(token->getMemoryRef(), _memory, opPointer1, getMemoryByte(token->getMemoryRef(), _memory, opPointer1, memType) ^ instruction.operand2, memType);
            if (instruction.operation == Enums::ComputerOperation::OR)
				setMemoryByte(token->getMemoryRef(), _memory, opPointer1, getMemoryByte(token->getMemoryRef(), _memory, opPointer1, memType) | instruction.operand2, memType);
            if (instruction.operation == Enums::ComputerOperation::AND)
				setMemoryByte(token->getMemoryRef(), _memory, opPointer1, getMemoryByte(token->getMemoryRef(), _memory, opPointer1, memType) & instruction.operand2, memType);
        }

        //if instructions
		instruction.operand1 = getMemoryByte(token->getMemoryRef(), _memory, opPointer1, memType);
        if (instruction.operation == Enums::ComputerOperation::IFG) {
            if (instruction.operand1 > instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == Enums::ComputerOperation::IFGE) {
            if (instruction.operand1 >= instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == Enums::ComputerOperation::IFE) {
            if (instruction.operand1 == instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == Enums::ComputerOperation::IFNE) {
            if (instruction.operand1 != instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == Enums::ComputerOperation::IFLE) {
            if (instruction.operand1 <= instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == Enums::ComputerOperation::IFL) {
            if (instruction.operand1 < instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }

        if (instruction.operation == Enums::ComputerOperation::ELSE) {
            if( condPointer > 0 )
                condTable[condPointer-1] = !condTable[condPointer-1];
        }

        if (instruction.operation == Enums::ComputerOperation::ENDIF) {
            if( condPointer > 0 )
                condPointer--;
        }
    }
    return processingResult;
}

void CellFunctionComputerImpl::serializePrimitives (QDataStream& stream) const
{
    stream << _memory << _code;
}

void CellFunctionComputerImpl::deserializePrimitives (QDataStream& stream)
{
    //load remaining attributes
    stream >> _memory >> _code;
	_memory = _memory.left(_parameters->cellFunctionComputerCellMemorySize);
	_memory.resize(_parameters->cellFunctionComputerCellMemorySize);
	_code = _code.left(3 * _parameters->cellFunctionComputerMaxInstructions);
}

QByteArray CellFunctionComputerImpl::getInternalData () const
{
	QByteArray data;
	data.push_back(_code.size() / 3);
	data.push_back(_code);
	return data;
}

void CellFunctionComputerImpl::writeInstruction (InstructionCoded const& instructionCoded)
{
    //machine code: [INSTR - 4 Bits][MEM/MEMMEM/CMEM - 2 Bit][MEM/MEMMEM/CMEM/CONST - 2 Bit]
    _code.push_back((static_cast<quint8>(instructionCoded.operation) << 4)
		| (static_cast<quint8>(instructionCoded.opType1) << 2) | static_cast<quint8>(instructionCoded.opType2));
    _code.push_back(instructionCoded.operand1);
    _code.push_back(instructionCoded.operand2);
}

void CellFunctionComputerImpl::readInstruction (int& instructionPointer, InstructionCoded& instructionCoded) const
{
    //machine code: [INSTR - 4 Bits][MEM/ADDR/CMEM - 2 Bit][MEM/ADDR/CMEM/CONST - 2 Bit]
	instructionCoded.operation = static_cast<Enums::ComputerOperation::Type>((_code[instructionPointer] >> 4) & 0xF);
	instructionCoded.opType1 = static_cast<Enums::ComputerOptype::Type>(((_code[instructionPointer] >> 2) & 0x3) % 3);
	instructionCoded.opType2 = static_cast<Enums::ComputerOptype::Type>(_code[instructionPointer] & 0x3);
	instructionCoded.operand1 = _code[instructionPointer+1];//readInteger(_code,instructionPointer + 1);
	instructionCoded.operand2 = _code[instructionPointer+2];//readInteger(_code,instructionPointer + 2);

    //increment instruction pointer
    instructionPointer += 3;
}

QString CellFunctionComputerImpl::applyTableToCode(QString s)
{
	QString prefix;
	QString postfix;
	for (int i = 0; i < 2; ++i) {
		if (s.left(1) == "[" || s.left(1) == "(") {
			prefix = prefix + s.left(1);
			s = s.mid(1);
		}
	}
	for (int i = 0; i < 2; ++i) {
		if (s.right(1) == "]" || s.right(1) == ")") {
			postfix = s.right(1) + postfix;
			s.chop(1);
		}
	}
	s = _symbolTable->applyTableToCode(s);
	return prefix + s + postfix;
}

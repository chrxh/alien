#include <QString>
#include <qdebug.h>

#include "model/simulationcontext.h"
#include "model/config.h"
#include "model/metadata/symboltable.h"
#include "model/entities/cell.h"
#include "model/entities/token.h"
#include "model/simulationparameters.h"

#include "cellfunctioncomputerimpl.h"

CellFunctionComputerImpl::CellFunctionComputerImpl (SimulationContext* context)
    : CellFunctionComputer(context)
    , _memory(context->getSimulationParameters()->CELL_MEMSIZE, 0)
	, _symbolTable(context->getSymbolTable())
	, _parameters(context->getSimulationParameters())
{
}

CellFunctionComputerImpl::CellFunctionComputerImpl (QByteArray data, SimulationContext* context)
	: CellFunctionComputerImpl(context)
{
	if (!data.isEmpty()) {
		int numInstructions = data[0];
		int minSize = 3 * std::min(numInstructions, context->getSimulationParameters()->CELL_NUM_INSTR);
		_code = data.left(minSize);
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
        if( instruction.operation == COMPUTER_OPERATION::MOV)
            text += "mov";
        if( instruction.operation == COMPUTER_OPERATION::ADD)
            text += "add";
        if( instruction.operation == COMPUTER_OPERATION::SUB)
            text += "sub";
        if( instruction.operation == COMPUTER_OPERATION::MUL)
            text += "mul";
        if( instruction.operation == COMPUTER_OPERATION::DIV)
            text += "div";
        if( instruction.operation == COMPUTER_OPERATION::XOR)
            text += "xor";
        if( instruction.operation == COMPUTER_OPERATION::OR)
            text += "or";
        if( instruction.operation == COMPUTER_OPERATION::AND)
            text += "and";
        if( (instruction.operation >= COMPUTER_OPERATION::IFG) && (instruction.operation <= COMPUTER_OPERATION::IFL) ) {
            text += "if";
            ++conditionLevel;
        }
        if( instruction.operation == COMPUTER_OPERATION::ELSE) {
            if( conditionLevel > 0)
                text.chop(2);
            text += "else";
        }
        if( instruction.operation == COMPUTER_OPERATION::ENDIF) {
            if( conditionLevel > 0) {
                text.chop(2);
                --conditionLevel;
            }
            text += "endif";
        }

        //write operands
        if( instruction.opType1 == COMPUTER_OPTYPE::MEM)
            textOp1 = "["+ QString("0x%1").arg(convertToAddress(instruction.operand1, _parameters->TOKEN_MEMSIZE),0, 16)+"]";
        if( instruction.opType1 == COMPUTER_OPTYPE::MEMMEM)
            textOp1 = "[["+ QString("0x%1").arg(convertToAddress(instruction.operand1, _parameters->TOKEN_MEMSIZE),0, 16)+"]]";
        if( instruction.opType1 == COMPUTER_OPTYPE::CMEM)
            textOp1 = "("+ QString("0x%1").arg(convertToAddress(instruction.operand1, _parameters->CELL_MEMSIZE),0, 16)+")";
        if( instruction.opType2 == COMPUTER_OPTYPE::MEM)
            textOp2 = "["+ QString("0x%1").arg(convertToAddress(instruction.operand2, _parameters->TOKEN_MEMSIZE),0, 16)+"]";
        if( instruction.opType2 == COMPUTER_OPTYPE::MEMMEM)
            textOp2 = "[["+ QString("0x%1").arg(convertToAddress(instruction.operand2, _parameters->TOKEN_MEMSIZE),0, 16)+"]]";
        if( instruction.opType2 == COMPUTER_OPTYPE::CMEM)
            textOp2 = "("+ QString("0x%1").arg(convertToAddress(instruction.operand2, _parameters->CELL_MEMSIZE),0, 16)+")";
        if( instruction.opType2 == COMPUTER_OPTYPE::CONST)
            textOp2 = QString("0x%1").arg(convertToAddress(instruction.operand2, _parameters->TOKEN_MEMSIZE),0, 16);

        //write separation/comparator
        if (instruction.operation <= COMPUTER_OPERATION::AND) {
            text += " " + textOp1 + ", " + textOp2;
        }
        if (instruction.operation == COMPUTER_OPERATION::IFG)
            text += " " + textOp1 + " > " + textOp2;
        if (instruction.operation == COMPUTER_OPERATION::IFGE)
            text += " " + textOp1 + " >= " + textOp2;
        if (instruction.operation == COMPUTER_OPERATION::IFE)
            text += " " + textOp1 + " = " + textOp2;
        if (instruction.operation == COMPUTER_OPERATION::IFNE)
            text += " " + textOp1 + " != " + textOp2;
        if (instruction.operation == COMPUTER_OPERATION::IFLE)
            text += " " + textOp1 + " <= " + textOp2;
        if (instruction.operation == COMPUTER_OPERATION::IFL)
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
		instructionCoded.operation = COMPUTER_OPERATION::MOV;
	else if (instructionUncoded.name.toLower() == "add")
		instructionCoded.operation = COMPUTER_OPERATION::ADD;
	else if (instructionUncoded.name.toLower() == "sub")
		instructionCoded.operation = COMPUTER_OPERATION::SUB;
	else if (instructionUncoded.name.toLower() == "mul")
		instructionCoded.operation = COMPUTER_OPERATION::MUL;
	else if (instructionUncoded.name.toLower() == "div")
		instructionCoded.operation = COMPUTER_OPERATION::DIV;
	else if (instructionUncoded.name.toLower() == "xor")
		instructionCoded.operation = COMPUTER_OPERATION::XOR;
	else if (instructionUncoded.name.toLower() == "or")
		instructionCoded.operation = COMPUTER_OPERATION::OR;
	else if (instructionUncoded.name.toLower() == "and")
		instructionCoded.operation = COMPUTER_OPERATION::AND;
	else if (instructionUncoded.name.toLower() == "if") {
		if (instructionUncoded.comp.toLower() == ">")
			instructionCoded.operation = COMPUTER_OPERATION::IFG;
		else if ((instructionUncoded.comp.toLower() == ">=") || (instructionUncoded.comp.toLower() == "=>"))
			instructionCoded.operation = COMPUTER_OPERATION::IFGE;
		else if ((instructionUncoded.comp.toLower() == "=") || (instructionUncoded.comp.toLower() == "=="))
			instructionCoded.operation = COMPUTER_OPERATION::IFE;
		else if (instructionUncoded.comp.toLower() == "!=")
			instructionCoded.operation = COMPUTER_OPERATION::IFNE;
		else if ((instructionUncoded.comp.toLower() == "<=") || (instructionUncoded.comp.toLower() == "=<"))
			instructionCoded.operation = COMPUTER_OPERATION::IFLE;
		else if (instructionUncoded.comp.toLower() == "<")
			instructionCoded.operation = COMPUTER_OPERATION::IFL;
		else {
			return false;
		}
	}
	else if (instructionUncoded.name.toLower() == "else")
		instructionCoded.operation = COMPUTER_OPERATION::ELSE;
	else if (instructionUncoded.name.toLower() == "endif")
		instructionCoded.operation = COMPUTER_OPERATION::ENDIF;
	else {
		return false;
	}

	if (instructionCoded.operation != COMPUTER_OPERATION::ELSE && instructionCoded.operation != COMPUTER_OPERATION::ENDIF) {
		if ((instructionUncoded.operand1.left(2) == "[[") && (instructionUncoded.operand1.right(2) == "]]")) {
			instructionCoded.opType1 =  COMPUTER_OPTYPE::MEMMEM;
			instructionUncoded.operand1 = instructionUncoded.operand1.remove(0, 2);
			instructionUncoded.operand1.chop(2);
		}
		else if ((instructionUncoded.operand1.left(1) == "[") && (instructionUncoded.operand1.right(1) == "]")) {
			instructionCoded.opType1 = COMPUTER_OPTYPE::MEM;
			instructionUncoded.operand1 = instructionUncoded.operand1.remove(0, 1);
			instructionUncoded.operand1.chop(1);
		}
		else if ((instructionUncoded.operand1.left(1) == "(") && (instructionUncoded.operand1.right(1) == ")")) {
			instructionCoded.opType1 = COMPUTER_OPTYPE::CMEM;
			instructionUncoded.operand1 = instructionUncoded.operand1.remove(0, 1);
			instructionUncoded.operand1.chop(1);
		}
		else {
			return false;
		}

		if ((instructionUncoded.operand2.left(2) == "[[") && (instructionUncoded.operand2.right(2) == "]]")) {
			instructionCoded.opType2 = COMPUTER_OPTYPE::MEMMEM;
			instructionUncoded.operand2 = instructionUncoded.operand2.remove(0, 2);
			instructionUncoded.operand2.chop(2);
		}
		else if ((instructionUncoded.operand2.left(1) == "[") && (instructionUncoded.operand2.right(1) == "]")) {
			instructionCoded.opType2 = COMPUTER_OPTYPE::MEM;
			instructionUncoded.operand2 = instructionUncoded.operand2.remove(0, 1);
			instructionUncoded.operand2.chop(1);
		}
		else if ((instructionUncoded.operand2.left(1) == "(") && (instructionUncoded.operand2.right(1) == ")")) {
			instructionCoded.opType2 = COMPUTER_OPTYPE::CMEM;
			instructionUncoded.operand2 = instructionUncoded.operand2.remove(0, 1);
			instructionUncoded.operand2.chop(1);
		}
		else
			instructionCoded.opType2 = COMPUTER_OPTYPE::CONST;


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

    std::vector<bool> condTable(_parameters->CELL_NUM_INSTR);
    int condPointer(0);
    int bytePos = 0;
    while( bytePos < _code.size() ) {

        //decode instruction
		InstructionCoded instruction;
        readInstruction(bytePos, instruction);

        //operand 1: pointer to mem
        quint8 opPointer1 = 0;
		MemoryType memType = MemoryType::TOKEN;
        if (instruction.opType1 == COMPUTER_OPTYPE::MEM)
            opPointer1 = convertToAddress(instruction.operand1, _parameters->TOKEN_MEMSIZE);
        if (instruction.opType1 == COMPUTER_OPTYPE::MEMMEM) {
            instruction.operand1 = token->memory[convertToAddress(instruction.operand1, _parameters->TOKEN_MEMSIZE)];
            opPointer1 = convertToAddress(instruction.operand1, _parameters->TOKEN_MEMSIZE);
        }
		if (instruction.opType1 == COMPUTER_OPTYPE::CMEM) {
			opPointer1 = convertToAddress(instruction.operand1, _parameters->CELL_MEMSIZE);
			memType = MemoryType::CELL;
		}

        //operand 2: loading value
        if (instruction.opType2 == COMPUTER_OPTYPE::MEM)
            instruction.operand2 = token->memory[convertToAddress(instruction.operand2, _parameters->TOKEN_MEMSIZE)];
        if (instruction.opType2 == COMPUTER_OPTYPE::MEMMEM) {
            instruction.operand2 = token->memory[convertToAddress(instruction.operand2, _parameters->TOKEN_MEMSIZE)];
            instruction.operand2 = token->memory[convertToAddress(instruction.operand2, _parameters->TOKEN_MEMSIZE)];
        }
        if (instruction.opType2 == COMPUTER_OPTYPE::CMEM)
            instruction.operand2 = _memory[convertToAddress(instruction.operand2, _parameters->CELL_MEMSIZE)];

        //execute instruction
        bool execute = true;
        for(int k = 0; k < condPointer; ++k)
            if( !condTable[k] )
                execute = false;
        if( execute ) {
			if (instruction.operation == COMPUTER_OPERATION::MOV)
				setMemoryByte(token->memory, _memory, opPointer1, instruction.operand2, memType);
            if (instruction.operation == COMPUTER_OPERATION::ADD)
				setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) + instruction.operand2, memType);
            if (instruction.operation == COMPUTER_OPERATION::SUB)
				setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) - instruction.operand2, memType);
            if (instruction.operation == COMPUTER_OPERATION::MUL)
				setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) * instruction.operand2, memType);
            if (instruction.operation == COMPUTER_OPERATION::DIV) {
                if( instruction.operand2 > 0)
					setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) / instruction.operand2, memType);
                else
					setMemoryByte(token->memory, _memory, opPointer1, 0, memType);
            }
            if (instruction.operation == COMPUTER_OPERATION::XOR)
				setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) ^ instruction.operand2, memType);
            if (instruction.operation == COMPUTER_OPERATION::OR)
				setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) | instruction.operand2, memType);
            if (instruction.operation == COMPUTER_OPERATION::AND)
				setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) & instruction.operand2, memType);
        }

        //if instructions
		instruction.operand1 = getMemoryByte(token->memory, _memory, opPointer1, memType);
        if (instruction.operation == COMPUTER_OPERATION::IFG) {
            if (instruction.operand1 > instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == COMPUTER_OPERATION::IFGE) {
            if (instruction.operand1 >= instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == COMPUTER_OPERATION::IFE) {
            if (instruction.operand1 == instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == COMPUTER_OPERATION::IFNE) {
            if (instruction.operand1 != instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == COMPUTER_OPERATION::IFLE) {
            if (instruction.operand1 <= instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == COMPUTER_OPERATION::IFL) {
            if (instruction.operand1 < instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }

        if (instruction.operation == COMPUTER_OPERATION::ELSE) {
            if( condPointer > 0 )
                condTable[condPointer-1] = !condTable[condPointer-1];
        }

        if (instruction.operation == COMPUTER_OPERATION::ENDIF) {
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
	_memory = _memory.left(_parameters->CELL_MEMSIZE);
	_memory.resize(_parameters->TOKEN_MEMSIZE);
	_code = _code.left(3 * _parameters->CELL_NUM_INSTR);
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
	instructionCoded.operation = static_cast<COMPUTER_OPERATION>((_code[instructionPointer] >> 4) & 0xF);
	instructionCoded.opType1 = static_cast<COMPUTER_OPTYPE>(((_code[instructionPointer] >> 2) & 0x3) % 3);
	instructionCoded.opType2 = static_cast<COMPUTER_OPTYPE>(_code[instructionPointer] & 0x3);
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

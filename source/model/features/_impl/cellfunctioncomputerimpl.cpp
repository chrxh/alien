#include <QString>
#include <qdebug.h>

#include "model/simulationcontext.h"
#include "model/config.h"
#include "model/metadata/symboltable.h"
#include "model/entities/cell.h"
#include "model/entities/token.h"

#include "cellfunctioncomputerimpl.h"

CellFunctionComputerImpl::CellFunctionComputerImpl (SimulationContext* context)
    : CellFunctionComputer(context)
    , _memory(simulationParameters.CELL_MEMSIZE, 0)
	, _symbolTable(context->getSymbolTable())
{
}

CellFunctionComputerImpl::CellFunctionComputerImpl (QByteArray data, SimulationContext* context)
	: CellFunctionComputerImpl(context)
{
	if (!data.isEmpty()) {
		int numInstructions = data[0];
		int minSize = 3 * std::min(numInstructions, simulationParameters.CELL_NUM_INSTR);
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
        quint8 instr, opTyp1, opTyp2;
        qint8 op1, op2;
        decodeInstruction(instructionPointer, instr, opTyp1, opTyp2, op1, op2);

        //write spacing
        for(int j = 0; j < conditionLevel; ++j )
            text += "  ";

        //write operation
        if( instr == static_cast<int>(COMPUTER_OPERATION::MOV) )
            text += "mov";
        if( instr == static_cast<int>(COMPUTER_OPERATION::ADD) )
            text += "add";
        if( instr == static_cast<int>(COMPUTER_OPERATION::SUB) )
            text += "sub";
        if( instr == static_cast<int>(COMPUTER_OPERATION::MUL) )
            text += "mul";
        if( instr == static_cast<int>(COMPUTER_OPERATION::DIV) )
            text += "div";
        if( instr == static_cast<int>(COMPUTER_OPERATION::XOR) )
            text += "xor";
        if( instr == static_cast<int>(COMPUTER_OPERATION::OR) )
            text += "or";
        if( instr == static_cast<int>(COMPUTER_OPERATION::AND) )
            text += "and";
        if( (instr >= static_cast<int>(COMPUTER_OPERATION::IFG)) && (instr <= static_cast<int>(COMPUTER_OPERATION::IFL)) ) {
            text += "if";
            ++conditionLevel;
        }
        if( instr == static_cast<int>(COMPUTER_OPERATION::ELSE) ) {
            if( conditionLevel > 0)
                text.chop(2);
            text += "else";
        }
        if( instr == static_cast<int>(COMPUTER_OPERATION::ENDIF) ) {
            if( conditionLevel > 0) {
                text.chop(2);
                --conditionLevel;
            }
            text += "endif";
        }

        //write operands
        if( opTyp1 == static_cast<int>(COMPUTER_OPTYPE::MEM) )
            textOp1 = "["+ QString("0x%1").arg(convertToAddress(op1, simulationParameters.TOKEN_MEMSIZE),0, 16)+"]";
        if( opTyp1 == static_cast<int>(COMPUTER_OPTYPE::MEMMEM) )
            textOp1 = "[["+ QString("0x%1").arg(convertToAddress(op1, simulationParameters.TOKEN_MEMSIZE),0, 16)+"]]";
        if( opTyp1 == static_cast<int>(COMPUTER_OPTYPE::CMEM) )
            textOp1 = "("+ QString("0x%1").arg(convertToAddress(op1, simulationParameters.CELL_MEMSIZE),0, 16)+")";
        if( opTyp2 == static_cast<int>(COMPUTER_OPTYPE::MEM) )
            textOp2 = "["+ QString("0x%1").arg(convertToAddress(op2, simulationParameters.TOKEN_MEMSIZE),0, 16)+"]";
        if( opTyp2 == static_cast<int>(COMPUTER_OPTYPE::MEMMEM) )
            textOp2 = "[["+ QString("0x%1").arg(convertToAddress(op2, simulationParameters.TOKEN_MEMSIZE),0, 16)+"]]";
        if( opTyp2 == static_cast<int>(COMPUTER_OPTYPE::CMEM) )
            textOp2 = "("+ QString("0x%1").arg(convertToAddress(op2, simulationParameters.CELL_MEMSIZE),0, 16)+")";
        if( opTyp2 == static_cast<int>(COMPUTER_OPTYPE::CONST) )
            textOp2 = QString("0x%1").arg(convertToAddress(op2, simulationParameters.TOKEN_MEMSIZE),0, 16);

        //write separation/comparator
        if (instr <= static_cast<int>(COMPUTER_OPERATION::AND)) {
            text += " " + textOp1 + ", " + textOp2;
        }
        if (instr == static_cast<int>(COMPUTER_OPERATION::IFG))
            text += " " + textOp1 + " > " + textOp2;
        if (instr == static_cast<int>(COMPUTER_OPERATION::IFGE))
            text += " " + textOp1 + " >= " + textOp2;
        if (instr == static_cast<int>(COMPUTER_OPERATION::IFE))
            text += " " + textOp1 + " = " + textOp2;
        if (instr == static_cast<int>(COMPUTER_OPERATION::IFNE))
            text += " " + textOp1 + " != " + textOp2;
        if (instr == static_cast<int>(COMPUTER_OPERATION::IFLE))
            text += " " + textOp1 + " <= " + textOp2;
        if (instr == static_cast<int>(COMPUTER_OPERATION::IFL))
            text += " " + textOp1 + " < " + textOp2;
        if (instructionPointer < _code.size())
            text += "\n";
    }
    return text;
}

CellFunctionComputer::CompilationState CellFunctionComputerImpl::injectAndCompileInstructionCode (QString code)
{
    State state = State::LOOKING_FOR_INSTR_START;

	_code.clear();
    int linePos = 0;
	InstructionUncoded instructionUncoded;
	InstructionCoded instructionCoded;
	for (int bytePos = 0; bytePos < code.length(); ++bytePos) {
        QChar currentSymbol(code[bytePos]);

		if (!stateMachine(state, currentSymbol, instructionUncoded, bytePos, code.length())) {
			return{ false, linePos };
		}
        if( instructionUncoded.readingFinished ) {
			linePos++;
			if (!resolveInstruction(instructionCoded, instructionUncoded)) {
				return{ false, linePos };
			}
            codeInstruction(instructionCoded);
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

bool CellFunctionComputerImpl::resolveInstruction(InstructionCoded& instructionCoded, InstructionUncoded instructionUncoded)
{
	instructionUncoded.op1 = _symbolTable->applyTableToCode(instructionUncoded.op1);
	instructionUncoded.op2 = _symbolTable->applyTableToCode(instructionUncoded.op2);

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
		if ((instructionUncoded.op1.left(2) == "[[") && (instructionUncoded.op1.right(2) == "]]")) {
			instructionCoded.opType1 =  COMPUTER_OPTYPE::MEMMEM;
			instructionUncoded.op1 = instructionUncoded.op1.remove(0, 2);
			instructionUncoded.op1.chop(2);
		}
		else if ((instructionUncoded.op1.left(1) == "[") && (instructionUncoded.op1.right(1) == "]")) {
			instructionCoded.opType1 = COMPUTER_OPTYPE::MEM;
			instructionUncoded.op1 = instructionUncoded.op1.remove(0, 1);
			instructionUncoded.op1.chop(1);
		}
		else if ((instructionUncoded.op1.left(1) == "(") && (instructionUncoded.op1.right(1) == ")")) {
			instructionCoded.opType1 = COMPUTER_OPTYPE::CMEM;
			instructionUncoded.op1 = instructionUncoded.op1.remove(0, 1);
			instructionUncoded.op1.chop(1);
		}
		else {
			return false;
		}

		if ((instructionUncoded.op2.left(2) == "[[") && (instructionUncoded.op2.right(2) == "]]")) {
			instructionCoded.opType2 = COMPUTER_OPTYPE::MEMMEM;
			instructionUncoded.op2 = instructionUncoded.op2.remove(0, 2);
			instructionUncoded.op2.chop(2);
		}
		else if ((instructionUncoded.op2.left(1) == "[") && (instructionUncoded.op2.right(1) == "]")) {
			instructionCoded.opType2 = COMPUTER_OPTYPE::MEM;
			instructionUncoded.op2 = instructionUncoded.op2.remove(0, 1);
			instructionUncoded.op2.chop(1);
		}
		else if ((instructionUncoded.op2.left(1) == "(") && (instructionUncoded.op2.right(1) == ")")) {
			instructionCoded.opType2 = COMPUTER_OPTYPE::CMEM;
			instructionUncoded.op2 = instructionUncoded.op2.remove(0, 1);
			instructionUncoded.op2.chop(1);
		}
		else
			instructionCoded.opType2 = COMPUTER_OPTYPE::CONST;


		if (instructionUncoded.op1.left(2) == "0x") {
			bool ok(true);
			instructionCoded.operand1 = instructionUncoded.op1.remove(0, 2).toInt(&ok, 16);
			if (!ok) {
				return false;
			}
		}
		else {
			bool ok(true);
			instructionCoded.operand1 = instructionUncoded.op1.toInt(&ok, 10);
			if (!ok)
				return false;
		}
		if (instructionUncoded.op2.left(2) == "0x") {
			bool ok(true);
			instructionCoded.operand2 = instructionUncoded.op2.remove(0, 2).toInt(&ok, 16);
			if (!ok) {
				return false;
			}
		}
		else {
			bool ok(true);
			instructionCoded.operand2 = instructionUncoded.op2.toInt(&ok, 10);
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

bool CellFunctionComputerImpl::stateMachine(State &state, QChar &currentSymbol, InstructionUncoded& instruction
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
				instruction.op1 = currentSymbol;
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
				instruction.op1 += currentSymbol;
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
				instruction.op2 = currentSymbol;
			}
		}
		break;
		case State::LOOKING_FOR_OP2_START: {
			if (isNameChar(currentSymbol) || (currentSymbol == '-') || (currentSymbol == '_') || (currentSymbol == '[') || (currentSymbol == '(')) {
				state = State::LOOKING_FOR_OP2_END;
				instruction.op2 = currentSymbol;
				if (bytePos == (codeSize - 1))
					instruction.readingFinished = true;
			}
		}
		break;
		case State::LOOKING_FOR_OP2_END: {
			if (!isNameChar(currentSymbol) && (currentSymbol != '-') && (currentSymbol != '_') && (currentSymbol != '[') && (currentSymbol != ']') && (currentSymbol != '(') && (currentSymbol != ')'))
				instruction.readingFinished = true;
			else {
				instruction.op2 += currentSymbol;
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

    std::vector<bool> condTable(simulationParameters.CELL_NUM_INSTR);
    int condPointer(0);
    int bytePos = 0;
    while( bytePos < _code.size() ) {

        //decode instruction
        quint8 instr, opTyp1, opTyp2;
        qint8 op1, op2;
        decodeInstruction(bytePos, instr, opTyp1, opTyp2, op1, op2);

        //operand 1: pointer to mem
        quint8 opPointer1 = 0;
		MemoryType memType = MemoryType::TOKEN;
        if( opTyp1 == static_cast<int>(COMPUTER_OPTYPE::MEM) )
            opPointer1 = convertToAddress(op1, simulationParameters.TOKEN_MEMSIZE);
        if( opTyp1 == static_cast<int>(COMPUTER_OPTYPE::MEMMEM) ) {
            op1 = token->memory[convertToAddress(op1, simulationParameters.TOKEN_MEMSIZE)];
            opPointer1 = convertToAddress(op1, simulationParameters.TOKEN_MEMSIZE);
        }
		if (opTyp1 == static_cast<int>(COMPUTER_OPTYPE::CMEM)) {
			opPointer1 = convertToAddress(op1, simulationParameters.CELL_MEMSIZE);
			memType = MemoryType::CELL;
		}

        //operand 2: loading value
        if( opTyp2 == static_cast<int>(COMPUTER_OPTYPE::MEM) )
            op2 = token->memory[convertToAddress(op2, simulationParameters.TOKEN_MEMSIZE)];
        if( opTyp2 == static_cast<int>(COMPUTER_OPTYPE::MEMMEM) ) {
            op2 = token->memory[convertToAddress(op2, simulationParameters.TOKEN_MEMSIZE)];
            op2 = token->memory[convertToAddress(op2, simulationParameters.TOKEN_MEMSIZE)];
        }
        if( opTyp2 == static_cast<int>(COMPUTER_OPTYPE::CMEM) )
            op2 = _memory[convertToAddress(op2, simulationParameters.CELL_MEMSIZE)];

        //execute instruction
        bool execute = true;
        for(int k = 0; k < condPointer; ++k)
            if( !condTable[k] )
                execute = false;
        if( execute ) {
//        if( (condPointer == 0) || ((condPointer > 0) && condTable[condPointer-1]) ) {
			if (instr == static_cast<int>(COMPUTER_OPERATION::MOV))
				setMemoryByte(token->memory, _memory, opPointer1, op2, memType);
            if( instr == static_cast<int>(COMPUTER_OPERATION::ADD) )
				setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) + op2, memType);
            if( instr == static_cast<int>(COMPUTER_OPERATION::SUB) )
				setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) - op2, memType);
            if( instr == static_cast<int>(COMPUTER_OPERATION::MUL) )
				setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) * op2, memType);
            if( instr == static_cast<int>(COMPUTER_OPERATION::DIV) ) {
                if( op2 > 0)
					setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) / op2, memType);
                else
					setMemoryByte(token->memory, _memory, opPointer1, 0, memType);
            }
            if( instr == static_cast<int>(COMPUTER_OPERATION::XOR) )
				setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) ^ op2, memType);
            if( instr == static_cast<int>(COMPUTER_OPERATION::OR) )
				setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) | op2, memType);
            if( instr == static_cast<int>(COMPUTER_OPERATION::AND) )
				setMemoryByte(token->memory, _memory, opPointer1, getMemoryByte(token->memory, _memory, opPointer1, memType) & op2, memType);
        }

        //if instructions
		op1 = getMemoryByte(token->memory, _memory, opPointer1, memType);
        if(instr == static_cast<int>(COMPUTER_OPERATION::IFG)) {
            if (op1 > op2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if( instr == static_cast<int>(COMPUTER_OPERATION::IFGE) ) {
            if (op1 >= op2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if( instr == static_cast<int>(COMPUTER_OPERATION::IFE) ) {
            if (op1 == op2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if( instr == static_cast<int>(COMPUTER_OPERATION::IFNE) ) {
            if (op1 != op2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if( instr == static_cast<int>(COMPUTER_OPERATION::IFLE) ) {
            if (op1 <= op2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if( instr == static_cast<int>(COMPUTER_OPERATION::IFL) ) {
            if (op1 < op2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }

        //else instruction
        if( instr == static_cast<int>(COMPUTER_OPERATION::ELSE) ) {
            if( condPointer > 0 )
                condTable[condPointer-1] = !condTable[condPointer-1];
        }

        //endif instruction
        if( instr == static_cast<int>(COMPUTER_OPERATION::ENDIF) ) {
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
	_memory = _memory.left(simulationParameters.CELL_MEMSIZE);
	_memory.resize(simulationParameters.TOKEN_MEMSIZE);
	_code = _code.left(3 * simulationParameters.CELL_NUM_INSTR);
}

QByteArray CellFunctionComputerImpl::getInternalData () const
{
	QByteArray data;
	data.push_back(_code.size() / 3);
	data.push_back(_code);
	return data;
}

void CellFunctionComputerImpl::codeInstruction (InstructionCoded const& instructionCoded)
{
    //machine code: [INSTR - 4 Bits][MEM/MEMMEM/CMEM - 2 Bit][MEM/MEMMEM/CMEM/CONST - 2 Bit]
    _code.push_back((static_cast<quint8>(instructionCoded.operation) << 4)
		| (static_cast<quint8>(instructionCoded.opType1) << 2) | static_cast<quint8>(instructionCoded.opType2));
    _code.push_back(instructionCoded.operand1);
    _code.push_back(instructionCoded.operand2);
}

void CellFunctionComputerImpl::decodeInstruction (int& instructionPointer, quint8& instr, quint8& opTyp1, quint8& opTyp2
	, qint8& op1, qint8& op2) const
{
    //machine code: [INSTR - 4 Bits][MEM/ADDR/CMEM - 2 Bit][MEM/ADDR/CMEM/CONST - 2 Bit]
    instr = (_code[instructionPointer] >> 4) & 0xF;
    opTyp1 = ((_code[instructionPointer] >> 2) & 0x3) % 3;
    opTyp2 = _code[instructionPointer] & 0x3;
    op1 = _code[instructionPointer+1];//readInteger(_code,instructionPointer + 1);
    op2 = _code[instructionPointer+2];//readInteger(_code,instructionPointer + 2);

    //increment instruction pointer
    instructionPointer += 3;
}

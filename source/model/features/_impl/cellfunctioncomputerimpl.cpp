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
    State state(LOOKING_FOR_INSTR_START);

    int linePos = 0;
	Instruction instruction;
	for (int codePos = 0; codePos < code.length(); ++codePos) {
        QChar currentSymbol(code[codePos]);

		if (!stateMachine(state, currentSymbol, instruction, codePos, code.length())) {
			return{ false, linePos };
		}
		if ((currentSymbol == '\n') || ((codePos + 1) == code.length())) {
			linePos++;
			if (!instruction.name.isEmpty()) {
				instruction.read = true;
			}
		}
        if( instruction.read ) {
            instruction.op1 = _symbolTable->applyTableToCode(instruction.op1);
            instruction.op2 = _symbolTable->applyTableToCode(instruction.op2);

            //prepare data for instruction coding
            quint8 instrN(0), opTyp1(0), opTyp2(0);
            qint8 op1N(0), op2N(0);
            if( instruction.name.toLower() == "mov" )
                instrN = static_cast<int>(COMPUTER_OPERATION::MOV);
            else if ( instruction.name.toLower() == "add" )
                instrN = static_cast<int>(COMPUTER_OPERATION::ADD);
            else if ( instruction.name.toLower() == "sub" )
                instrN = static_cast<int>(COMPUTER_OPERATION::SUB);
            else if ( instruction.name.toLower() == "mul" )
                instrN = static_cast<int>(COMPUTER_OPERATION::MUL);
            else if ( instruction.name.toLower() == "div" )
                instrN = static_cast<int>(COMPUTER_OPERATION::DIV);
            else if ( instruction.name.toLower() == "xor" )
                instrN = static_cast<int>(COMPUTER_OPERATION::XOR);
            else if ( instruction.name.toLower() == "or" )
                instrN = static_cast<int>(COMPUTER_OPERATION::OR);
            else if ( instruction.name.toLower() == "and" )
                instrN = static_cast<int>(COMPUTER_OPERATION::AND);
            else if ( instruction.name.toLower() == "if" ) {
                if( instruction.comp.toLower() == ">" )
                    instrN = static_cast<int>(COMPUTER_OPERATION::IFG);
                else if( (instruction.comp.toLower() == ">=" ) || (instruction.comp.toLower() == "=>" ))
                    instrN = static_cast<int>(COMPUTER_OPERATION::IFGE);
                else if( (instruction.comp.toLower() == "=" ) || (instruction.comp.toLower() == "==" ))
                    instrN = static_cast<int>(COMPUTER_OPERATION::IFE);
                else if( instruction.comp.toLower() == "!=" )
                    instrN = static_cast<int>(COMPUTER_OPERATION::IFNE);
                else if( (instruction.comp.toLower() == "<=" ) || (instruction.comp.toLower() == "=<" ))
                    instrN = static_cast<int>(COMPUTER_OPERATION::IFLE);
                else if( instruction.comp.toLower() == "<" )
                    instrN = static_cast<int>(COMPUTER_OPERATION::IFL);
                else {
                    return {false, linePos};
                }
            }
            else if ( instruction.name.toLower() == "else" )
                instrN = static_cast<int>(COMPUTER_OPERATION::ELSE);
            else if ( instruction.name.toLower() == "endif" )
                instrN = static_cast<int>(COMPUTER_OPERATION::ENDIF);
            else {
                return {false, linePos};
            }

            if( (instrN != static_cast<int>(COMPUTER_OPERATION::ELSE)) && (instrN != static_cast<int>(COMPUTER_OPERATION::ENDIF)) ) {
                if( (instruction.op1.left(2) == "[[") && (instruction.op1.right(2) == "]]") ) {
                    opTyp1 = static_cast<int>(COMPUTER_OPTYPE::MEMMEM);
                    instruction.op1 = instruction.op1.remove(0,2);
                    instruction.op1.chop(2);
                }
                else if( (instruction.op1.left(1) == "[") && (instruction.op1.right(1) == "]") ) {
                    opTyp1 = static_cast<int>(COMPUTER_OPTYPE::MEM);
                    instruction.op1 = instruction.op1.remove(0,1);
                    instruction.op1.chop(1);
                }
                else if( (instruction.op1.left(1) == "(") && (instruction.op1.right(1) == ")") ) {
                    opTyp1 = static_cast<int>(COMPUTER_OPTYPE::CMEM);
                    instruction.op1 = instruction.op1.remove(0,1);
                    instruction.op1.chop(1);
                }
                else {
                    return {false, linePos};
                }

                if( (instruction.op2.left(2) == "[[") && (instruction.op2.right(2) == "]]") ) {
                    opTyp2 = static_cast<int>(COMPUTER_OPTYPE::MEMMEM);
                    instruction.op2 = instruction.op2.remove(0,2);
                    instruction.op2.chop(2);
                }
                else if( (instruction.op2.left(1) == "[") && (instruction.op2.right(1) == "]") ) {
                    opTyp2 = static_cast<int>(COMPUTER_OPTYPE::MEM);
                    instruction.op2 = instruction.op2.remove(0,1);
                    instruction.op2.chop(1);
                }
                else if( (instruction.op2.left(1) == "(") && (instruction.op2.right(1) == ")") ) {
                    opTyp2 = static_cast<int>(COMPUTER_OPTYPE::CMEM);
                    instruction.op2 = instruction.op2.remove(0,1);
                    instruction.op2.chop(1);
                }
                else
                    opTyp2 = static_cast<int>(COMPUTER_OPTYPE::CONST);


                if( instruction.op1.left(2) == "0x" ) {
                    bool ok(true);
                    op1N = instruction.op1.remove(0,2).toInt(&ok, 16);
                    if( !ok ) {
                        return {false, linePos};
                    }
                }
                else {
                    bool ok(true);
                    op1N = instruction.op1.toInt(&ok, 10);
                    if( !ok )
                        return {false, linePos};
                }
                if( instruction.op2.left(2) == "0x" ) {
                    bool ok(true);
                    op2N = instruction.op2.remove(0,2).toInt(&ok, 16);
                    if( !ok ) {
                        return {false, linePos};
                    }
                }
                else {
                    bool ok(true);
                    op2N = instruction.op2.toInt(&ok, 10);
                    if( !ok ) {
                        return {false, linePos};
                    }
                }
            }
            else {
                opTyp1 = 0;
                opTyp2 = 0;
                op1N = 0;
                op2N = 0;
            }

            codeInstruction(instrN, opTyp1, opTyp2, op1N, op2N);
            state = LOOKING_FOR_INSTR_START;
			instruction = Instruction();
        }
    }
    if( state == LOOKING_FOR_INSTR_START )
        return {true, linePos};
    else {
        return {false, linePos};
    }
}

bool CellFunctionComputerImpl::stateMachine(State &state, QChar &currentSymbol, Instruction& instruction
	, int symbolPos, int codeSize)
{
	switch (state) {
		case LOOKING_FOR_INSTR_START: {
			if (currentSymbol.isLetter()) {
				state = LOOKING_FOR_INSTR_END;
				instruction.name = currentSymbol;
			}
		}
		break;
		case LOOKING_FOR_INSTR_END: {
			if (!currentSymbol.isLetter()) {
				if ((instruction.name.toLower() == "else") || (instruction.name.toLower() == "endif"))
					instruction.read = true;
				else
					state = LOOKING_FOR_OP1_START;
			}
			else {
				instruction.name += currentSymbol;
				if ((symbolPos + 1) == codeSize && ((instruction.name.toLower() == "else") || (instruction.name.toLower() == "endif")))
					instruction.read = true;
			}
		}
		break;
		case LOOKING_FOR_OP1_START: {
			if (isNameChar(currentSymbol) || (currentSymbol == '-') || (currentSymbol == '_') || (currentSymbol == '[') || (currentSymbol == '(')) {
				state = LOOKING_FOR_OP1_END;
				instruction.op1 = currentSymbol;
			}
		}
		break;
		case LOOKING_FOR_OP1_END: {
			if ((currentSymbol == '<') || (currentSymbol == '>') || (currentSymbol == '=') || (currentSymbol == '!')) {
				state = LOOKING_FOR_COMPARATOR;
				instruction.comp = currentSymbol;
			}
			else if (currentSymbol == ',')
				state = LOOKING_FOR_OP2_START;
			else if (!isNameChar(currentSymbol) && (currentSymbol != '-') && (currentSymbol != '_') && (currentSymbol != '[') && (currentSymbol != ']') && (currentSymbol != '(') && (currentSymbol != ')'))
				state = LOOKING_FOR_SEPARATOR;
			else
				instruction.op1 += currentSymbol;
		}
		break;
		case LOOKING_FOR_SEPARATOR: {
			if (currentSymbol == ',')
				state = LOOKING_FOR_OP2_START;
			else if ((currentSymbol == '<') || (currentSymbol == '>') || (currentSymbol == '=') || (currentSymbol == '!')) {
				state = LOOKING_FOR_COMPARATOR;
				instruction.comp = currentSymbol;
			}
			else if (isNameChar(currentSymbol) || (currentSymbol == '-') || (currentSymbol == '_') || (currentSymbol == '[') || (currentSymbol == ']') || (currentSymbol == '(') || (currentSymbol == ')'))
				return false;
		}
		break;
		case LOOKING_FOR_COMPARATOR: {
			if ((currentSymbol == '<') || (currentSymbol == '>') || (currentSymbol == '=') || (currentSymbol == '!'))
				instruction.comp += currentSymbol;
			else if (!isNameChar(currentSymbol) && (currentSymbol != '-') && (currentSymbol != '_') && (currentSymbol != '[') && (currentSymbol != '('))
				state = LOOKING_FOR_OP2_START;
			else {
				state = LOOKING_FOR_OP2_END;
				instruction.op2 = currentSymbol;
			}
		}
		break;
		case LOOKING_FOR_OP2_START: {
			if (isNameChar(currentSymbol) || (currentSymbol == '-') || (currentSymbol == '_') || (currentSymbol == '[') || (currentSymbol == '(')) {
				state = LOOKING_FOR_OP2_END;
				instruction.op2 = currentSymbol;
				if (symbolPos == (codeSize - 1))
					instruction.read = true;
			}
		}
		break;
		case LOOKING_FOR_OP2_END: {
			if (!isNameChar(currentSymbol) && (currentSymbol != '-') && (currentSymbol != '_') && (currentSymbol != '[') && (currentSymbol != ']') && (currentSymbol != '(') && (currentSymbol != ')'))
				instruction.read = true;
			else {
				instruction.op2 += currentSymbol;
				if ((symbolPos + 1) == codeSize)
					instruction.read = true;
			}
		}
		break;
	}
	return true;
}

QByteArray& CellFunctionComputerImpl::getMemoryReference ()
{
    return _memory;
}


CellFeature::ProcessingResult CellFunctionComputerImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};

    std::vector<bool> condTable(simulationParameters.CELL_NUM_INSTR);
    int condPointer(0);
    int i(0);
    while( i < _code.size() ) {

        //decode instruction
        quint8 instr, opTyp1, opTyp2;
        qint8 op1, op2;
        decodeInstruction(i, instr, opTyp1, opTyp2, op1, op2);

        //operand 1: pointer to mem
        qint8* op1Pointer = 0;
        if( opTyp1 == static_cast<int>(COMPUTER_OPTYPE::MEM) )
            op1Pointer = (qint8*)&(token->memory[convertToAddress(op1, simulationParameters.TOKEN_MEMSIZE)]);
        if( opTyp1 == static_cast<int>(COMPUTER_OPTYPE::MEMMEM) ) {
            op1 = token->memory[convertToAddress(op1, simulationParameters.TOKEN_MEMSIZE)];
            op1Pointer = (qint8*)&(token->memory[convertToAddress(op1, simulationParameters.TOKEN_MEMSIZE)]);
        }
        if( opTyp1 == static_cast<int>(COMPUTER_OPTYPE::CMEM) )
            op1Pointer = (qint8*)&(_memory[convertToAddress(op1, simulationParameters.CELL_MEMSIZE)]);

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
            if( instr == static_cast<int>(COMPUTER_OPERATION::MOV) )
                *op1Pointer = op2;
            if( instr == static_cast<int>(COMPUTER_OPERATION::ADD) )
                *op1Pointer += op2;
            if( instr == static_cast<int>(COMPUTER_OPERATION::SUB) )
                *op1Pointer -= op2;
            if( instr == static_cast<int>(COMPUTER_OPERATION::MUL) )
                *op1Pointer *= op2;
            if( instr == static_cast<int>(COMPUTER_OPERATION::DIV) ) {
                if( op2 > 0)
                    *op1Pointer /= op2;
                else
                    *op1Pointer = 0;
            }
            if( instr == static_cast<int>(COMPUTER_OPERATION::XOR) )
                *op1Pointer ^= op2;
            if( instr == static_cast<int>(COMPUTER_OPERATION::OR) )
                *op1Pointer |= op2;
            if( instr == static_cast<int>(COMPUTER_OPERATION::AND) )
                *op1Pointer &= op2;
        }

            //if instructions
            if( instr == static_cast<int>(COMPUTER_OPERATION::IFG) ) {
                if( (qint8)(*op1Pointer) > (qint8)op2 )
                    condTable[condPointer] = true;
                else
                    condTable[condPointer] = false;
                condPointer++;
            }
            if( instr == static_cast<int>(COMPUTER_OPERATION::IFGE) ) {
                if( (qint8)(*op1Pointer) >= (qint8)op2 )
                    condTable[condPointer] = true;
                else
                    condTable[condPointer] = false;
                condPointer++;
            }
            if( instr == static_cast<int>(COMPUTER_OPERATION::IFE) ) {
                if( (qint8)(*op1Pointer) == (qint8)op2 )
                    condTable[condPointer] = true;
                else
                    condTable[condPointer] = false;
                condPointer++;
            }
            if( instr == static_cast<int>(COMPUTER_OPERATION::IFNE) ) {
                if( (qint8)(*op1Pointer) != (qint8)op2 )
                    condTable[condPointer] = true;
                else
                    condTable[condPointer] = false;
                condPointer++;
            }
            if( instr == static_cast<int>(COMPUTER_OPERATION::IFLE) ) {
                if( (qint8)(*op1Pointer) <= (qint8)op2 )
                    condTable[condPointer] = true;
                else
                    condTable[condPointer] = false;
                condPointer++;
            }
            if( instr == static_cast<int>(COMPUTER_OPERATION::IFL) ) {
                if( (qint8)(*op1Pointer) < (qint8)op2 )
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

void CellFunctionComputerImpl::codeInstruction (quint8 instr, quint8 opTyp1, quint8 opTyp2, qint8 op1, qint8 op2)
{
    //machine code: [INSTR - 4 Bits][MEM/MEMMEM/CMEM - 2 Bit][MEM/MEMMEM/CMEM/CONST - 2 Bit]
    _code.push_back((instr << 4) | (opTyp1 << 2) | opTyp2);
    _code.push_back(op1);
    _code.push_back(op2);
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

#include <QString>
#include <qdebug.h>

#include "Base/NumberGenerator.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/Settings.h"
#include "ModelBasic/SymbolTable.h"
#include "ModelBasic/CompilerHelper.h"
#include "UnitContext.h"
#include "Token.h"

#include "Cell.h"
#include "CellComputerFunctionImpl.h"

CellComputerFunctionImpl::CellComputerFunctionImpl(QByteArray const& code, QByteArray const& memory, UnitContext* context)
	: CellComputerFunction(context)
{
	int numInstructions = code.size() / 3;
	int copySize = 3 * std::min(numInstructions, context->getSimulationParameters().cellFunctionComputerMaxInstructions);
	_code = code.left(copySize);

	int memorySize = context->getSimulationParameters().cellFunctionComputerCellMemorySize;
	_memory = memory.left(memorySize);
	if (memorySize > _memory.size()) {
		_memory.append(memorySize - _memory.size(), 0);
	}
}

void CellComputerFunctionImpl::mutateImpl()
{
	auto numberGen = _context->getNumberGenerator();
	qint8 randomByte = static_cast<qint8>(numberGen->getRandomInt(256));
	if (numberGen->getRandomInt(2) == 0) {
		if (!_code.isEmpty()) {
			_code[numberGen->getRandomInt(_code.size())] = randomByte;
		}
	}
	else {
		if (!_memory.isEmpty()) {
			_memory[numberGen->getRandomInt(_memory.size())] = randomByte;
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

CellFeatureChain::ProcessingResult CellComputerFunctionImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};

	auto parameters = _context->getSimulationParameters();
	vector<bool> condTable(parameters.cellFunctionComputerMaxInstructions);
    int condPointer(0);
	CHECK((_code.size() % 3) == 0);
	for (int instructionPointer = 0; instructionPointer < _code.size(); ) {

        //decode instruction
		InstructionCoded instruction;
        CompilerHelper::readInstruction(_code, instructionPointer, instruction);

        //operand 1: pointer to mem
        quint8 opPointer1 = 0;
		MemoryType memType = MemoryType::TOKEN;
        if (instruction.opType1 == Enums::ComputerOptype::MEM)
            opPointer1 = CompilerHelper::convertToAddress(instruction.operand1, parameters.tokenMemorySize);
        if (instruction.opType1 == Enums::ComputerOptype::MEMMEM) {
            instruction.operand1 = token->getMemoryRef()[CompilerHelper::convertToAddress(instruction.operand1, parameters.tokenMemorySize)];
            opPointer1 = CompilerHelper::convertToAddress(instruction.operand1, parameters.tokenMemorySize);
        }
		if (instruction.opType1 == Enums::ComputerOptype::CMEM) {
			opPointer1 = CompilerHelper::convertToAddress(instruction.operand1, parameters.cellFunctionComputerCellMemorySize);
			memType = MemoryType::CELL;
		}

        //operand 2: loading value
        if (instruction.opType2 == Enums::ComputerOptype::MEM)
            instruction.operand2 = token->getMemoryRef()[CompilerHelper::convertToAddress(instruction.operand2, parameters.tokenMemorySize)];
        if (instruction.opType2 == Enums::ComputerOptype::MEMMEM) {
            instruction.operand2 = token->getMemoryRef()[CompilerHelper::convertToAddress(instruction.operand2, parameters.tokenMemorySize)];
            instruction.operand2 = token->getMemoryRef()[CompilerHelper::convertToAddress(instruction.operand2, parameters.tokenMemorySize)];
        }
        if (instruction.opType2 == Enums::ComputerOptype::CMEM)
            instruction.operand2 = _memory[CompilerHelper::convertToAddress(instruction.operand2, parameters.cellFunctionComputerCellMemorySize)];

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

void CellComputerFunctionImpl::appendDescriptionImpl(CellFeatureDescription & desc) const
{
	desc.setType(getType());
	desc.setConstData(_code);
	desc.setVolatileData(_memory);
}

QByteArray CellComputerFunctionImpl::getInternalData () const
{
	QByteArray data;
	data.push_back(_code);
	return data;
}


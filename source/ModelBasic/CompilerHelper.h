#pragma once

#include "Definitions.h"

class CompilerHelper
{
public:
	static void writeInstruction(QByteArray& data, InstructionCoded const& instructionCoded)
	{
		//machine code: [INSTR - 4 Bits][MEM/MEMMEM/CMEM - 2 Bit][MEM/MEMMEM/CMEM/CONST - 2 Bit]
		data.push_back((static_cast<quint8>(instructionCoded.operation) << 4)
			| (static_cast<quint8>(instructionCoded.opType1) << 2) | static_cast<quint8>(instructionCoded.opType2));
		data.push_back(instructionCoded.operand1);
		data.push_back(instructionCoded.operand2);
	}

	static void readInstruction(QByteArray const& data, int& instructionPointer, InstructionCoded& instructionCoded)
	{
		//machine code: [INSTR - 4 Bits][MEM/ADDR/CMEM - 2 Bit][MEM/ADDR/CMEM/CONST - 2 Bit]
		instructionCoded.operation = static_cast<Enums::ComputerOperation::Type>((data[instructionPointer] >> 4) & 0xF);
		instructionCoded.opType1 = static_cast<Enums::ComputerOptype::Type>(((data[instructionPointer] >> 2) & 0x3) % 3);
		instructionCoded.opType2 = static_cast<Enums::ComputerOptype::Type>(data[instructionPointer] & 0x3);
		instructionCoded.operand1 = data[instructionPointer + 1];//readInteger(_code,instructionPointer + 1);
		instructionCoded.operand2 = data[instructionPointer + 2];//readInteger(_code,instructionPointer + 2);

																 //increment instruction pointer
		instructionPointer += 3;
	}

	static quint8 convertToAddress(qint8 addr, quint32 size)
	{
		quint32 t((quint32)((quint8)addr));
		return ((t % size) + size) % size;
	}
};



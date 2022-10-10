#pragma once

#include "EngineInterface/Enums.h"
#include "EngineInterface/CellInstruction.h"

#include "SimulationData.cuh"
#include "Cell.cuh"
#include "Token.cuh"
#include "AccessTOs.cuh"

class CellComputationProcessor
{
public:
    __inline__ __device__ static void process(Token* token);

private:
    __inline__ __device__ static void
        readInstruction(char const* data, int& instructionPointer, CellInstruction& instructionCoded);

    __inline__ __device__ static uint8_t convertToAddress(int8_t addr, uint32_t size);

    enum class MemoryType {
        Token, Cell
    };

    __inline__ __device__ static int8_t
        getMemoryByte(char const* tokenMemory, char const* cellMemory, unsigned char pointer, MemoryType type);

    __inline__ __device__ static void
        setMemoryByte(char* tokenMemory, char* cellMemory, unsigned char pointer, char value, MemoryType type);

};

__inline__ __device__ void CellComputationProcessor::process(Token* token)
{
    auto cell = token->cell;
    bool condTable[MAX_CELL_STATIC_BYTES / 3 + 1];
    int condPointer(0);
    int numStaticBytes = min(static_cast<unsigned char>(cell->staticData[0]), cudaSimulationParameters.cellFunctionComputerMaxInstructions) * 3;
    for (int instructionPointer = 0; instructionPointer < numStaticBytes; ) {

        //decode instruction
        CellInstruction instruction;
        readInstruction(cell->staticData, instructionPointer, instruction);

        //operand 1: pointer to mem
        uint8_t opPointer1 = 0;
        MemoryType memType = MemoryType::Token;
        if (instruction.opType1 == Enums::ComputationOpType_Mem)
            opPointer1 = convertToAddress(instruction.operand1, cudaSimulationParameters.tokenMemorySize);
        if (instruction.opType1 == Enums::ComputationOpType_MemMem) {
            instruction.operand1 = token->memory[convertToAddress(instruction.operand1, cudaSimulationParameters.tokenMemorySize)];
            opPointer1 = convertToAddress(instruction.operand1, cudaSimulationParameters.tokenMemorySize);
        }
        if (instruction.opType1 == Enums::ComputationOpType_Cmem) {
            opPointer1 = convertToAddress(instruction.operand1, cudaSimulationParameters.cellFunctionComputerCellMemorySize);
            memType = MemoryType::Cell;
        }

        //operand 2: loading value
        if (instruction.opType2 == Enums::ComputationOpType_Mem)
            instruction.operand2 = token->memory[convertToAddress(instruction.operand2, cudaSimulationParameters.tokenMemorySize)];
        if (instruction.opType2 == Enums::ComputationOpType_MemMem) {
            instruction.operand2 = token->memory[convertToAddress(instruction.operand2, cudaSimulationParameters.tokenMemorySize)];
            instruction.operand2 = token->memory[convertToAddress(instruction.operand2, cudaSimulationParameters.tokenMemorySize)];
        }
        if (instruction.opType2 == Enums::ComputationOpType_Cmem)
            instruction.operand2 = cell->mutableData[convertToAddress(instruction.operand2, cudaSimulationParameters.cellFunctionComputerCellMemorySize)];

        //execute instruction
        bool execute = true;
        for (int k = 0; k < condPointer; ++k)
            if (!condTable[k])
                execute = false;
        if (execute) {
            if (instruction.operation == Enums::ComputationOperation_Mov)
                setMemoryByte(token->memory, cell->mutableData, opPointer1, instruction.operand2, memType);
            if (instruction.operation == Enums::ComputationOperation_Add)
                setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) + instruction.operand2, memType);
            if (instruction.operation == Enums::ComputationOperation_Sub)
                setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) - instruction.operand2, memType);
            if (instruction.operation == Enums::ComputationOperation_Mul)
                setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) * instruction.operand2, memType);
            if (instruction.operation == Enums::ComputationOperation_Div) {
                if (instruction.operand2 > 0)
                    setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) / instruction.operand2, memType);
                else
                    setMemoryByte(token->memory, cell->mutableData, opPointer1, 0, memType);
            }
            if (instruction.operation == Enums::ComputationOperation_Xor)
                setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) ^ instruction.operand2, memType);
            if (instruction.operation == Enums::ComputationOperation_Or)
                setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) | instruction.operand2, memType);
            if (instruction.operation == Enums::ComputationOperation_And)
                setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) & instruction.operand2, memType);
        }

        //if instructions
        instruction.operand1 = getMemoryByte(token->memory, cell->mutableData, opPointer1, memType);
        if (instruction.operation == Enums::ComputationOperation_Ifg) {
            if (instruction.operand1 > instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == Enums::ComputationOperation_Ifge) {
            if (instruction.operand1 >= instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == Enums::ComputationOperation_Ife) {
            if (instruction.operand1 == instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == Enums::ComputationOperation_Ifne) {
            if (instruction.operand1 != instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == Enums::ComputationOperation_Ifle) {
            if (instruction.operand1 <= instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }
        if (instruction.operation == Enums::ComputationOperation_Ifl) {
            if (instruction.operand1 < instruction.operand2)
                condTable[condPointer] = true;
            else
                condTable[condPointer] = false;
            condPointer++;
        }

        if (instruction.operation == Enums::ComputationOperation_Else) {
            if (condPointer > 0)
                condTable[condPointer - 1] = !condTable[condPointer - 1];
        }

        if (instruction.operation == Enums::ComputationOperation_Endif) {
            if (condPointer > 0)
                condPointer--;
        }
    }
}

__inline__ __device__ void
    CellComputationProcessor::readInstruction(char const* data, int& instructionPointer, CellInstruction& instructionCoded)
{
    //machine code: [INSTR - 4 Bits][MEM/ADDR/CMEM - 2 Bit][MEM/ADDR/CMEM/CONST - 2 Bit]
    instructionCoded.operation = (data[instructionPointer + 1] >> 4) & 0xF;
    instructionCoded.opType1 = ((data[instructionPointer + 1] >> 2) & 0x3) % 3;
    instructionCoded.opType2 = data[instructionPointer + 1] & 0x3;
    instructionCoded.operand1 = data[instructionPointer + 2];
    instructionCoded.operand2 = data[instructionPointer + 3];

    instructionPointer += 3;
}

__inline__ __device__ uint8_t CellComputationProcessor::convertToAddress(int8_t addr, uint32_t size)
{
    auto t = static_cast<uint32_t>(static_cast<uint8_t>(addr));
    return ((t % size) + size) % size;
}

__inline__ __device__ int8_t
CellComputationProcessor::getMemoryByte(char const* tokenMemory, char const* cellMemory, unsigned char pointer, MemoryType type)
{
    if (type == MemoryType::Token) {
        return tokenMemory[pointer];
    }
    if (type == MemoryType::Cell) {
        return cellMemory[pointer];
    }
    return tokenMemory[pointer];
}

__inline__ __device__ void
CellComputationProcessor::setMemoryByte(char* tokenMemory, char* cellMemory, unsigned char pointer, char value, MemoryType type)
{
    if (type == MemoryType::Token) {
        tokenMemory[pointer] = value;
    }
    if (type == MemoryType::Cell) {
        cellMemory[pointer] = value;
    }
}

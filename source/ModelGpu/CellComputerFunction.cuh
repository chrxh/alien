#pragma once

#include "SimulationData.cuh"

class CellComputerFunction
{
public:
    __inline__ __device__ static void processing(Token* token)
    {
        auto cell = token->cell;
        bool condTable[MAX_CELL_STATIC_BYTES / 3 + 1];
        int condPointer(0);
        for (int instructionPointer = 0; instructionPointer < cell->numStaticBytes; ) {

            //decode instruction
            InstructionCoded instruction;
            readInstruction(cell->staticData, instructionPointer, instruction);

            //operand 1: pointer to mem
            uint8_t opPointer1 = 0;
            MemoryType memType = MemoryType::Token;
            if (instruction.opType1 == Enums::ComputerOptype::MEM)
                opPointer1 = convertToAddress(instruction.operand1, cudaSimulationParameters.tokenMemorySize);
            if (instruction.opType1 == Enums::ComputerOptype::MEMMEM) {
                instruction.operand1 = token->memory[convertToAddress(instruction.operand1, cudaSimulationParameters.tokenMemorySize)];
                opPointer1 = convertToAddress(instruction.operand1, cudaSimulationParameters.tokenMemorySize);
            }
            if (instruction.opType1 == Enums::ComputerOptype::CMEM) {
                opPointer1 = convertToAddress(instruction.operand1, cudaSimulationParameters.cellFunctionComputerCellMemorySize);
                memType = MemoryType::Cell;
            }

            //operand 2: loading value
            if (instruction.opType2 == Enums::ComputerOptype::MEM)
                instruction.operand2 = token->memory[convertToAddress(instruction.operand2, cudaSimulationParameters.tokenMemorySize)];
            if (instruction.opType2 == Enums::ComputerOptype::MEMMEM) {
                instruction.operand2 = token->memory[convertToAddress(instruction.operand2, cudaSimulationParameters.tokenMemorySize)];
                instruction.operand2 = token->memory[convertToAddress(instruction.operand2, cudaSimulationParameters.tokenMemorySize)];
            }
            if (instruction.opType2 == Enums::ComputerOptype::CMEM)
                instruction.operand2 = cell->mutableData[convertToAddress(instruction.operand2, cudaSimulationParameters.cellFunctionComputerCellMemorySize)];

            //execute instruction
            bool execute = true;
            for (int k = 0; k < condPointer; ++k)
                if (!condTable[k])
                    execute = false;
            if (execute) {
                if (instruction.operation == Enums::ComputerOperation::MOV)
                    setMemoryByte(token->memory, cell->mutableData, opPointer1, instruction.operand2, memType);
                if (instruction.operation == Enums::ComputerOperation::ADD)
                    setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) + instruction.operand2, memType);
                if (instruction.operation == Enums::ComputerOperation::SUB)
                    setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) - instruction.operand2, memType);
                if (instruction.operation == Enums::ComputerOperation::MUL)
                    setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) * instruction.operand2, memType);
                if (instruction.operation == Enums::ComputerOperation::DIV) {
                    if (instruction.operand2 > 0)
                        setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) / instruction.operand2, memType);
                    else
                        setMemoryByte(token->memory, cell->mutableData, opPointer1, 0, memType);
                }
                if (instruction.operation == Enums::ComputerOperation::XOR)
                    setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) ^ instruction.operand2, memType);
                if (instruction.operation == Enums::ComputerOperation::OR)
                    setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) | instruction.operand2, memType);
                if (instruction.operation == Enums::ComputerOperation::AND)
                    setMemoryByte(token->memory, cell->mutableData, opPointer1, getMemoryByte(token->memory, cell->mutableData, opPointer1, memType) & instruction.operand2, memType);
            }

            //if instructions
            instruction.operand1 = getMemoryByte(token->memory, cell->mutableData, opPointer1, memType);
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
                if (condPointer > 0)
                    condTable[condPointer - 1] = !condTable[condPointer - 1];
            }

            if (instruction.operation == Enums::ComputerOperation::ENDIF) {
                if (condPointer > 0)
                    condPointer--;
            }
        }
    }

    private:
        __inline__ __device__ static void
        readInstruction(char const* data, int& instructionPointer, InstructionCoded& instructionCoded)
        {
            //machine code: [INSTR - 4 Bits][MEM/ADDR/CMEM - 2 Bit][MEM/ADDR/CMEM/CONST - 2 Bit]
            instructionCoded.operation =
                static_cast<Enums::ComputerOperation::Type>((data[instructionPointer] >> 4) & 0xF);
            instructionCoded.opType1 =
                static_cast<Enums::ComputerOptype::Type>(((data[instructionPointer] >> 2) & 0x3) % 3);
            instructionCoded.opType2 =
                static_cast<Enums::ComputerOptype::Type>(data[instructionPointer] & 0x3);
            instructionCoded.operand1 = data[instructionPointer + 1];
            instructionCoded.operand2 = data[instructionPointer + 2];

            instructionPointer += 3;
        }

        __inline__ __device__ static uint8_t convertToAddress(int8_t addr, uint32_t size)
        {
            auto t = static_cast<uint32_t>(static_cast<uint8_t>(addr));
            return ((t % size) + size) % size;
        }

        enum class MemoryType {
            Token, Cell
        };

        __inline__ __device__ static int8_t
        getMemoryByte(char const* tokenMemory, char const* cellMemory, unsigned char pointer, MemoryType type)
        {
            if (type == MemoryType::Token) {
                return tokenMemory[pointer];
            }
            if (type == MemoryType::Cell) {
                return cellMemory[pointer];
            }
            return tokenMemory[pointer];
        }

        __inline__ __device__ static void
        setMemoryByte(char* tokenMemory, char* cellMemory, unsigned char pointer, char value, MemoryType type)
        {
            if (type == MemoryType::Token) {
                tokenMemory[pointer] = value;
            }
            if (type == MemoryType::Cell) {
                cellMemory[pointer] = value;
            }
        }
};
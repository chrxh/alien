#pragma once

#include "Definitions.h"
#include "SymbolMap.h"
#include "SimulationParameters.h"

struct CompilationResult
{
    bool compilationOk = true;
    int lineOfFirstError = 0;
    std::string compilation;
};

/**
 * Simple compiler for cell's machine language
 */
class CellComputationCompiler
{
public:
    static CompilationResult compileSourceCode(std::string const& code, SymbolMap const& symbols);
    static std::string
    decompileSourceCode(std::string const& data, SymbolMap const& symbols, SimulationParameters const& parameters);

    static int getMaxBytes(SimulationParameters const& parameters);

private:
    static void writeInstruction(std::string& data, InstructionCoded const& instructionCoded);
    static void readInstruction(
        std::string const& data,
        int& instructionPointer,
        InstructionCoded& instructionCoded);
    static uint8_t convertToAddress(int8_t addr, uint32_t size);
};

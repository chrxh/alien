#pragma once

#include "EngineInterface/CellInstruction.h"

#include "Definitions.h"
#include "SymbolMap.h"
#include "Descriptions.h"
#include "SimulationParameters.h"


struct CompilationResult
{
    bool compilationOk = true;
    int lineOfFirstError = 0;
    StaticData compilation;
};

/**
 * Simple compiler for cell's machine language
 */
class CellComputationCompiler
{
public:
    static CompilationResult compileSourceCode(std::string const& code, SymbolMap const& symbols, SimulationParameters const& parameters);
    static std::string decompileSourceCode(StaticData const& data, SymbolMap const& symbols, SimulationParameters const& parameters);

    static std::optional<int> extractAddress(std::string const& s);
    static int getBytesPerInstruction();

private:
    static int getMaxCompiledCodeSize(SimulationParameters const& parameters);
    static void writeInstruction(StaticData& data, CellInstruction const& instructionCoded);
    static void readInstruction(StaticData const& data, int& instructionPointer,
        CellInstruction& instructionCoded);
    static uint8_t convertToAddress(int8_t addr, uint32_t size);
};

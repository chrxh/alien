#include "GenomeTranslator.h"

#include <variant>
#include <boost/range/adaptors.hpp>

#include "Base/Definitions.h"

namespace
{
    uint8_t convertAngleToByte(float value) { return static_cast<uint8_t>(static_cast<int8_t>(value / 180 * 128)); }
    uint8_t convertDistanceToByte(float value) { return static_cast<uint8_t>(static_cast<int8_t>((value - 1.0f) * 128)); }
    uint8_t convertNeuronPropertyToByte(float value)
    {
        CHECK(std::abs(value) <= 2);
        return static_cast<uint8_t>(static_cast<int8_t>(value / 2 * 128));
    }
    uint8_t convertBoolToByte(bool value) { return value ? 1 : 0; }
    std::vector<uint8_t> convertWordToBytes(int value) { return {static_cast<uint8_t>(value & 0xff), static_cast<uint8_t>((value >> 8) % 0xff)}; }
}

std::vector<uint8_t> GenomeTranslator::encode(GenomeDescription const& cells)
{
    std::vector<uint8_t> result;
    result.reserve(cells.size() * 6);
    for (auto const& [index, cell] : cells | boost::adaptors::indexed(0)) {
        result.emplace_back(static_cast<uint8_t>(cell.getCellFunctionType()));
        result.emplace_back(convertAngleToByte(cell.referenceAngle));
        result.emplace_back(convertDistanceToByte(cell.referenceDistance));
        result.emplace_back(cell.maxConnections);
        result.emplace_back(cell.executionOrderNumber);
        result.emplace_back(cell.color);
        result.emplace_back(convertBoolToByte(cell.inputBlocked));
        result.emplace_back(convertBoolToByte(cell.outputBlocked));
        switch (cell.getCellFunctionType()) {
        case Enums::CellFunction_Neuron: {
            auto neuron = std::get<NeuronGenomeDescription>(*cell.cellFunction);
            for (int row = 0; row < MAX_CHANNELS; ++row) {
                for (int col = 0; col < MAX_CHANNELS; ++col) {
                    result.emplace_back(convertNeuronPropertyToByte(neuron.weights[row][col]));
                }
            }
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                result.emplace_back(convertNeuronPropertyToByte(neuron.bias[i]));
            }
        } break;
        case Enums::CellFunction_Transmitter: {
        } break;
        case Enums::CellFunction_Constructor: {
            auto constructor = std::get<ConstructorGenomeDescription>(*cell.cellFunction);
            result.emplace_back(static_cast<uint8_t>(constructor.mode));
            result.emplace_back(convertBoolToByte(constructor.singleConstruction));
            result.emplace_back(convertBoolToByte(constructor.separateConstruction));
            result.emplace_back(convertBoolToByte(constructor.makeSticky));
            result.emplace_back(static_cast<uint8_t>(constructor.angleAlignment));
            auto makeGenomeCopy = constructor.genome.size() == 0;
            result.emplace_back(convertBoolToByte(makeGenomeCopy));
            if (!makeGenomeCopy) {
                auto lengthBytes = convertWordToBytes(static_cast<int>(constructor.genome.size()));
                result.insert(result.end(), lengthBytes.begin(), lengthBytes.end());
                result.insert(result.end(), constructor.genome.begin(), constructor.genome.end());
            }
        } break;
        case Enums::CellFunction_Sensor: {
        } break;
        case Enums::CellFunction_Nerve: {
        } break;
        case Enums::CellFunction_Attacker: {
        } break;
        case Enums::CellFunction_Injector: {
        } break;
        case Enums::CellFunction_Muscle: {
        } break;
        case Enums::CellFunction_Placeholder1: {
        } break;
        case Enums::CellFunction_Placeholder2: {
        } break;
        }
    }
    return result;
}

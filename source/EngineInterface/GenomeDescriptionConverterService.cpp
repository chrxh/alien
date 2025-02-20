#include "GenomeDescriptionConverterService.h"

#include <variant>

#include "Base/Definitions.h"

#include "GenomeConstants.h"

namespace
{
    void writeByte(std::vector<uint8_t>& data, int value)
    {
        data.emplace_back(static_cast<uint8_t>(value));
    }
    void writeOptionalByte(std::vector<uint8_t>& data, std::optional<int> value)
    {
        data.emplace_back(static_cast<uint8_t>(value.value_or(-1)));
    }
    void writeByteWithInfinity(std::vector<uint8_t>& data, int value)
    {
        data.emplace_back(static_cast<uint8_t>(std::min(255, value)));
    }
    void writeBool(std::vector<uint8_t>& data, bool value)
    {
        data.emplace_back(value ? 1 : 0);
    }
    void writeFloat(std::vector<uint8_t>& data, float value) { data.emplace_back(static_cast<uint8_t>(static_cast<int8_t>(value * 128))); }
    void writeWord(std::vector<uint8_t>& data, int value)
    {
        data.emplace_back(static_cast<uint8_t>(value & 0xff));
        data.emplace_back(static_cast<uint8_t>((value >> 8) % 0xff));
    }
    void writeAngle(std::vector<uint8_t>& data, float value)
    {
        if (value > 180.0f) {
            value -= 360.0f;
        }
        if (value < -180.0f) {
            value += 360.0f;
        }
        data.emplace_back(static_cast<uint8_t>(static_cast<int8_t>(value / 180 * 120)));
    }
    void writeDensity(std::vector<uint8_t>& data, float value)
    {
        data.emplace_back(static_cast<uint8_t>(static_cast<int8_t>((value * 2 - 1) * 128)));
    }
    void writeEnergy(std::vector<uint8_t>& data, float value)
    {
        writeFloat(data, (value - 150.0f) / 100);
    }
    void writeNeuronProperty(std::vector<uint8_t>& data, float value)
    {
        value = std::max(-3.9f, std::min(3.9f, value));
        writeFloat(data, value / 4);
    }
    void writeDistance(std::vector<uint8_t>& data, float value)
    {
        data.emplace_back(static_cast<uint8_t>((value - 0.5f) * 255));
    }
    void writeStiffness(std::vector<uint8_t>& data, float value) { data.emplace_back(static_cast<uint8_t>(value * 255)); }
    void writeGenome(std::vector<uint8_t>& data, std::variant<MakeGenomeCopy, std::vector<uint8_t>> const& value)
    {
        auto makeGenomeCopy = std::holds_alternative<MakeGenomeCopy>(value);
        writeBool(data, makeGenomeCopy);
        if (!makeGenomeCopy) {
            auto genome = std::get<std::vector<uint8_t>>(value);
            writeWord(data, static_cast<int>(genome.size()));
            data.insert(data.end(), genome.begin(), genome.end());
        }
    }

    uint8_t readByte(std::vector<uint8_t> const& data, int& pos)
    {
        if (pos >= data.size()) {
            return 0;
        }
        uint8_t result = data[pos++];
        return result;
    }
    std::optional<int> readOptionalByte(std::vector<uint8_t> const& data, int& pos)
    {
        auto value = static_cast<int>(readByte(data, pos));
        return value > 127 ? std::nullopt : std::make_optional(value);
    }
    std::optional<int> readOptionalByte(std::vector<uint8_t> const& data, int& pos, int moduloValue)
    {
        auto value = static_cast<int>(readByte(data, pos));
        return value > 127 ? std::nullopt : std::make_optional(value % moduloValue);
    }

    int convertByteToByteWithInfinity(uint8_t const& b)
    {
        return b == 255 ? std::numeric_limits<int>::max() : b;

    }
    int readByteWithInfinity(std::vector<uint8_t> const& data, int& pos)
    {
        return convertByteToByteWithInfinity(readByte(data, pos));
    }
    bool readBool(std::vector<uint8_t> const& data, int& pos)
    {
        return static_cast<int8_t>(readByte(data, pos)) > 0;
    }
    int readWord(std::vector<uint8_t> const& data, int& pos)
    {
        return static_cast<int>(readByte(data, pos)) | (static_cast<int>(readByte(data, pos) << 8));
    }
    //between -1 and 1
    float readFloat(std::vector<uint8_t> const& data, int& pos)
    {
        return static_cast<float>(static_cast<int8_t>(readByte(data, pos))) / 128;
    }
    //between -180 and 180
    float readAngle(std::vector<uint8_t> const& data, int& pos)
    {
        return static_cast<float>(static_cast<int8_t>(readByte(data, pos))) / 120 * 180;
    }
    //between 36 and 1060
    float readEnergy(std::vector<uint8_t> const& data, int& pos)
    {
        return readFloat(data, pos) * 100 + 150.0f; 
    }
    //between 0 and 1
    float readDensity(std::vector<uint8_t> const& data, int& pos)
    {
        return (readFloat(data, pos) + 1.0f) / 2;
    }
    float readNeuronProperty(std::vector<uint8_t> const& data, int& pos) { return readFloat(data, pos) * 4; }
    float readDistance(std::vector<uint8_t> const& data, int& pos)
    {
        return toFloat(readByte(data, pos)) / 255 + 0.5f;
    }
    float readStiffness(std::vector<uint8_t> const& data, int& pos)
    {
        return toFloat(readByte(data, pos)) / 255;
    }

    std::variant<MakeGenomeCopy, std::vector<uint8_t>> readGenome(std::vector<uint8_t> const& data, int& pos)
    {
        std::variant<MakeGenomeCopy, std::vector<uint8_t>> result;

        bool makeGenomeCopy = readBool(data, pos);
        if (makeGenomeCopy) {
            result = MakeGenomeCopy();
        } else {
            auto size = readWord(data, pos);
            size = std::min(size, toInt(data.size()) - pos);
            std::vector<uint8_t> copiedGenome;
            copiedGenome.reserve(size);
            for (int i = 0; i < size; ++i) {
                copiedGenome.emplace_back(readByte(data, pos));
            }
            result = copiedGenome;
        }
        return result;
    }
}

std::vector<uint8_t> GenomeDescriptionConverterService::convertDescriptionToBytes(GenomeDescription const& genome)
{
    auto const& cells = genome._cells;
    std::vector<uint8_t> result;
    result.reserve(cells.size() * (Const::CellBasicBytes + Const::ConstructorFixedBytes) + Const::GenomeHeaderSize);
    writeByte(result, genome._header._shape);
    writeByte(result, genome._header._numBranches);
    writeBool(result, genome._header._separateConstruction);
    writeByte(result, genome._header._angleAlignment);
    writeStiffness(result, genome._header._stiffness);
    writeDistance(result, genome._header._connectionDistance);
    writeByteWithInfinity(result, genome._header._numRepetitions);
    writeAngle(result, genome._header._concatenationAngle1);
    writeAngle(result, genome._header._concatenationAngle2);
    writeAngle(result, genome._header._frontAngle);

    for (auto const& cell : cells) {
        writeByte(result, cell.getCellType());
        writeAngle(result, cell._referenceAngle);
        writeEnergy(result, cell._energy);
        writeByte(result, cell._numRequiredAdditionalConnections);
        writeByte(result, cell._color);
        writeBool(result, cell._signalRoutingRestriction._active);
        writeAngle(result, cell._signalRoutingRestriction._baseAngle);
        writeAngle(result, cell._signalRoutingRestriction._openingAngle);

        auto weights = cell._neuralNetwork.getWeights();
        for (int row = 0; row < MAX_CHANNELS; ++row) {
            for (int col = 0; col < MAX_CHANNELS; ++col) {
                writeNeuronProperty(result, weights[row, col]);
            }
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            writeNeuronProperty(result, cell._neuralNetwork._biases[i]);
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            writeByte(result, cell._neuralNetwork._activationFunctions[i]);
        }

        switch (cell.getCellType()) {
        case CellType_Base: {
        } break;
        case CellType_Depot: {
            auto const& transmitter = std::get<DepotGenomeDescription>(cell._cellTypeData);
            writeByte(result, transmitter._mode);
        } break;
        case CellType_Constructor: {
            auto const& constructor = std::get<ConstructorGenomeDescription>(cell._cellTypeData);
            writeByte(result, constructor._autoTriggerInterval);
            writeWord(result, constructor._constructionActivationTime);
            writeAngle(result, constructor._constructionAngle1);
            writeAngle(result, constructor._constructionAngle2);
            writeGenome(result, constructor._genome);
        } break;
        case CellType_Sensor: {
            auto const& sensor = std::get<SensorGenomeDescription>(cell._cellTypeData);
            writeByte(result, sensor._autoTriggerInterval);
            writeDensity(result, sensor._minDensity);
            writeOptionalByte(result, sensor._restrictToColor);
            writeByte(result, sensor._restrictToMutants);
            writeOptionalByte(result, sensor._minRange);
            writeOptionalByte(result, sensor._maxRange);
        } break;
        case CellType_Oscillator: {
            auto const& oscillator = std::get<OscillatorGenomeDescription>(cell._cellTypeData);
            writeByte(result, oscillator._autoTriggerInterval);
            writeByte(result, oscillator._alternationInterval);
        } break;
        case CellType_Attacker: {
            auto const& attacker = std::get<AttackerGenomeDescription>(cell._cellTypeData);
            writeByte(result, attacker._mode);
        } break;
        case CellType_Injector: {
            auto const& injector = std::get<InjectorGenomeDescription>(cell._cellTypeData);
            writeByte(result, injector._mode);
            writeGenome(result, injector._genome);
        } break;
        case CellType_Muscle: {
            auto const& muscle = std::get<MuscleGenomeDescription>(cell._cellTypeData);
            auto mode = muscle.getMode();
            writeByte(result, mode);
            if (mode == MuscleMode_AutoBending) {
                auto const& bending = std::get<AutoBendingGenomeDescription>(muscle._mode);
                writeFloat(result, bending._maxAngleDeviation);
                writeFloat(result, bending._frontBackVelRatio);
            } else if (mode == MuscleMode_ManualBending) {
                auto const& bending = std::get<ManualBendingGenomeDescription>(muscle._mode);
                writeFloat(result, bending._maxAngleDeviation);
                writeFloat(result, bending._frontBackVelRatio);
            }
        } break;
        case CellType_Defender: {
            auto const& defender = std::get<DefenderGenomeDescription>(cell._cellTypeData);
            writeByte(result, defender._mode);
        } break;
        case CellType_Reconnector: {
            auto const& reconnector = std::get<ReconnectorGenomeDescription>(cell._cellTypeData);
            writeOptionalByte(result, reconnector._restrictToColor);
            writeByte(result, reconnector._restrictToMutants);
        } break;
        case CellType_Detonator: {
            auto const& detonator = std::get<DetonatorGenomeDescription>(cell._cellTypeData);
            writeWord(result, detonator._countdown);
        } break;
        }
    }
    return result;
}

namespace
{
    struct ConversionResult
    {
        GenomeDescription genome;
        int lastBytePosition = 0;
    };
    
    ConversionResult convertBytesToDescriptionIntern(
        std::vector<uint8_t> const& data,
        size_t maxBytePosition,
        size_t maxEntries)
    {
        ConversionResult result;

        int nodeIndex = 0;
        auto& bytePosition = result.lastBytePosition;

        result.genome._header._shape = readByte(data, bytePosition) % ConstructionShape_Count;
        result.genome._header._numBranches = (readByte(data, bytePosition) + 5) % 6 + 1;
        result.genome._header._separateConstruction = readBool(data, bytePosition);
        result.genome._header._angleAlignment = readByte(data, bytePosition) % ConstructorAngleAlignment_Count;
        result.genome._header._stiffness = readStiffness(data, bytePosition);
        result.genome._header._connectionDistance = readDistance(data, bytePosition);
        result.genome._header._numRepetitions = readByteWithInfinity(data, bytePosition);
        result.genome._header._concatenationAngle1 = readAngle(data, bytePosition);
        result.genome._header._concatenationAngle2 = readAngle(data, bytePosition);
        result.genome._header._frontAngle = readAngle(data, bytePosition);
        
        while (bytePosition < maxBytePosition && nodeIndex < maxEntries) {
            CellType cellType = readByte(data, bytePosition) % CellType_Count;

            CellGenomeDescription cell;
            cell._referenceAngle = readAngle(data, bytePosition);
            cell._energy = readEnergy(data, bytePosition);
            cell._numRequiredAdditionalConnections = readByte(data, bytePosition) % MAX_CELL_BONDS;
            cell._color = readByte(data, bytePosition) % MAX_COLORS;
            cell._signalRoutingRestriction._active = readBool(data, bytePosition);
            cell._signalRoutingRestriction._baseAngle = readAngle(data, bytePosition);
            cell._signalRoutingRestriction._openingAngle = readAngle(data, bytePosition);

            auto weights = cell._neuralNetwork.getWeights();
            for (int row = 0; row < MAX_CHANNELS; ++row) {
                for (int col = 0; col < MAX_CHANNELS; ++col) {
                    weights[row, col] = readNeuronProperty(data, bytePosition);
                }
            }
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cell._neuralNetwork._biases[i] = readNeuronProperty(data, bytePosition);
            }
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cell._neuralNetwork._activationFunctions[i] = readByte(data, bytePosition) % ActivationFunction_Count;
            }

            switch (cellType) {
            case CellType_Base: {
                BaseGenomeDescription base;
                cell._cellTypeData = base;
            } break;
            case CellType_Depot: {
                DepotGenomeDescription transmitter;
                transmitter._mode = readByte(data, bytePosition) % EnergyDistributionMode_Count;
                cell._cellTypeData = transmitter;
            } break;
            case CellType_Constructor: {
                ConstructorGenomeDescription constructor;
                constructor._autoTriggerInterval = readByte(data, bytePosition);
                constructor._constructionActivationTime = readWord(data, bytePosition);
                constructor._constructionAngle1 = readAngle(data, bytePosition);
                constructor._constructionAngle2 = readAngle(data, bytePosition);
                constructor._genome = readGenome(data, bytePosition);
                cell._cellTypeData = constructor;
            } break;
            case CellType_Sensor: {
                SensorGenomeDescription sensor;
                sensor._autoTriggerInterval = readByte(data, bytePosition);
                sensor._minDensity = readDensity(data, bytePosition);
                sensor._restrictToColor = readOptionalByte(data, bytePosition, MAX_COLORS);
                sensor._restrictToMutants = readByte(data, bytePosition) % SensorRestrictToMutants_Count;
                sensor._minRange = readOptionalByte(data, bytePosition);
                sensor._maxRange = readOptionalByte(data, bytePosition);
                cell._cellTypeData = sensor;
            } break;
            case CellType_Oscillator: {
                OscillatorGenomeDescription oscillator;
                oscillator._autoTriggerInterval = readByte(data, bytePosition);
                oscillator._alternationInterval = readByte(data, bytePosition);
                cell._cellTypeData = oscillator;
            } break;
            case CellType_Attacker: {
                AttackerGenomeDescription attacker;
                attacker._mode = readByte(data, bytePosition) % EnergyDistributionMode_Count;
                cell._cellTypeData = attacker;
            } break;
            case CellType_Injector: {
                InjectorGenomeDescription injector;
                injector._mode = readByte(data, bytePosition) % InjectorMode_Count;
                injector._genome = readGenome(data, bytePosition);
                cell._cellTypeData = injector;
            } break;
            case CellType_Muscle: {
                MuscleGenomeDescription muscle;
                auto mode = readByte(data, bytePosition) % MuscleMode_Count;
                if (mode == MuscleMode_AutoBending) {
                    AutoBendingGenomeDescription bending;
                    bending._maxAngleDeviation = readFloat(data, bytePosition);
                    bending._frontBackVelRatio = readFloat(data, bytePosition);
                    muscle._mode = bending;
                } else if (mode == MuscleMode_ManualBending) {
                    ManualBendingGenomeDescription bending;
                    bending._maxAngleDeviation = readFloat(data, bytePosition);
                    bending._frontBackVelRatio = readFloat(data, bytePosition);
                    muscle._mode = bending;
                }
                cell._cellTypeData = muscle;
            } break;
            case CellType_Defender: {
                DefenderGenomeDescription defender;
                defender._mode = readByte(data, bytePosition) % DefenderMode_Count;
                cell._cellTypeData = defender;
            } break;
            case CellType_Reconnector: {
                ReconnectorGenomeDescription reconnector;
                reconnector._restrictToColor = readOptionalByte(data, bytePosition, MAX_COLORS);
                reconnector._restrictToMutants = readByte(data, bytePosition) % ReconnectorRestrictToMutants_Count;
                cell._cellTypeData = reconnector;
            } break;
            case CellType_Detonator: {
                DetonatorGenomeDescription detonator;
                detonator._countdown = readWord(data, bytePosition);
                cell._cellTypeData = detonator;
            } break;
            }
            result.genome._cells.emplace_back(cell);
            ++nodeIndex;
        }
        return result;
    }

}

GenomeDescription GenomeDescriptionConverterService::convertBytesToDescription(std::vector<uint8_t> const& data)
{
    return convertBytesToDescriptionIntern(data, data.size(), data.size()).genome;
}

int GenomeDescriptionConverterService::convertNodeAddressToNodeIndex(std::vector<uint8_t> const& data, int nodeAddress)
{
    //wasteful approach but sufficient for GUI
    return convertBytesToDescriptionIntern(data, nodeAddress, data.size()).genome._cells.size();
}

int GenomeDescriptionConverterService::convertNodeIndexToNodeAddress(std::vector<uint8_t> const& data, int nodeIndex)
{
    //wasteful approach but sufficient for GUI
    return convertBytesToDescriptionIntern(data, data.size(), nodeIndex).lastBytePosition;
}

int GenomeDescriptionConverterService::getNumNodesRecursively(std::vector<uint8_t> const& data, bool includeRepetitions)
{
    auto genome = convertBytesToDescriptionIntern(data, data.size(), data.size()).genome;
    auto result = toInt(genome._cells.size());
    for (auto const& node : genome._cells) {
        if (auto subgenome = node.getGenome()) {
            result += getNumNodesRecursively(*subgenome, includeRepetitions);
        }
    }

    auto numRepetitions = genome._header._numRepetitions == std::numeric_limits<int>::max() ? 1 : genome._header._numRepetitions;
    return includeRepetitions ? result * numRepetitions * genome._header.getNumBranches() : result;
}

int GenomeDescriptionConverterService::getNumRepetitions(std::vector<uint8_t> const& data)
{
    return convertByteToByteWithInfinity(data.at(Const::GenomeHeaderNumRepetitionsPos));
}

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

std::vector<uint8_t> GenomeDescriptionConverterService::convertDescriptionToBytes(GenomeDescription const& genome, GenomeEncodingSpecification const& spec)
{
    auto const& cells = genome.cells;
    std::vector<uint8_t> result;
    result.reserve(cells.size() * (Const::CellBasicBytes + Const::ConstructorFixedBytes) + Const::GenomeHeaderSize);
    writeByte(result, genome.header.shape);
    writeByte(result, genome.header.numBranches);
    writeBool(result, genome.header.separateConstruction);
    writeByte(result, genome.header.angleAlignment);
    writeStiffness(result, genome.header.stiffness);
    writeDistance(result, genome.header.connectionDistance);
    if (spec._numRepetitions) {
        writeByteWithInfinity(result, genome.header.numRepetitions);
    }
    if (spec._concatenationAngle1) {
        writeAngle(result, genome.header.concatenationAngle1);
    }
    if (spec._concatenationAngle2) {
        writeAngle(result, genome.header.concatenationAngle2);
    }

    for (auto const& cell : cells) {
        writeByte(result, cell.getCellType());
        writeAngle(result, cell.referenceAngle);
        writeEnergy(result, cell.energy);
        writeOptionalByte(result, cell.numRequiredAdditionalConnections);
        writeByte(result, cell.color);
        writeBool(result, cell.signalRoutingRestriction.active);
        writeAngle(result, cell.signalRoutingRestriction.baseAngle);
        writeAngle(result, cell.signalRoutingRestriction.openingAngle);

        auto weights = cell.neuralNetwork.getWeights();
        for (int row = 0; row < MAX_CHANNELS; ++row) {
            for (int col = 0; col < MAX_CHANNELS; ++col) {
                writeNeuronProperty(result, weights[row, col]);
            }
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            writeNeuronProperty(result, cell.neuralNetwork.biases[i]);
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            writeByte(result, cell.neuralNetwork.activationFunctions[i]);
        }

        switch (cell.getCellType()) {
        case CellType_Base: {
        } break;
        case CellType_Depot: {
            auto const& transmitter = std::get<DepotGenomeDescription>(cell.cellTypeData);
            writeByte(result, transmitter.mode);
        } break;
        case CellType_Constructor: {
            auto const& constructor = std::get<ConstructorGenomeDescription>(cell.cellTypeData);
            writeByte(result, constructor.mode);
            writeWord(result, constructor.constructionActivationTime);
            writeAngle(result, constructor.constructionAngle1);
            writeAngle(result, constructor.constructionAngle2);
            writeGenome(result, constructor.genome);
        } break;
        case CellType_Sensor: {
            auto const& sensor = std::get<SensorGenomeDescription>(cell.cellTypeData);
            writeDensity(result, sensor.minDensity);
            writeOptionalByte(result, sensor.restrictToColor);
            writeByte(result, sensor.restrictToMutants);
            writeOptionalByte(result, sensor.minRange);
            writeOptionalByte(result, sensor.maxRange);
        } break;
        case CellType_Oscillator: {
            auto const& oscillator = std::get<OscillatorGenomeDescription>(cell.cellTypeData);
            writeByte(result, oscillator.pulseMode);
            writeByte(result, oscillator.alternationMode);
        } break;
        case CellType_Attacker: {
            auto const& attacker = std::get<AttackerGenomeDescription>(cell.cellTypeData);
            writeByte(result, attacker.mode);
        } break;
        case CellType_Injector: {
            auto const& injector = std::get<InjectorGenomeDescription>(cell.cellTypeData);
            writeByte(result, injector.mode);
            writeGenome(result, injector.genome);
        } break;
        case CellType_Muscle: {
            auto const& muscle = std::get<MuscleGenomeDescription>(cell.cellTypeData);
            writeByte(result, muscle.mode);
        } break;
        case CellType_Defender: {
            auto const& defender = std::get<DefenderGenomeDescription>(cell.cellTypeData);
            writeByte(result, defender.mode);
        } break;
        case CellType_Reconnector: {
            auto const& reconnector = std::get<ReconnectorGenomeDescription>(cell.cellTypeData);
            writeOptionalByte(result, reconnector.restrictToColor);
            writeByte(result, reconnector.restrictToMutants);
        } break;
        case CellType_Detonator: {
            auto const& detonator = std::get<DetonatorGenomeDescription>(cell.cellTypeData);
            writeWord(result, detonator.countdown);
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
        size_t maxEntries,
        GenomeEncodingSpecification const& spec)
    {
        ConversionResult result;

        int nodeIndex = 0;
        auto& bytePosition = result.lastBytePosition;

        result.genome.header.shape = readByte(data, bytePosition) % ConstructionShape_Count;
        result.genome.header.numBranches = (readByte(data, bytePosition) + 5) % 6 + 1;
        result.genome.header.separateConstruction = readBool(data, bytePosition);
        result.genome.header.angleAlignment = readByte(data, bytePosition) % ConstructorAngleAlignment_Count;
        result.genome.header.stiffness = readStiffness(data, bytePosition);
        result.genome.header.connectionDistance = readDistance(data, bytePosition);
        if (spec._numRepetitions) {
            result.genome.header.numRepetitions = readByteWithInfinity(data, bytePosition);
        }
        if (spec._concatenationAngle1) {
            result.genome.header.concatenationAngle1 = readAngle(data, bytePosition);
        }
        if (spec._concatenationAngle2) {
            result.genome.header.concatenationAngle2 = readAngle(data, bytePosition);
        }
        
        while (bytePosition < maxBytePosition && nodeIndex < maxEntries) {
            CellType cellType = readByte(data, bytePosition) % CellType_Count;

            CellGenomeDescription cell;
            cell.referenceAngle = readAngle(data, bytePosition);
            cell.energy = readEnergy(data, bytePosition);
            cell.numRequiredAdditionalConnections = readOptionalByte(data, bytePosition, MAX_CELL_BONDS + 1);
            cell.color = readByte(data, bytePosition) % MAX_COLORS;
            cell.signalRoutingRestriction.active = readBool(data, bytePosition);
            cell.signalRoutingRestriction.baseAngle = readAngle(data, bytePosition);
            cell.signalRoutingRestriction.openingAngle = readAngle(data, bytePosition);

            auto weights = cell.neuralNetwork.getWeights();
            for (int row = 0; row < MAX_CHANNELS; ++row) {
                for (int col = 0; col < MAX_CHANNELS; ++col) {
                    weights[row, col] = readNeuronProperty(data, bytePosition);
                }
            }
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cell.neuralNetwork.biases[i] = readNeuronProperty(data, bytePosition);
            }
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cell.neuralNetwork.activationFunctions[i] = readByte(data, bytePosition) % ActivationFunction_Count;
            }

            switch (cellType) {
            case CellType_Base: {
                BaseGenomeDescription base;
                cell.cellTypeData = base;
            } break;
            case CellType_Depot: {
                DepotGenomeDescription transmitter;
                transmitter.mode = readByte(data, bytePosition) % EnergyDistributionMode_Count;
                cell.cellTypeData = transmitter;
            } break;
            case CellType_Constructor: {
                ConstructorGenomeDescription constructor;
                constructor.mode = readByte(data, bytePosition);
                constructor.constructionActivationTime = readWord(data, bytePosition);
                constructor.constructionAngle1 = readAngle(data, bytePosition);
                constructor.constructionAngle2 = readAngle(data, bytePosition);
                constructor.genome = readGenome(data, bytePosition);
                cell.cellTypeData = constructor;
            } break;
            case CellType_Sensor: {
                SensorGenomeDescription sensor;
                sensor.minDensity = readDensity(data, bytePosition);
                sensor.restrictToColor = readOptionalByte(data, bytePosition, MAX_COLORS);
                sensor.restrictToMutants = readByte(data, bytePosition) % SensorRestrictToMutants_Count;
                sensor.minRange = readOptionalByte(data, bytePosition);
                sensor.maxRange = readOptionalByte(data, bytePosition);
                cell.cellTypeData = sensor;
            } break;
            case CellType_Oscillator: {
                OscillatorGenomeDescription oscillator;
                oscillator.pulseMode = readByte(data, bytePosition);
                oscillator.alternationMode = readByte(data, bytePosition);
                cell.cellTypeData = oscillator;
            } break;
            case CellType_Attacker: {
                AttackerGenomeDescription attacker;
                attacker.mode = readByte(data, bytePosition) % EnergyDistributionMode_Count;
                cell.cellTypeData = attacker;
            } break;
            case CellType_Injector: {
                InjectorGenomeDescription injector;
                injector.mode = readByte(data, bytePosition) % InjectorMode_Count;
                injector.genome = readGenome(data, bytePosition);
                cell.cellTypeData = injector;
            } break;
            case CellType_Muscle: {
                MuscleGenomeDescription muscle;
                muscle.mode = readByte(data, bytePosition) % MuscleMode_Count;
                cell.cellTypeData = muscle;
            } break;
            case CellType_Defender: {
                DefenderGenomeDescription defender;
                defender.mode = readByte(data, bytePosition) % DefenderMode_Count;
                cell.cellTypeData = defender;
            } break;
            case CellType_Reconnector: {
                ReconnectorGenomeDescription reconnector;
                reconnector.restrictToColor = readOptionalByte(data, bytePosition, MAX_COLORS);
                reconnector.restrictToMutants = readByte(data, bytePosition) % ReconnectorRestrictToMutants_Count;
                cell.cellTypeData = reconnector;
            } break;
            case CellType_Detonator: {
                DetonatorGenomeDescription detonator;
                detonator.countdown = readWord(data, bytePosition);
                cell.cellTypeData = detonator;
            } break;
            }
            result.genome.cells.emplace_back(cell);
            ++nodeIndex;
        }
        return result;
    }

}

GenomeDescription GenomeDescriptionConverterService::convertBytesToDescription(std::vector<uint8_t> const& data, GenomeEncodingSpecification const& spec)
{
    return convertBytesToDescriptionIntern(data, data.size(), data.size(), spec).genome;
}

int GenomeDescriptionConverterService::convertNodeAddressToNodeIndex(std::vector<uint8_t> const& data, int nodeAddress, GenomeEncodingSpecification const& spec)
{
    //wasteful approach but sufficient for GUI
    return convertBytesToDescriptionIntern(data, nodeAddress, data.size(), spec).genome.cells.size();
}

int GenomeDescriptionConverterService::convertNodeIndexToNodeAddress(std::vector<uint8_t> const& data, int nodeIndex, GenomeEncodingSpecification const& spec)
{
    //wasteful approach but sufficient for GUI
    return convertBytesToDescriptionIntern(data, data.size(), nodeIndex, spec).lastBytePosition;
}

int GenomeDescriptionConverterService::getNumNodesRecursively(std::vector<uint8_t> const& data, bool includeRepetitions, GenomeEncodingSpecification const& spec)
{
    auto genome = convertBytesToDescriptionIntern(data, data.size(), data.size(), spec).genome;
    auto result = toInt(genome.cells.size());
    for (auto const& node : genome.cells) {
        if (auto subgenome = node.getGenome()) {
            result += getNumNodesRecursively(*subgenome, includeRepetitions, spec);
        }
    }

    auto numRepetitions = genome.header.numRepetitions == std::numeric_limits<int>::max() ? 1 : genome.header.numRepetitions;
    return includeRepetitions ? result * numRepetitions * genome.header.getNumBranches() : result;
}

int GenomeDescriptionConverterService::getNumRepetitions(std::vector<uint8_t> const& data)
{
    return convertByteToByteWithInfinity(data.at(Const::GenomeHeaderNumRepetitionsPos));
}

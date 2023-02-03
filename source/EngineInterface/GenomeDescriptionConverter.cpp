#include "GenomeDescriptionConverter.h"

#include <variant>

#include "Base/Definitions.h"

namespace
{
    void writeByte(std::vector<uint8_t>& data, int value) {data.emplace_back(static_cast<uint8_t>(value)); }
    void writeBool(std::vector<uint8_t>& data, bool value) { data.emplace_back(value ? 1 : 0); }
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
    void writeDistance(std::vector<uint8_t>& data, float value)
    {
        writeFloat(data, value - 1.0f);
    }
    void writeNeuronProperty(std::vector<uint8_t>& data, float value)
    {
        value = std::max(-3.9f, std::min(3.9f, value));
        writeFloat(data, value / 4);
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
    bool readBool(std::vector<uint8_t> const& data, int& pos) { return static_cast<int8_t>(readByte(data, pos)) > 0; }
    int readWord(std::vector<uint8_t> const& data, int& pos)
    {
        return static_cast<int>(readByte(data, pos)) | (static_cast<int>(readByte(data, pos) << 8));
    }
    float readFloat(std::vector<uint8_t> const& data, int& pos) { return static_cast<float>(static_cast<int8_t>(readByte(data, pos))) / 128.0f; }
    float readAngle(std::vector<uint8_t> const& data, int& pos) { return static_cast<float>(static_cast<int8_t>(readByte(data, pos))) / 120 * 180; }
    float readDensity(std::vector<uint8_t> const& data, int& pos) { return (readFloat(data, pos) + 1.0f) / 2; }
    float readNeuronProperty(std::vector<uint8_t> const& data, int& pos) { return readFloat(data, pos) * 4; }
    float readDistance(std::vector<uint8_t> const& data, int& pos)
    {
        return readFloat(data, pos) + 1.0f;
    }
    float readStiffness(std::vector<uint8_t> const& data, int& pos)
    {
        return static_cast<float>(readByte(data, pos)) / 255;
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

std::vector<uint8_t> GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription const& genome)
{

    std::vector<uint8_t> result;
    result.reserve(genome.size() * 6);
    int index = 0;
    for (auto const& cell : genome) {
        writeByte(result, cell.getCellFunctionType());
        writeAngle(result, cell.referenceAngle);
        writeDistance(result, cell.referenceDistance);
        writeByte(result, cell.maxConnections);
        writeByte(result, cell.executionOrderNumber);
        writeByte(result, cell.color);
        writeBool(result, cell.inputBlocked);
        writeBool(result, cell.outputBlocked);
        switch (cell.getCellFunctionType()) {
        case CellFunction_Neuron: {
            auto const& neuron = std::get<NeuronGenomeDescription>(*cell.cellFunction);
            for (int row = 0; row < MAX_CHANNELS; ++row) {
                for (int col = 0; col < MAX_CHANNELS; ++col) {
                    writeNeuronProperty(result, neuron.weights[row][col]);
                }
            }
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                writeNeuronProperty(result, neuron.biases[i]);
            }
        } break;
        case CellFunction_Transmitter: {
            auto const& transmitter = std::get<TransmitterGenomeDescription>(*cell.cellFunction);
            writeByte(result, transmitter.mode);
        } break;
        case CellFunction_Constructor: {
            auto const& constructor = std::get<ConstructorGenomeDescription>(*cell.cellFunction);
            writeByte(result, constructor.mode);
            writeBool(result, constructor.singleConstruction);
            writeBool(result, constructor.separateConstruction);
            writeBool(result, constructor.adaptMaxConnections);
            writeByte(result, constructor.angleAlignment);
            writeStiffness(result, constructor.stiffness);
            writeWord(result, constructor.constructionActivationTime);
            writeGenome(result, constructor.genome);
        } break;
        case CellFunction_Sensor: {
            auto const& sensor = std::get<SensorGenomeDescription>(*cell.cellFunction);
            writeByte(result, sensor.fixedAngle.has_value() ? SensorMode_FixedAngle : SensorMode_Neighborhood);
            writeAngle(result, sensor.fixedAngle.has_value() ? *sensor.fixedAngle : 0.0f);
            writeDensity(result, sensor.minDensity);
            writeByte(result, sensor.color);
        } break;
        case CellFunction_Nerve: {
            auto const& nerve = std::get<NerveGenomeDescription>(*cell.cellFunction);
            writeByte(result, nerve.pulseMode);
            writeByte(result, nerve.alternationMode);
        } break;
        case CellFunction_Attacker: {
            auto const& attacker = std::get<AttackerGenomeDescription>(*cell.cellFunction);
            writeByte(result, attacker.mode);
        } break;
        case CellFunction_Injector: {
            auto const& injector = std::get<InjectorGenomeDescription>(*cell.cellFunction);
            writeByte(result, injector.mode);
            writeGenome(result, injector.genome);
        } break;
        case CellFunction_Muscle: {
            auto const& muscle = std::get<MuscleGenomeDescription>(*cell.cellFunction);
            writeByte(result, muscle.mode);
        } break;
        case CellFunction_Placeholder1: {
        } break;
        case CellFunction_Placeholder2: {
        } break;
        }
        ++index;
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
    
    ConversionResult
    convertBytesToDescriptionIntern(
        std::vector<uint8_t> const& data,
        SimulationParameters const& parameters,
        size_t maxBytePosition,
        size_t maxEntries)
    {
        ConversionResult result;
        int cellIndex = 0;
        auto& bytePosition = result.lastBytePosition;
        while (bytePosition < maxBytePosition && cellIndex < maxEntries) {
            CellFunction cellFunction = readByte(data, bytePosition) % CellFunction_Count;

            CellGenomeDescription cell;
            cell.referenceAngle = readAngle(data, bytePosition);
            cell.referenceDistance = std::max(readDistance(data, bytePosition), parameters.cellMinDistance);
            cell.maxConnections = readByte(data, bytePosition) % (parameters.cellMaxBonds + 1);
            cell.executionOrderNumber = readByte(data, bytePosition) % parameters.cellMaxExecutionOrderNumbers;
            cell.color = readByte(data, bytePosition) % 7;
            cell.inputBlocked = readBool(data, bytePosition);
            cell.outputBlocked = readBool(data, bytePosition);

            switch (cellFunction) {
            case CellFunction_Neuron: {
                NeuronGenomeDescription neuron;
                for (int row = 0; row < MAX_CHANNELS; ++row) {
                    for (int col = 0; col < MAX_CHANNELS; ++col) {
                        neuron.weights[row][col] = readNeuronProperty(data, bytePosition);
                    }
                }
                for (int i = 0; i < MAX_CHANNELS; ++i) {
                    neuron.biases[i] = readNeuronProperty(data, bytePosition);
                }
                cell.cellFunction = neuron;
            } break;
            case CellFunction_Transmitter: {
                TransmitterGenomeDescription transmitter;
                transmitter.mode = readByte(data, bytePosition) % EnergyDistributionMode_Count;
                cell.cellFunction = transmitter;
            } break;
            case CellFunction_Constructor: {
                ConstructorGenomeDescription constructor;
                constructor.mode = readByte(data, bytePosition);
                constructor.singleConstruction = readBool(data, bytePosition);
                constructor.separateConstruction = readBool(data, bytePosition);
                constructor.adaptMaxConnections = readBool(data, bytePosition);
                constructor.angleAlignment = readByte(data, bytePosition) % ConstructorAngleAlignment_Count;
                constructor.stiffness = readStiffness(data, bytePosition);
                constructor.constructionActivationTime = readWord(data, bytePosition);
                constructor.genome = readGenome(data, bytePosition);
                cell.cellFunction = constructor;
            } break;
            case CellFunction_Sensor: {
                SensorGenomeDescription sensor;
                auto mode = readByte(data, bytePosition) % SensorMode_Count;
                auto angle = readAngle(data, bytePosition);
                if (mode == SensorMode_FixedAngle) {
                    sensor.fixedAngle = angle;
                }
                sensor.minDensity = readDensity(data, bytePosition);
                sensor.color = readByte(data, bytePosition) % MAX_COLORS;
                cell.cellFunction = sensor;
            } break;
            case CellFunction_Nerve: {
                NerveGenomeDescription nerve;
                nerve.pulseMode = readByte(data, bytePosition);
                nerve.alternationMode = readByte(data, bytePosition);
                cell.cellFunction = nerve;
            } break;
            case CellFunction_Attacker: {
                AttackerGenomeDescription attacker;
                attacker.mode = readByte(data, bytePosition) % EnergyDistributionMode_Count;
                cell.cellFunction = attacker;
            } break;
            case CellFunction_Injector: {
                InjectorGenomeDescription injector;
                injector.mode = readByte(data, bytePosition) % InjectorMode_Count;
                injector.genome = readGenome(data, bytePosition);
                cell.cellFunction = injector;
            } break;
            case CellFunction_Muscle: {
                MuscleGenomeDescription muscle;
                muscle.mode = readByte(data, bytePosition) % MuscleMode_Count;
                cell.cellFunction = muscle;
            } break;
            case CellFunction_Placeholder1: {
                cell.cellFunction = PlaceHolderGenomeDescription1();
            } break;
            case CellFunction_Placeholder2: {
                cell.cellFunction = PlaceHolderGenomeDescription2();
            } break;
            }
            result.genome.emplace_back(cell);
            ++cellIndex;
        }
        return result;
    }

}

GenomeDescription GenomeDescriptionConverter::convertBytesToDescription(std::vector<uint8_t> const& data, SimulationParameters const& parameters)
{
    return convertBytesToDescriptionIntern(data, parameters, data.size(), data.size()).genome;
}

int GenomeDescriptionConverter::convertByteIndexToCellIndex(std::vector<uint8_t> const& data, int byteIndex)
{
    //wasteful approach but sufficient for GUI
    return convertBytesToDescriptionIntern(data, SimulationParameters(), byteIndex, data.size()).genome.size();
}

int GenomeDescriptionConverter::convertCellIndexToByteIndex(std::vector<uint8_t> const& data, int cellIndex)
{
    //wasteful approach but sufficient for GUI
    return convertBytesToDescriptionIntern(data, SimulationParameters(), data.size(), cellIndex).lastBytePosition;
}

#include "GenomeDescriptionConverter.h"

#include <variant>

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
        data.emplace_back(static_cast<uint8_t>(static_cast<int8_t>((value - 1.0f) * 128)));
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
            auto size = readWord(data, pos) % MAX_GENOME_BYTES;
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
        case Enums::CellFunction_Neuron: {
            auto const& neuron = std::get<NeuronGenomeDescription>(*cell.cellFunction);
            for (int row = 0; row < MAX_CHANNELS; ++row) {
                for (int col = 0; col < MAX_CHANNELS; ++col) {
                    writeNeuronProperty(result, neuron.weights[row][col]);
                }
            }
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                writeNeuronProperty(result, neuron.bias[i]);
            }
        } break;
        case Enums::CellFunction_Transmitter: {
            auto const& transmitter = std::get<TransmitterGenomeDescription>(*cell.cellFunction);
            writeByte(result, transmitter.mode);
        } break;
        case Enums::CellFunction_Constructor: {
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
        case Enums::CellFunction_Sensor: {
            auto const& sensor = std::get<SensorGenomeDescription>(*cell.cellFunction);
            writeByte(result, sensor.fixedAngle.has_value() ? Enums::SensorMode_FixedAngle : Enums::SensorMode_Neighborhood);
            writeAngle(result, sensor.fixedAngle.has_value() ? *sensor.fixedAngle : 0.0f);
            writeDensity(result, sensor.minDensity);
            writeByte(result, sensor.color);
        } break;
        case Enums::CellFunction_Nerve: {
            auto const& nerve = std::get<NerveGenomeDescription>(*cell.cellFunction);
            writeByte(result, nerve.pulseMode);
            writeByte(result, nerve.alternationMode);
        } break;
        case Enums::CellFunction_Attacker: {
            auto const& attacker = std::get<AttackerGenomeDescription>(*cell.cellFunction);
            writeByte(result, attacker.mode);
        } break;
        case Enums::CellFunction_Injector: {
            auto const& injector = std::get<InjectorGenomeDescription>(*cell.cellFunction);
            writeGenome(result, injector.genome);
        } break;
        case Enums::CellFunction_Muscle: {
            auto const& muscle = std::get<MuscleGenomeDescription>(*cell.cellFunction);
            writeByte(result, muscle.mode);
        } break;
        case Enums::CellFunction_Placeholder1: {
        } break;
        case Enums::CellFunction_Placeholder2: {
        } break;
        }
        ++index;
    }
    return result;
}

GenomeDescription GenomeDescriptionConverter::convertBytesToDescription(std::vector<uint8_t> const& data, SimulationParameters const& parameters)
{
    int pos = 0;
    GenomeDescription result;
    while (pos < data.size()) {
        Enums::CellFunction cellFunction = readByte(data, pos) % Enums::CellFunction_Count;

        CellGenomeDescription cell;
        cell.referenceAngle = readAngle(data, pos);
        cell.referenceDistance = readFloat(data, pos) + 1.0f;
        cell.maxConnections = readByte(data, pos) % (parameters.cellMaxBonds + 1);
        cell.executionOrderNumber = readByte(data, pos) % parameters.cellMaxExecutionOrderNumbers;
        cell.color = readByte(data, pos) % 7;
        cell.inputBlocked = readBool(data, pos);
        cell.outputBlocked = readBool(data, pos);

        switch (cellFunction) {
        case Enums::CellFunction_Neuron: {
            NeuronGenomeDescription neuron;
            for (int row = 0; row < MAX_CHANNELS; ++row) {
                for (int col = 0; col < MAX_CHANNELS; ++col) {
                    neuron.weights[row][col] = readNeuronProperty(data, pos);
                }
            }
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                neuron.bias[i] = readNeuronProperty(data, pos);
            }
            cell.cellFunction = neuron;
        } break;
        case Enums::CellFunction_Transmitter: {
            TransmitterGenomeDescription transmitter;
            transmitter.mode = readByte(data, pos) % Enums::EnergyDistributionMode_Count;
            cell.cellFunction = transmitter;
        } break;
        case Enums::CellFunction_Constructor: {
            ConstructorGenomeDescription constructor;
            constructor.mode = readByte(data, pos);
            constructor.singleConstruction = readBool(data, pos);
            constructor.separateConstruction = readBool(data, pos);
            constructor.adaptMaxConnections = readBool(data, pos);
            constructor.angleAlignment = readByte(data, pos) % Enums::ConstructorAngleAlignment_Count;
            constructor.stiffness = readStiffness(data, pos);
            constructor.constructionActivationTime = readWord(data, pos);
            constructor.genome = readGenome(data, pos);
            cell.cellFunction = constructor;
        } break;
        case Enums::CellFunction_Sensor: {
            SensorGenomeDescription sensor;
            auto mode = readByte(data, pos) % Enums::SensorMode_Count;
            auto angle = readAngle(data, pos);
            if (mode == Enums::SensorMode_FixedAngle) {
                sensor.fixedAngle = angle;
            }
            sensor.minDensity = readDensity(data, pos);
            sensor.color = readByte(data, pos) % MAX_COLORS;
            cell.cellFunction = sensor;
        } break;
        case Enums::CellFunction_Nerve: {
            NerveGenomeDescription nerve;
            nerve.pulseMode = readByte(data, pos);
            nerve.alternationMode = readByte(data, pos);
            cell.cellFunction = nerve;
        } break;
        case Enums::CellFunction_Attacker: {
            AttackerGenomeDescription attacker;
            attacker.mode = readByte(data, pos) % Enums::EnergyDistributionMode_Count;
            cell.cellFunction = attacker;
        } break;
        case Enums::CellFunction_Injector: {
            InjectorGenomeDescription injector;
            injector.genome = readGenome(data, pos);
            cell.cellFunction = injector;
        } break;
        case Enums::CellFunction_Muscle: {
            MuscleGenomeDescription muscle;
            muscle.mode = readByte(data, pos) % Enums::MuscleMode_Count;
            cell.cellFunction = muscle;
        } break;
        case Enums::CellFunction_Placeholder1: {
            cell.cellFunction = PlaceHolderGenomeDescription1();
        } break;
        case Enums::CellFunction_Placeholder2: {
            cell.cellFunction = PlaceHolderGenomeDescription2();
        } break;
        }
        result.emplace_back(cell);
    };
    return result;
}

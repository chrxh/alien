#include "GenomeDescriptionConverter.h"

#include <variant>
#include <boost/range/adaptors.hpp>

#include "Base/Definitions.h"

namespace
{
    void writeInt(std::vector<uint8_t>& data, int value) {data.emplace_back(static_cast<uint8_t>(value)); }
    void writeAngle(std::vector<uint8_t>& data, float value) { data.emplace_back(static_cast<uint8_t>(static_cast<int8_t>(value / 180 * 128))); }
    void writeDistance(std::vector<uint8_t>& data, float value) {data.emplace_back(static_cast<uint8_t>(static_cast<int8_t>((value - 1.0f) * 128))); }
    void writeNeuronProperty(std::vector<uint8_t>& data, float value)
    {
        CHECK(std::abs(value) <= 2);
        data.emplace_back(static_cast<uint8_t>(static_cast<int8_t>(value / 2 * 128)));
    }
    void writeBool(std::vector<uint8_t>& data, bool value) {data.emplace_back(value ? 1 : 0); }
    void writeFloat(std::vector<uint8_t>& data, float value) { data.emplace_back(static_cast<uint8_t>(static_cast<int8_t>(value * 128))); }
    void writeWord(std::vector<uint8_t>& data, int value)
    {
        data.emplace_back(static_cast<uint8_t>(value & 0xff));
        data.emplace_back(static_cast<uint8_t>((value >> 8) % 0xff));
    }
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
    float readFloat(std::vector<uint8_t> const& data, int& pos)
    {
        return static_cast<float>(static_cast<int8_t>(readByte(data, pos))) / 128.0f;
    }

    float readAngle(std::vector<uint8_t> const& data, int& pos) { return readFloat(data, pos) * 180; }
    float readDensity(std::vector<uint8_t> const& data, int& pos) { return (readFloat(data, pos) + 1.0f) / 2; }

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
    for (auto const& [index, cell] : genome | boost::adaptors::indexed(0)) {
        writeInt(result, cell.getCellFunctionType());
        writeAngle(result, cell.referenceAngle);
        writeDistance(result, cell.referenceDistance);
        writeInt(result, cell.maxConnections);
        writeInt(result, cell.executionOrderNumber);
        writeInt(result, cell.color);
        writeBool(result, cell.inputBlocked);
        writeBool(result, cell.outputBlocked);
        switch (cell.getCellFunctionType()) {
        case Enums::CellFunction_Neuron: {
            auto neuron = std::get<NeuronGenomeDescription>(*cell.cellFunction);
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
        } break;
        case Enums::CellFunction_Constructor: {
            auto constructor = std::get<ConstructorGenomeDescription>(*cell.cellFunction);
            writeInt(result, constructor.mode);
            writeBool(result, constructor.singleConstruction);
            writeBool(result, constructor.separateConstruction);
            writeBool(result, constructor.makeSticky);
            writeInt(result, constructor.angleAlignment);
            writeGenome(result, constructor.genome);
        } break;
        case Enums::CellFunction_Sensor: {
            auto sensor = std::get<SensorGenomeDescription>(*cell.cellFunction);
            writeInt(result, sensor.fixedAngle.has_value() ? Enums::SensorMode_FixedAngle : Enums::SensorMode_Neighborhood);
            writeAngle(result, sensor.fixedAngle.has_value() ? *sensor.fixedAngle : 0.0f);
            writeFloat(result, sensor.minDensity * 2 - 1.0f);
            writeInt(result, sensor.color);
        } break;
        case Enums::CellFunction_Nerve: {
        } break;
        case Enums::CellFunction_Attacker: {
        } break;
        case Enums::CellFunction_Injector: {
            auto injector = std::get<InjectorGenomeDescription>(*cell.cellFunction);
            writeGenome(result, injector.genome);
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

GenomeDescription GenomeDescriptionConverter::convertBytesToDescription(std::vector<uint8_t> const& data, SimulationParameters const& parameters)
{
    int pos = 0;
    GenomeDescription result;
    do {
        Enums::CellFunction cellFunction = readByte(data, pos) % Enums::CellFunction_Count;

        CellGenomeDescription cell;
        cell.referenceAngle = readFloat(data, pos) * 180;
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
                    neuron.weights[row][col] = readFloat(data, pos) * 2;
                }
            }
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                neuron.bias[i] = readFloat(data, pos) * 2;
            }
            cell.cellFunction = neuron;
        } break;
        case Enums::CellFunction_Transmitter: {
            cell.cellFunction = TransmitterGenomeDescription();
        } break;
        case Enums::CellFunction_Constructor: {
            ConstructorGenomeDescription constructor;
            constructor.mode = readByte(data, pos);
            constructor.singleConstruction = readBool(data, pos);
            constructor.separateConstruction = readBool(data, pos);
            constructor.makeSticky = readBool(data, pos);
            constructor.angleAlignment = readByte(data, pos) % 7;
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
        } break;
        case Enums::CellFunction_Nerve: {
            cell.cellFunction = NerveGenomeDescription();
        } break;
        case Enums::CellFunction_Attacker: {
            cell.cellFunction = AttackerGenomeDescription();
        } break;
        case Enums::CellFunction_Injector: {
            InjectorGenomeDescription injector;
            injector.genome = readGenome(data, pos);
            cell.cellFunction = injector;
        } break;
        case Enums::CellFunction_Muscle: {
            cell.cellFunction = MuscleGenomeDescription();
        } break;
        case Enums::CellFunction_Placeholder1: {
            cell.cellFunction = PlaceHolderGenomeDescription1();
        } break;
        case Enums::CellFunction_Placeholder2: {
            cell.cellFunction = PlaceHolderGenomeDescription2();
        } break;
        }
        result.emplace_back(cell);
    } while (pos < data.size());
    return result;
}

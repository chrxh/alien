#pragma once

#include <functional>
#include <variant>

#include <Base/Hashes.h>

template <>
struct std::hash<NeuralNetGenomeDesc>
{
    std::size_t operator()(NeuralNetGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        //for (const auto& weight : desc._weights) {
        //    hash_combine(seed, weight);
        //}
        for (const auto& bias : desc._biases) {
            hash_combine(seed, bias);
        }
        for (const auto& func : desc._activationFunctions) {
            hash_combine(seed, static_cast<int>(func));
        }
        for (const auto& connectionWeight : desc._connectionWeights) {
            hash_combine(seed, connectionWeight);
        }
        return seed;
    }
};

template <>
struct std::hash<BaseGenomeDesc>
{
    std::size_t operator()(BaseGenomeDesc const& desc) const { return 0; }
};

template <>
struct std::hash<DepotGenomeDesc>
{
    std::size_t operator()(DepotGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._storageLimit);
        hash_combine(seed, desc._initialStoredUsableEnergy);
        return seed;
    }
};

template <>
struct std::hash<ConstructorGenomeDesc>
{
    std::size_t operator()(ConstructorGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        if (desc._autoTriggerInterval) {
            hash_combine(seed, *desc._autoTriggerInterval);
        } else {
            hash_combine(seed, -1);
        }
        hash_combine(seed, desc._geneIndex);
        hash_combine(seed, desc._constructionActivationTime);
        hash_combine(seed, desc._reservedEnergy);
        hash_combine(seed, desc._separation);
        hash_combine(seed, desc._numBranches);
        hash_combine(seed, desc._numConcatenations);
        return seed;
    }
};

template <>
struct std::hash<SensorGenomeDesc>
{
    std::size_t operator()(SensorGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._autoTrigger);
        hash_combine(seed, desc._tagForAttackers);
        hash_combine(seed, desc._minRange);
        hash_combine(seed, desc._maxRange);
        hash_combine(seed, desc.getMode());

        // Hash mode-specific data
        if (desc.getMode() == SensorMode_DetectEnergy) {
            auto const& mode = std::get<DetectEnergyGenomeDesc>(desc._mode);
            hash_combine(seed, mode._minDensity);
        } else if (desc.getMode() == SensorMode_DetectSolid) {
            // No additional data
        } else if (desc.getMode() == SensorMode_DetectFreeCell) {
            auto const& mode = std::get<DetectFreeCellGenomeDesc>(desc._mode);
            hash_combine(seed, mode._minDensity);
            hash_combine(seed, mode._restrictToColors);
        } else if (desc.getMode() == SensorMode_DetectCreature) {
            auto const& mode = std::get<DetectCreatureGenomeDesc>(desc._mode);
            if (mode._minNumCells) {
                hash_combine(seed, *mode._minNumCells);
            } else {
                hash_combine(seed, -1);
            }
            if (mode._maxNumCells) {
                hash_combine(seed, *mode._maxNumCells);
            } else {
                hash_combine(seed, -1);
            }
            hash_combine(seed, mode._restrictToColors);
            hash_combine(seed, static_cast<int>(mode._restrictToLineage));
        }
        return seed;
    }
};

template <>
struct std::hash<SquareSignalGenomeDesc>
{
    std::size_t operator()(SquareSignalGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._period);
        return seed;
    }
};

template <>
struct std::hash<SawtoothSignalGenomeDesc>
{
    std::size_t operator()(SawtoothSignalGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._period);
        return seed;
    }
};

template <>
struct std::hash<GeneratorModeGenomeDesc>
{
    std::size_t operator()(GeneratorModeGenomeDesc const& desc) const { return variant_hasher<SquareSignalGenomeDesc, SawtoothSignalGenomeDesc>{}(desc); }
};

template <>
struct std::hash<GeneratorGenomeDesc>
{
    std::size_t operator()(GeneratorGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._additive);
        hash_combine(seed, desc._minValue);
        hash_combine(seed, desc._maxValue);
        hash_combine(seed, desc._timeOffset);
        hash_combine(seed, std::hash<GeneratorModeGenomeDesc>{}(desc._mode));
        return seed;
    }
};

template <>
struct std::hash<AttackFreeCellGenomeDesc>
{
    std::size_t operator()(AttackFreeCellGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._restrictToColors);
        return seed;
    }
};

template <>
struct std::hash<AttackCreatureGenomeDesc>
{
    std::size_t operator()(AttackCreatureGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        return seed;
    }
};

template <>
struct std::hash<AttackerModeGenomeDesc>
{
    std::size_t operator()(AttackerModeGenomeDesc const& desc) const { return variant_hasher<AttackFreeCellGenomeDesc, AttackCreatureGenomeDesc>{}(desc); }
};

template <>
struct std::hash<AttackerGenomeDesc>
{
    std::size_t operator()(AttackerGenomeDesc const& desc) const { return std::hash<AttackerModeGenomeDesc>{}(desc._mode); }
};

template <>
struct std::hash<InjectorGenomeDesc>
{
    std::size_t operator()(InjectorGenomeDesc const& desc) const { return std::hash<int>{}(desc._geneIndex); }
};

template <>
struct std::hash<AutoBendingGenomeDesc>
{
    std::size_t operator()(AutoBendingGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxAngleDeviation);
        hash_combine(seed, desc._forwardBackwardRatio);
        return seed;
    }
};

template <>
struct std::hash<ManualBendingGenomeDesc>
{
    std::size_t operator()(ManualBendingGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxAngleDeviation);
        hash_combine(seed, desc._forwardBackwardRatio);
        return seed;
    }
};

template <>
struct std::hash<AngleBendingGenomeDesc>
{
    std::size_t operator()(AngleBendingGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxAngleDeviation);
        hash_combine(seed, desc._attractionRepulsionRatio);
        return seed;
    }
};

template <>
struct std::hash<AutoCrawlingGenomeDesc>
{
    std::size_t operator()(AutoCrawlingGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxDistanceDeviation);
        hash_combine(seed, desc._forwardBackwardRatio);
        return seed;
    }
};

template <>
struct std::hash<ManualCrawlingGenomeDesc>
{
    std::size_t operator()(ManualCrawlingGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxDistanceDeviation);
        hash_combine(seed, desc._forwardBackwardRatio);
        return seed;
    }
};

template <>
struct std::hash<DirectMovementGenomeDesc>
{
    std::size_t operator()(DirectMovementGenomeDesc const& desc) const { return 2; }
};

template <>
struct std::hash<MuscleModeGenomeDesc>
{
    std::size_t operator()(MuscleModeGenomeDesc const& desc) const
    {
        return variant_hasher<
            AutoBendingGenomeDesc,
            ManualBendingGenomeDesc,
            AngleBendingGenomeDesc,
            AutoCrawlingGenomeDesc,
            ManualCrawlingGenomeDesc,
            DirectMovementGenomeDesc>{}(desc);
    }
};

template <>
struct std::hash<MuscleGenomeDesc>
{
    std::size_t operator()(MuscleGenomeDesc const& desc) const { return std::hash<MuscleModeGenomeDesc>{}(desc._mode); }
};

template <>
struct std::hash<DefenderGenomeDesc>
{
    std::size_t operator()(DefenderGenomeDesc const& desc) const { return std::hash<int>{}(static_cast<int>(desc._mode)); }
};

template <>
struct std::hash<ReconnectSolidGenomeDesc>
{
    std::size_t operator()(ReconnectSolidGenomeDesc const& desc) const { return 0; }
};

template <>
struct std::hash<ReconnectFreeCellGenomeDesc>
{
    std::size_t operator()(ReconnectFreeCellGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._restrictToColors);
        return seed;
    }
};

template <>
struct std::hash<ReconnectCreatureGenomeDesc>
{
    std::size_t operator()(ReconnectCreatureGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        if (desc._minNumCells) {
            hash_combine(seed, *desc._minNumCells);
        } else {
            hash_combine(seed, -1);
        }
        if (desc._maxNumCells) {
            hash_combine(seed, *desc._maxNumCells);
        } else {
            hash_combine(seed, -1);
        }
        hash_combine(seed, desc._restrictToColors);
        hash_combine(seed, static_cast<int>(desc._restrictToLineage));
        return seed;
    }
};

template <>
struct std::hash<ReconnectorModeGenomeDesc>
{
    std::size_t operator()(ReconnectorModeGenomeDesc const& desc) const
    {
        return variant_hasher<ReconnectSolidGenomeDesc, ReconnectFreeCellGenomeDesc, ReconnectCreatureGenomeDesc>{}(desc);
    }
};

template <>
struct std::hash<ReconnectorGenomeDesc>
{
    std::size_t operator()(ReconnectorGenomeDesc const& desc) const { return std::hash<ReconnectorModeGenomeDesc>{}(desc._mode); }
};

template <>
struct std::hash<DetonatorGenomeDesc>
{
    std::size_t operator()(DetonatorGenomeDesc const& desc) const { return std::hash<int>{}(desc._countdown); }
};

template <>
struct std::hash<DigestorGenomeDesc>
{
    std::size_t operator()(DigestorGenomeDesc const& desc) const { return 0; }
};

template <>
struct std::hash<SignalDelayGenomeDesc>
{
    std::size_t operator()(SignalDelayGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._delay);
        return seed;
    }
};

template <>
struct std::hash<SignalRecorderGenomeDesc>
{
    std::size_t operator()(SignalRecorderGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._readOnly);
        hash_combine(seed, desc._numWrittenSignalEntries);
        return seed;
    }
};

template <>
struct std::hash<SignalStorageGenomeDesc>
{
    std::size_t operator()(SignalStorageGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._readOnly);
        return seed;
    }
};

template <>
struct std::hash<SignalIntegratorGenomeDesc>
{
    std::size_t operator()(SignalIntegratorGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._newSignalWeight);
        return seed;
    }
};

template <>
struct std::hash<MemoryModeGenomeDesc>
{
    std::size_t operator()(MemoryModeGenomeDesc const& desc) const
    {
        return variant_hasher<SignalDelayGenomeDesc, SignalRecorderGenomeDesc, SignalStorageGenomeDesc, SignalIntegratorGenomeDesc>{}(desc);
    }
};

template <>
struct std::hash<SignalEntryGenomeDesc>
{
    std::size_t operator()(SignalEntryGenomeDesc const& desc) const
    {
        std::size_t result = 0;
        for (auto const& channel : desc._channels) {
            hash_combine(result, channel);
        }
        return result;
    }
};

template <>
struct std::hash<MemoryGenomeDesc>
{
    std::size_t operator()(MemoryGenomeDesc const& desc) const
    {
        std::size_t result = std::hash<MemoryModeGenomeDesc>{}(desc._mode);
        for (auto const& entry : desc._signalEntries) {
            hash_combine(result, std::hash<SignalEntryGenomeDesc>{}(entry));
        }
        hash_combine(result, desc._channelBitMask);
        return result;
    }
};

template <>
struct std::hash<SenderGenomeDesc>
{
    std::size_t operator()(SenderGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._range);
        hash_combine(seed, desc._maxTimesSent);
        return seed;
    }
};

template <>
struct std::hash<ReceiverGenomeDesc>
{
    std::size_t operator()(ReceiverGenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._restrictToColors);
        hash_combine(seed, static_cast<int>(desc._restrictToLineage));
        return seed;
    }
};

template <>
struct std::hash<CommunicatorModeGenomeDesc>
{
    std::size_t operator()(CommunicatorModeGenomeDesc const& desc) const { return variant_hasher<SenderGenomeDesc, ReceiverGenomeDesc>{}(desc); }
};

template <>
struct std::hash<CommunicatorGenomeDesc>
{
    std::size_t operator()(CommunicatorGenomeDesc const& desc) const { return std::hash<CommunicatorModeGenomeDesc>{}(desc._mode); }
};

template <>
struct std::hash<VoidGenomeDesc>
{
    std::size_t operator()(VoidGenomeDesc const& desc) const { return 0; }
};

template <>
struct std::hash<CellTypeGenomeDesc>
{
    std::size_t operator()(CellTypeGenomeDesc const& desc) const
    {
        return variant_hasher<
            BaseGenomeDesc,
            DepotGenomeDesc,
            SensorGenomeDesc,
            GeneratorGenomeDesc,
            AttackerGenomeDesc,
            InjectorGenomeDesc,
            MuscleGenomeDesc,
            DefenderGenomeDesc,
            ReconnectorGenomeDesc,
            DetonatorGenomeDesc,
            DigestorGenomeDesc,
            MemoryGenomeDesc,
            CommunicatorGenomeDesc,
            VoidGenomeDesc>{}(desc);
    }
};

template <>
struct std::hash<NodeDesc>
{
    std::size_t operator()(NodeDesc const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._referenceAngle);
        hash_combine(seed, desc._color);
        hash_combine(seed, std::hash<NeuralNetGenomeDesc>{}(desc._neuralNetwork));
        hash_combine(seed, std::hash<CellTypeGenomeDesc>{}(desc._cellType));
        if (desc._constructor.has_value()) {
            hash_combine(seed, std::hash<ConstructorGenomeDesc>{}(desc._constructor.value()));
        } else {
            hash_combine(seed, -1);
        }
        return seed;
    }
};

template <>
struct std::hash<GeneDesc>
{
    std::size_t operator()(GeneDesc const& desc) const
    {
        std::size_t seed = 0;
        for (auto const& node : desc._nodes) {
            hash_combine(seed, std::hash<NodeDesc>{}(node));
        }
        hash_combine(seed, static_cast<int>(desc._shape));
        hash_combine(seed, desc._stiffness);
        hash_combine(seed, desc._connectionDistance);
        hash_combine(seed, desc._homogeneCellType);
        return seed;
    }
};

template <>
struct std::hash<GenomeDesc>
{
    std::size_t operator()(GenomeDesc const& desc) const
    {
        std::size_t seed = 0;
        for (auto const& gene : desc._genes) {
            hash_combine(seed, std::hash<GeneDesc>{}(gene));
        }
        hash_combine(seed, desc._frontAngle);
        hash_combine(seed, desc._accumulatedMutations);
        hash_combine(seed, desc._prevLineageId);
        for (int i = 0; i < 2; ++i) {
            hash_combine(seed, desc._mutationRates._neuronMutations[i]._eventProbability);
            hash_combine(seed, desc._mutationRates._neuronMutations[i]._weightSigma);
            hash_combine(seed, desc._mutationRates._neuronMutations[i]._biasSigma);
            hash_combine(seed, desc._mutationRates._neuronMutations[i]._activationFunctionProbability);
            hash_combine(seed, desc._mutationRates._connectionMutations[i]._eventProbability);
            hash_combine(seed, desc._mutationRates._connectionMutations[i]._sigma);
            hash_combine(seed, desc._mutationRates._cellTypePropertiesMutations[i]._eventProbability);
            hash_combine(seed, desc._mutationRates._cellTypePropertiesMutations[i]._sigma);
            hash_combine(seed, desc._mutationRates._cellTypePropertiesMutations[i]._probability);
        }
        hash_combine(seed, desc._mutationRates._cellTypeModeMutation._eventProbability);
        hash_combine(seed, desc._mutationRates._cellTypeMutation._eventProbability);
        return seed;
    }
};

template <>
struct std::hash<SubGenomeDesc>
{
    std::size_t operator()(SubGenomeDesc const& genomeWithRootIndex) const
    {
        std::size_t seed = 0;
        hash_combine(seed, genomeWithRootIndex.genome);
        hash_combine(seed, genomeWithRootIndex.trimmed);
        hash_combine(seed, genomeWithRootIndex.startIndex);
        return seed;
    }
};

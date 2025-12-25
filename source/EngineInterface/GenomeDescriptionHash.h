#pragma once

#include <functional>
#include <variant>

#include <Base/Hashes.h>

#include "GenomeDescription.h"

template <>
struct std::hash<NeuralNetworkGenomeDescription>
{
    std::size_t operator()(NeuralNetworkGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        for (const auto& weight : desc._weights) {
            hash_combine(seed, weight);
        }
        for (const auto& bias : desc._biases) {
            hash_combine(seed, bias);
        }
        for (const auto& func : desc._activationFunctions) {
            hash_combine(seed, static_cast<int>(func));
        }
        return seed;
    }
};

template <>
struct std::hash<BaseGenomeDescription>
{
    std::size_t operator()(BaseGenomeDescription const& desc) const { return 0; }
};

template <>
struct std::hash<DepotGenomeDescription>
{
    std::size_t operator()(DepotGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._storageLimit);
        hash_combine(seed, desc._initialStoredUsableEnergy);
        return seed;
    }
};

template <>
struct std::hash<ConstructorGenomeDescription>
{
    std::size_t operator()(ConstructorGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        if (desc._autoTriggerInterval) {
            hash_combine(seed, *desc._autoTriggerInterval);
        } else {
            hash_combine(seed, -1);
        }
        hash_combine(seed, desc._geneIndex);
        hash_combine(seed, desc._constructionActivationTime);
        return seed;
    }
};

template <>
struct std::hash<SensorGenomeDescription>
{
    std::size_t operator()(SensorGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        if (desc._autoTriggerInterval) {
            hash_combine(seed, *desc._autoTriggerInterval);
        } else {
            hash_combine(seed, -1);
        }
        hash_combine(seed, desc._minRange);
        hash_combine(seed, desc._maxRange);
        hash_combine(seed, desc.getMode());
        
        // Hash mode-specific data
        if (desc.getMode() == SensorMode_DetectEnergy) {
            auto const& mode = std::get<DetectEnergyGenomeDescription>(desc._mode);
            hash_combine(seed, mode._minDensity);
        } else if (desc.getMode() == SensorMode_DetectStructure) {
            // No additional data
        } else if (desc.getMode() == SensorMode_DetectFreeCell) {
            auto const& mode = std::get<DetectFreeCellGenomeDescription>(desc._mode);
            hash_combine(seed, mode._minDensity);
            if (mode._restrictToColor) {
                hash_combine(seed, *mode._restrictToColor);
            } else {
                hash_combine(seed, -1);
            }
        } else if (desc.getMode() == SensorMode_DetectCreature) {
            auto const& mode = std::get<DetectCreatureGenomeDescription>(desc._mode);
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
            if (mode._restrictToColor) {
                hash_combine(seed, *mode._restrictToColor);
            } else {
                hash_combine(seed, -1);
            }
            hash_combine(seed, static_cast<int>(mode._restrictToLineage));
        }
        return seed;
    }
};

template <>
struct std::hash<GeneratorGenomeDescription>
{
    std::size_t operator()(GeneratorGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._autoTriggerInterval);
        hash_combine(seed, static_cast<int>(desc._pulseType));
        hash_combine(seed, desc._alternationInterval);
        return seed;
    }
};

template <>
struct std::hash<AttackFreeCellGenomeDescription>
{
    std::size_t operator()(AttackFreeCellGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        if (desc._restrictToColor) {
            hash_combine(seed, *desc._restrictToColor);
        } else {
            hash_combine(seed, -1);
        }
        return seed;
    }
};

template <>
struct std::hash<AttackCreatureGenomeDescription>
{
    std::size_t operator()(AttackCreatureGenomeDescription const& desc) const
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
        if (desc._restrictToColor) {
            hash_combine(seed, *desc._restrictToColor);
        } else {
            hash_combine(seed, -1);
        }
        hash_combine(seed, static_cast<int>(desc._restrictToLineage));
        return seed;
    }
};

template <>
struct std::hash<AttackerModeGenomeDescription>
{
    std::size_t operator()(AttackerModeGenomeDescription const& desc) const
    {
        return variant_hasher<AttackFreeCellGenomeDescription, AttackCreatureGenomeDescription>{}(desc);
    }
};

template <>
struct std::hash<AttackerGenomeDescription>
{
    std::size_t operator()(AttackerGenomeDescription const& desc) const
    {
        return std::hash<AttackerModeGenomeDescription>{}(desc._mode);
    }
};

template <>
struct std::hash<InjectorGenomeDescription>
{
    std::size_t operator()(InjectorGenomeDescription const& desc) const { return std::hash<int>{}(desc._geneIndex); }
};

template <>
struct std::hash<AutoBendingGenomeDescription>
{
    std::size_t operator()(AutoBendingGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxAngleDeviation);
        hash_combine(seed, desc._forwardBackwardRatio);
        return seed;
    }
};

template <>
struct std::hash<ManualBendingGenomeDescription>
{
    std::size_t operator()(ManualBendingGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxAngleDeviation);
        hash_combine(seed, desc._forwardBackwardRatio);
        return seed;
    }
};

template <>
struct std::hash<AngleBendingGenomeDescription>
{
    std::size_t operator()(AngleBendingGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxAngleDeviation);
        hash_combine(seed, desc._attractionRepulsionRatio);
        return seed;
    }
};

template <>
struct std::hash<AutoCrawlingGenomeDescription>
{
    std::size_t operator()(AutoCrawlingGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxDistanceDeviation);
        hash_combine(seed, desc._forwardBackwardRatio);
        return seed;
    }
};

template <>
struct std::hash<ManualCrawlingGenomeDescription>
{
    std::size_t operator()(ManualCrawlingGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxDistanceDeviation);
        hash_combine(seed, desc._forwardBackwardRatio);
        return seed;
    }
};

template <>
struct std::hash<DirectMovementGenomeDescription>
{
    std::size_t operator()(DirectMovementGenomeDescription const& desc) const { return 2; }
};

template <>
struct std::hash<MuscleModeGenomeDescription>
{
    std::size_t operator()(MuscleModeGenomeDescription const& desc) const
    {
        return variant_hasher<
            AutoBendingGenomeDescription,
            ManualBendingGenomeDescription,
            AngleBendingGenomeDescription,
            AutoCrawlingGenomeDescription,
            ManualCrawlingGenomeDescription,
            DirectMovementGenomeDescription>{}(desc);
    }
};

template <>
struct std::hash<MuscleGenomeDescription>
{
    std::size_t operator()(MuscleGenomeDescription const& desc) const { return std::hash<MuscleModeGenomeDescription>{}(desc._mode); }
};

template <>
struct std::hash<DefenderGenomeDescription>
{
    std::size_t operator()(DefenderGenomeDescription const& desc) const { return std::hash<int>{}(static_cast<int>(desc._mode)); }
};

template <>
struct std::hash<ReconnectStructureGenomeDescription>
{
    std::size_t operator()(ReconnectStructureGenomeDescription const& desc) const { return 0; }
};

template <>
struct std::hash<ReconnectFreeCellGenomeDescription>
{
    std::size_t operator()(ReconnectFreeCellGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        if (desc._restrictToColor) {
            hash_combine(seed, *desc._restrictToColor);
        } else {
            hash_combine(seed, -1);
        }
        return seed;
    }
};

template <>
struct std::hash<ReconnectCreatureGenomeDescription>
{
    std::size_t operator()(ReconnectCreatureGenomeDescription const& desc) const
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
        if (desc._restrictToColor) {
            hash_combine(seed, *desc._restrictToColor);
        } else {
            hash_combine(seed, -1);
        }
        hash_combine(seed, static_cast<int>(desc._restrictToLineage));
        return seed;
    }
};

template <>
struct std::hash<ReconnectorModeGenomeDescription>
{
    std::size_t operator()(ReconnectorModeGenomeDescription const& desc) const
    {
        return variant_hasher<ReconnectStructureGenomeDescription, ReconnectFreeCellGenomeDescription, ReconnectCreatureGenomeDescription>{}(desc);
    }
};

template <>
struct std::hash<ReconnectorGenomeDescription>
{
    std::size_t operator()(ReconnectorGenomeDescription const& desc) const { return std::hash<ReconnectorModeGenomeDescription>{}(desc._mode); }
};

template <>
struct std::hash<DetonatorGenomeDescription>
{
    std::size_t operator()(DetonatorGenomeDescription const& desc) const { return std::hash<int>{}(desc._countdown); }
};

template <>
struct std::hash<DigestorGenomeDescription>
{
    std::size_t operator()(DigestorGenomeDescription const& desc) const { return 0; }
};

template <>
struct std::hash<SignalDelayGenomeDescription>
{
    std::size_t operator()(SignalDelayGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._delayWithRecording);
        hash_combine(seed, desc._delayWithoutRecording);
        return seed;
    }
};

template <>
struct std::hash<SignalRecorderGenomeDescription>
{
    std::size_t operator()(SignalRecorderGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._readOnly);
        hash_combine(seed, desc._numEntries);
        return seed;
    }
};

template <>
struct std::hash<SignalStorageGenomeDescription>
{
    std::size_t operator()(SignalStorageGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._numEntries);
        return seed;
    }
};

template <>
struct std::hash<SignalIntegratorGenomeDescription>
{
    std::size_t operator()(SignalIntegratorGenomeDescription const& desc) const { return 0; }
};

template <>
struct std::hash<MemoryModeGenomeDescription>
{
    std::size_t operator()(MemoryModeGenomeDescription const& desc) const
    {
        return variant_hasher<SignalDelayGenomeDescription, SignalRecorderGenomeDescription, SignalStorageGenomeDescription, SignalIntegratorGenomeDescription>{}(desc);
    }
};

template <>
struct std::hash<MemoryEntryGenomeDescription>
{
    std::size_t operator()(MemoryEntryGenomeDescription const& desc) const
    {
        std::size_t result = 0;
        for (auto const& channel : desc._channels) {
            hash_combine(result, channel);
        }
        return result;
    }
};

template <>
struct std::hash<MemoryGenomeDescription>
{
    std::size_t operator()(MemoryGenomeDescription const& desc) const
    {
        std::size_t result = std::hash<MemoryModeGenomeDescription>{}(desc._mode);
        for (auto const& entry : desc._memoryEntries) {
            hash_combine(result, std::hash<MemoryEntryGenomeDescription>{}(entry));
        }
        return result;
    }
};

template <>
struct std::hash<CellTypeGenomeDescription>
{
    std::size_t operator()(CellTypeGenomeDescription const& desc) const
    {
        return variant_hasher<
            BaseGenomeDescription,
            DepotGenomeDescription,
            ConstructorGenomeDescription,
            SensorGenomeDescription,
            GeneratorGenomeDescription,
            AttackerGenomeDescription,
            InjectorGenomeDescription,
            MuscleGenomeDescription,
            DefenderGenomeDescription,
            ReconnectorGenomeDescription,
            DetonatorGenomeDescription,
            DigestorGenomeDescription,
            MemoryGenomeDescription>{}(desc);
    }
};

template <>
struct std::hash<SignalRestrictionGenomeDescription>
{
    std::size_t operator()(SignalRestrictionGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._active);
        hash_combine(seed, desc._baseAngle);
        hash_combine(seed, desc._openingAngle);
        return seed;
    }
};

template <>
struct std::hash<NodeDescription>
{
    std::size_t operator()(NodeDescription const& desc) const
    {
        std::size_t seed = 0;
        hash_combine(seed, desc._referenceAngle);
        hash_combine(seed, desc._color);
        hash_combine(seed, desc._numAdditionalConnections);
        hash_combine(seed, std::hash<NeuralNetworkGenomeDescription>{}(desc._neuralNetwork));
        hash_combine(seed, std::hash<CellTypeGenomeDescription>{}(desc._cellType));
        hash_combine(seed, std::hash<SignalRestrictionGenomeDescription>{}(desc._signalRestriction));
        return seed;
    }
};

template <>
struct std::hash<GeneDescription>
{
    std::size_t operator()(GeneDescription const& desc) const
    {
        std::size_t seed = 0;
        for (auto const& node : desc._nodes) {
            hash_combine(seed, std::hash<NodeDescription>{}(node));
        }
        hash_combine(seed, static_cast<int>(desc._shape));
        hash_combine(seed, desc._separation);
        hash_combine(seed, desc._numBranches);
        hash_combine(seed, desc._numConcatenations);
        hash_combine(seed, static_cast<int>(desc._angleAlignment));
        hash_combine(seed, desc._stiffness);
        hash_combine(seed, desc._connectionDistance);
        return seed;
    }
};

template <>
struct std::hash<GenomeDescription>
{
    std::size_t operator()(GenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        for (auto const& gene : desc._genes) {
            hash_combine(seed, std::hash<GeneDescription>{}(gene));
        }
        hash_combine(seed, desc._frontAngle);
        return seed;
    }
};

template <>
struct std::hash<SubGenomeDescription>
{
    std::size_t operator()(SubGenomeDescription const& genomeWithRootIndex) const
    {
        std::size_t seed = 0;
        hash_combine(seed, genomeWithRootIndex.genome);
        hash_combine(seed, genomeWithRootIndex.trimmed);
        hash_combine(seed, genomeWithRootIndex.startIndex);
        return seed;
    }
};

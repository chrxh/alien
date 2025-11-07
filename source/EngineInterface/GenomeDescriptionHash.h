#pragma once

#include <functional>
#include <variant>

#include <boost/container_hash/hash_fwd.hpp>

#include <Base/Hashes.h>

#include "GenomeDescription.h"

template <>
struct std::hash<NeuralNetworkGenomeDescription>
{
    std::size_t operator()(NeuralNetworkGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        for (const auto& weight : desc._weights) {
            boost::hash_combine(seed, weight);
        }
        for (const auto& bias : desc._biases) {
            boost::hash_combine(seed, bias);
        }
        for (const auto& func : desc._activationFunctions) {
            boost::hash_combine(seed, static_cast<int>(func));
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
    std::size_t operator()(DepotGenomeDescription const& desc) const { return std::hash<int>{}(static_cast<int>(desc._mode)); }
};

template <>
struct std::hash<ConstructorGenomeDescription>
{
    std::size_t operator()(ConstructorGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        if (desc._autoTriggerInterval) {
            boost::hash_combine(seed, *desc._autoTriggerInterval);
        } else {
            boost::hash_combine(seed, -1);
        }
        boost::hash_combine(seed, desc._geneIndex);
        boost::hash_combine(seed, desc._constructionActivationTime);
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
            boost::hash_combine(seed, *desc._autoTriggerInterval);
        } else {
            boost::hash_combine(seed, -1);
        }
        boost::hash_combine(seed, desc._minDensity);
        if (desc._minRange) {
            boost::hash_combine(seed, *desc._minRange);
        } else {
            boost::hash_combine(seed, -1);
        }
        if (desc._maxRange) {
            boost::hash_combine(seed, *desc._maxRange);
        } else {
            boost::hash_combine(seed, -1);
        }
        if (desc._restrictToColor) {
            boost::hash_combine(seed, *desc._restrictToColor);
        } else {
            boost::hash_combine(seed, -1);
        }
        boost::hash_combine(seed, static_cast<int>(desc._restrictToCreatures));
        return seed;
    }
};

template <>
struct std::hash<GeneratorGenomeDescription>
{
    std::size_t operator()(GeneratorGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, desc._autoTriggerInterval);
        boost::hash_combine(seed, static_cast<int>(desc._pulseType));
        boost::hash_combine(seed, desc._alternationInterval);
        return seed;
    }
};

template <>
struct std::hash<AttackerGenomeDescription>
{
    std::size_t operator()(AttackerGenomeDescription const& desc) const { return 1; }
};

template <>
struct std::hash<InjectorGenomeDescription>
{
    std::size_t operator()(InjectorGenomeDescription const& desc) const { return std::hash<int>{}(static_cast<int>(desc._mode)); }
};

template <>
struct std::hash<AutoBendingGenomeDescription>
{
    std::size_t operator()(AutoBendingGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, desc._maxAngleDeviation);
        boost::hash_combine(seed, desc._forwardBackwardRatio);
        return seed;
    }
};

template <>
struct std::hash<ManualBendingGenomeDescription>
{
    std::size_t operator()(ManualBendingGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, desc._maxAngleDeviation);
        boost::hash_combine(seed, desc._forwardBackwardRatio);
        return seed;
    }
};

template <>
struct std::hash<AngleBendingGenomeDescription>
{
    std::size_t operator()(AngleBendingGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, desc._maxAngleDeviation);
        boost::hash_combine(seed, desc._attractionRepulsionRatio);
        return seed;
    }
};

template <>
struct std::hash<AutoCrawlingGenomeDescription>
{
    std::size_t operator()(AutoCrawlingGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, desc._maxDistanceDeviation);
        boost::hash_combine(seed, desc._forwardBackwardRatio);
        return seed;
    }
};

template <>
struct std::hash<ManualCrawlingGenomeDescription>
{
    std::size_t operator()(ManualCrawlingGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, desc._maxDistanceDeviation);
        boost::hash_combine(seed, desc._forwardBackwardRatio);
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
struct std::hash<ReconnectorGenomeDescription>
{
    std::size_t operator()(ReconnectorGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        if (desc._restrictToColor) {
            boost::hash_combine(seed, *desc._restrictToColor);
        } else {
            boost::hash_combine(seed, -1);
        }
        boost::hash_combine(seed, static_cast<int>(desc._restrictToCreatures));
        return seed;
    }
};

template <>
struct std::hash<DetonatorGenomeDescription>
{
    std::size_t operator()(DetonatorGenomeDescription const& desc) const { return std::hash<int>{}(desc._countdown); }
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
            DetonatorGenomeDescription>{}(desc);
    }
};

template <>
struct std::hash<SignalRestrictionGenomeDescription>
{
    std::size_t operator()(SignalRestrictionGenomeDescription const& desc) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, desc._active);
        boost::hash_combine(seed, desc._baseAngle);
        boost::hash_combine(seed, desc._openingAngle);
        return seed;
    }
};

template <>
struct std::hash<NodeDescription>
{
    std::size_t operator()(NodeDescription const& desc) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, desc._referenceAngle);
        boost::hash_combine(seed, desc._color);
        boost::hash_combine(seed, desc._numAdditionalConnections);
        boost::hash_combine(seed, std::hash<NeuralNetworkGenomeDescription>{}(desc._neuralNetwork));
        boost::hash_combine(seed, std::hash<CellTypeGenomeDescription>{}(desc._cellType));
        boost::hash_combine(seed, std::hash<SignalRestrictionGenomeDescription>{}(desc._signalRestriction));
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
            boost::hash_combine(seed, std::hash<NodeDescription>{}(node));
        }
        boost::hash_combine(seed, static_cast<int>(desc._shape));
        boost::hash_combine(seed, desc._separation);
        boost::hash_combine(seed, desc._numBranches);
        boost::hash_combine(seed, desc._numConcatenations);
        boost::hash_combine(seed, static_cast<int>(desc._angleAlignment));
        boost::hash_combine(seed, desc._stiffness);
        boost::hash_combine(seed, desc._connectionDistance);
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
            boost::hash_combine(seed, std::hash<GeneDescription>{}(gene));
        }
        boost::hash_combine(seed, desc._frontAngle);
        return seed;
    }
};

template <>
struct std::hash<SubGenomeDescription>
{
    std::size_t operator()(SubGenomeDescription const& genomeWithRootIndex) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, genomeWithRootIndex.genome);
        boost::hash_combine(seed, genomeWithRootIndex.trimmed);
        boost::hash_combine(seed, genomeWithRootIndex.startIndex);
        return seed;
    }
};
#pragma once

#include <functional>
#include <variant>
#include "GenomeDescription.h"

namespace std {

/**
 * Hash function specializations for GenomeDescription and all related types.
 * 
 * This header provides std::hash specializations that allow GenomeDescription
 * and its constituent types to be used as keys in std::unordered_map and 
 * std::unordered_set containers.
 * 
 * The hash functions are designed to be:
 * - Consistent with operator== (equal objects have equal hashes)
 * - Fast and reasonably collision-resistant
 * - Comprehensive (all members that affect equality are included)
 * 
 * Usage example:
 *   std::unordered_map<GenomeDescription, std::string> genomeMap;
 *   GenomeDescription genome = // ... construct genome
 *   genomeMap[genome] = "My genome";
 */

// Hash combining utility function to avoid collisions
// Uses the same algorithm as boost::hash_combine
template<typename T>
inline void hash_combine(std::size_t& seed, const T& val) {
    seed ^= std::hash<T>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Helper to hash variant types
template<typename... Types>
struct variant_hasher {
    std::size_t operator()(const std::variant<Types...>& v) const {
        return std::visit([](const auto& val) -> std::size_t {
            return std::hash<std::decay_t<decltype(val)>>{}(val);
        }, v);
    }
};

// Hash specialization for NeuralNetworkGenomeDescription
template<>
struct hash<NeuralNetworkGenomeDescription> {
    std::size_t operator()(const NeuralNetworkGenomeDescription& desc) const {
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

// Hash specialization for BaseGenomeDescription
template<>
struct hash<BaseGenomeDescription> {
    std::size_t operator()(const BaseGenomeDescription& desc) const {
        // BaseGenomeDescription has no members, so return a constant
        return 0;
    }
};

// Hash specialization for DepotGenomeDescription
template<>
struct hash<DepotGenomeDescription> {
    std::size_t operator()(const DepotGenomeDescription& desc) const {
        return std::hash<int>{}(static_cast<int>(desc._mode));
    }
};

// Hash specialization for ConstructorGenomeDescription
template<>
struct hash<ConstructorGenomeDescription> {
    std::size_t operator()(const ConstructorGenomeDescription& desc) const {
        std::size_t seed = 0;
        if (desc._autoTriggerInterval) {
            hash_combine(seed, *desc._autoTriggerInterval);
        } else {
            hash_combine(seed, -1); // Use -1 for nullopt
        }
        hash_combine(seed, desc._geneIndex);
        hash_combine(seed, desc._constructionActivationTime);
        return seed;
    }
};

// Hash specialization for SensorGenomeDescription
template<>
struct hash<SensorGenomeDescription> {
    std::size_t operator()(const SensorGenomeDescription& desc) const {
        std::size_t seed = 0;
        if (desc._autoTriggerInterval) {
            hash_combine(seed, *desc._autoTriggerInterval);
        } else {
            hash_combine(seed, -1);
        }
        hash_combine(seed, desc._minDensity);
        if (desc._minRange) {
            hash_combine(seed, *desc._minRange);
        } else {
            hash_combine(seed, -1);
        }
        if (desc._maxRange) {
            hash_combine(seed, *desc._maxRange);
        } else {
            hash_combine(seed, -1);
        }
        if (desc._restrictToColor) {
            hash_combine(seed, *desc._restrictToColor);
        } else {
            hash_combine(seed, -1);
        }
        hash_combine(seed, static_cast<int>(desc._restrictToCreatures));
        return seed;
    }
};

// Hash specialization for GeneratorGenomeDescription
template<>
struct hash<GeneratorGenomeDescription> {
    std::size_t operator()(const GeneratorGenomeDescription& desc) const {
        std::size_t seed = 0;
        hash_combine(seed, desc._autoTriggerInterval);
        hash_combine(seed, static_cast<int>(desc._pulseType));
        hash_combine(seed, desc._alternationInterval);
        return seed;
    }
};

// Hash specialization for AttackerGenomeDescription
template<>
struct hash<AttackerGenomeDescription> {
    std::size_t operator()(const AttackerGenomeDescription& desc) const {
        // AttackerGenomeDescription has no members, so return a constant
        return 1;
    }
};

// Hash specialization for InjectorGenomeDescription
template<>
struct hash<InjectorGenomeDescription> {
    std::size_t operator()(const InjectorGenomeDescription& desc) const {
        return std::hash<int>{}(static_cast<int>(desc._mode));
    }
};

// Hash specialization for AutoBendingGenomeDescription
template<>
struct hash<AutoBendingGenomeDescription> {
    std::size_t operator()(const AutoBendingGenomeDescription& desc) const {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxAngleDeviation);
        hash_combine(seed, desc._frontBackVelRatio);
        return seed;
    }
};

// Hash specialization for ManualBendingGenomeDescription
template<>
struct hash<ManualBendingGenomeDescription> {
    std::size_t operator()(const ManualBendingGenomeDescription& desc) const {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxAngleDeviation);
        hash_combine(seed, desc._frontBackVelRatio);
        return seed;
    }
};

// Hash specialization for AngleBendingGenomeDescription
template<>
struct hash<AngleBendingGenomeDescription> {
    std::size_t operator()(const AngleBendingGenomeDescription& desc) const {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxAngleDeviation);
        hash_combine(seed, desc._frontBackVelRatio);
        return seed;
    }
};

// Hash specialization for AutoCrawlingGenomeDescription
template<>
struct hash<AutoCrawlingGenomeDescription> {
    std::size_t operator()(const AutoCrawlingGenomeDescription& desc) const {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxDistanceDeviation);
        hash_combine(seed, desc._frontBackVelRatio);
        return seed;
    }
};

// Hash specialization for ManualCrawlingGenomeDescription
template<>
struct hash<ManualCrawlingGenomeDescription> {
    std::size_t operator()(const ManualCrawlingGenomeDescription& desc) const {
        std::size_t seed = 0;
        hash_combine(seed, desc._maxDistanceDeviation);
        hash_combine(seed, desc._frontBackVelRatio);
        return seed;
    }
};

// Hash specialization for DirectMovementGenomeDescription
template<>
struct hash<DirectMovementGenomeDescription> {
    std::size_t operator()(const DirectMovementGenomeDescription& desc) const {
        // DirectMovementGenomeDescription has no members, so return a constant
        return 2;
    }
};

// Hash specialization for MuscleModeGenomeDescription (variant)
template<>
struct hash<MuscleModeGenomeDescription> {
    std::size_t operator()(const MuscleModeGenomeDescription& desc) const {
        return variant_hasher<
            AutoBendingGenomeDescription,
            ManualBendingGenomeDescription,
            AngleBendingGenomeDescription,
            AutoCrawlingGenomeDescription,
            ManualCrawlingGenomeDescription,
            DirectMovementGenomeDescription>{}(desc);
    }
};

// Hash specialization for MuscleGenomeDescription
template<>
struct hash<MuscleGenomeDescription> {
    std::size_t operator()(const MuscleGenomeDescription& desc) const {
        return std::hash<MuscleModeGenomeDescription>{}(desc._mode);
    }
};

// Hash specialization for DefenderGenomeDescription
template<>
struct hash<DefenderGenomeDescription> {
    std::size_t operator()(const DefenderGenomeDescription& desc) const {
        return std::hash<int>{}(static_cast<int>(desc._mode));
    }
};

// Hash specialization for ReconnectorGenomeDescription
template<>
struct hash<ReconnectorGenomeDescription> {
    std::size_t operator()(const ReconnectorGenomeDescription& desc) const {
        std::size_t seed = 0;
        if (desc._restrictToColor) {
            hash_combine(seed, *desc._restrictToColor);
        } else {
            hash_combine(seed, -1);
        }
        hash_combine(seed, static_cast<int>(desc._restrictToCreatures));
        return seed;
    }
};

// Hash specialization for DetonatorGenomeDescription
template<>
struct hash<DetonatorGenomeDescription> {
    std::size_t operator()(const DetonatorGenomeDescription& desc) const {
        return std::hash<int>{}(desc._countdown);
    }
};

// Hash specialization for CellTypeGenomeDescription (variant)
template<>
struct hash<CellTypeGenomeDescription> {
    std::size_t operator()(const CellTypeGenomeDescription& desc) const {
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

// Hash specialization for SignalRoutingRestrictionGenomeDescription
template<>
struct hash<SignalRoutingRestrictionGenomeDescription> {
    std::size_t operator()(const SignalRoutingRestrictionGenomeDescription& desc) const {
        std::size_t seed = 0;
        hash_combine(seed, desc._active);
        hash_combine(seed, desc._baseAngle);
        hash_combine(seed, desc._openingAngle);
        return seed;
    }
};

// Hash specialization for NodeDescription
template<>
struct hash<NodeDescription> {
    std::size_t operator()(const NodeDescription& desc) const {
        std::size_t seed = 0;
        hash_combine(seed, desc._referenceAngle);
        hash_combine(seed, desc._color);
        hash_combine(seed, desc._numAdditionalConnections);
        hash_combine(seed, std::hash<NeuralNetworkGenomeDescription>{}(desc._neuralNetwork));
        hash_combine(seed, std::hash<CellTypeGenomeDescription>{}(desc._cellTypeData));
        hash_combine(seed, std::hash<SignalRoutingRestrictionGenomeDescription>{}(desc._signalRoutingRestriction));
        return seed;
    }
};

// Hash specialization for GeneDescription
template<>
struct hash<GeneDescription> {
    std::size_t operator()(const GeneDescription& desc) const {
        std::size_t seed = 0;
        for (const auto& node : desc._nodes) {
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

// Hash specialization for GenomeDescription
template<>
struct hash<GenomeDescription> {
    std::size_t operator()(const GenomeDescription& desc) const {
        std::size_t seed = 0;
        for (const auto& gene : desc._genes) {
            hash_combine(seed, std::hash<GeneDescription>{}(gene));
        }
        hash_combine(seed, desc._frontAngle);
        return seed;
    }
};

} // namespace std
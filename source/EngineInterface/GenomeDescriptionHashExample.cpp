/**
 * Example demonstrating how to use GenomeDescription as a key in hash-based containers.
 * 
 * This example shows that GenomeDescription can now be used directly as a key in 
 * std::unordered_map and std::unordered_set without any additional user code.
 */

#include "EngineInterface/GenomeDescription.h"
#include <unordered_map>
#include <unordered_set>

void example_usage() {
    // Create some sample genomes
    auto genome1 = GenomeDescription()
        .genes({
            GeneDescription()
                .separation(true)
                .nodes({
                    NodeDescription()
                        .cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
                    NodeDescription()
                        .cellTypeData(SensorGenomeDescription().minDensity(0.1f)),
                })
        })
        .frontAngle(1.57f);

    auto genome2 = GenomeDescription()
        .genes({
            GeneDescription()
                .separation(false)
                .nodes({
                    NodeDescription()
                        .cellTypeData(AttackerGenomeDescription()),
                })
        })
        .frontAngle(3.14f);

    // Use GenomeDescription as a key in unordered_map
    std::unordered_map<GenomeDescription, std::string> genomeNames;
    genomeNames[genome1] = "Constructor-Sensor Genome";
    genomeNames[genome2] = "Simple Attacker Genome";

    // Use GenomeDescription in unordered_set
    std::unordered_set<GenomeDescription> uniqueGenomes;
    uniqueGenomes.insert(genome1);
    uniqueGenomes.insert(genome2);
    
    // Fast lookups are now possible
    auto it = genomeNames.find(genome1);
    if (it != genomeNames.end()) {
        // Found the genome, use it->second for the associated value
    }
    
    // Check if a genome exists in the set
    bool exists = uniqueGenomes.count(genome1) > 0;
}
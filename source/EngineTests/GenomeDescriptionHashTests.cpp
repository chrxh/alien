#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "EngineInterface/GenomeDescription.h"
#include "EngineInterface/GenomeDescriptionHash.h"

class GenomeDescriptionHashTests : public ::testing::Test
{
public:
    virtual ~GenomeDescriptionHashTests() = default;

protected:
    // Helper method to create a simple genome for testing
    GenomeDescription createSimpleGenome()
    {
        return GenomeDescription()
            .genes({
                GeneDescription()
                    .separation(true)
                    .nodes({
                        NodeDescription()
                            .cellTypeData(ConstructorGenomeDescription().geneIndex(0)),
                        NodeDescription()
                            .cellTypeData(BaseGenomeDescription()),
                    }),
                GeneDescription()
                    .separation(false)
                    .numBranches(2)
                    .nodes({
                        NodeDescription()
                            .cellTypeData(SensorGenomeDescription().minDensity(0.1f)),
                        NodeDescription()
                            .cellTypeData(MuscleGenomeDescription()
                                .mode(AutoBendingGenomeDescription().maxAngleDeviation(0.5f))),
                    })
            })
            .frontAngle(1.57f);
    }

    // Helper method to create another genome for testing
    GenomeDescription createDifferentGenome()
    {
        return GenomeDescription()
            .genes({
                GeneDescription()
                    .separation(true)
                    .nodes({
                        NodeDescription()
                            .cellTypeData(AttackerGenomeDescription()),
                        NodeDescription()
                            .cellTypeData(DefenderGenomeDescription().mode(DefenderMode_DefendAgainstAttacker)),
                    })
            })
            .frontAngle(3.14f);
    }
};

TEST_F(GenomeDescriptionHashTests, HashConsistency)
{
    auto genome1 = createSimpleGenome();
    auto genome2 = createSimpleGenome(); // Same content
    auto genome3 = createDifferentGenome();

    std::hash<GenomeDescription> hasher;
    
    // Same genomes should have same hash
    EXPECT_EQ(hasher(genome1), hasher(genome2));
    
    // Different genomes should have different hashes (high probability)
    EXPECT_NE(hasher(genome1), hasher(genome3));
}

TEST_F(GenomeDescriptionHashTests, UnorderedMapUsage)
{
    std::unordered_map<GenomeDescription, std::string> genomeMap;
    
    auto genome1 = createSimpleGenome();
    auto genome2 = createDifferentGenome();
    
    // Insert genomes as keys
    genomeMap[genome1] = "Simple genome";
    genomeMap[genome2] = "Different genome";
    
    // Verify we can retrieve values
    EXPECT_EQ(genomeMap[genome1], "Simple genome");
    EXPECT_EQ(genomeMap[genome2], "Different genome");
    
    // Verify size
    EXPECT_EQ(genomeMap.size(), 2);
    
    // Test find operation
    auto it1 = genomeMap.find(genome1);
    EXPECT_NE(it1, genomeMap.end());
    EXPECT_EQ(it1->second, "Simple genome");
    
    auto it2 = genomeMap.find(genome2);
    EXPECT_NE(it2, genomeMap.end());
    EXPECT_EQ(it2->second, "Different genome");
}

TEST_F(GenomeDescriptionHashTests, UnorderedSetUsage)
{
    std::unordered_set<GenomeDescription> genomeSet;
    
    auto genome1 = createSimpleGenome();
    auto genome2 = createDifferentGenome();
    auto genome3 = createSimpleGenome(); // Same as genome1
    
    // Insert genomes
    genomeSet.insert(genome1);
    genomeSet.insert(genome2);
    genomeSet.insert(genome3); // Should not increase size since it's same as genome1
    
    // Verify size - should be 2, not 3
    EXPECT_EQ(genomeSet.size(), 2);
    
    // Verify contains operations
    EXPECT_TRUE(genomeSet.find(genome1) != genomeSet.end());
    EXPECT_TRUE(genomeSet.find(genome2) != genomeSet.end());
    EXPECT_TRUE(genomeSet.find(genome3) != genomeSet.end()); // Same as genome1
}

TEST_F(GenomeDescriptionHashTests, HashSubComponents)
{
    // Test that individual components can also be hashed
    std::hash<GeneDescription> geneHasher;
    std::hash<NodeDescription> nodeHasher;
    std::hash<ConstructorGenomeDescription> constructorHasher;
    
    auto gene = GeneDescription().separation(true);
    auto node = NodeDescription().color(5);
    auto constructor = ConstructorGenomeDescription().geneIndex(3);
    
    // Just verify they produce hash values without throwing
    auto geneHash = geneHasher(gene);
    auto nodeHash = nodeHasher(node);
    auto constructorHash = constructorHasher(constructor);
    
    EXPECT_TRUE(true); // If we reach here, hashing worked
}

TEST_F(GenomeDescriptionHashTests, ComplexGenomeHashing)
{
    // Create a more complex genome with various cell types
    auto complexGenome = GenomeDescription()
        .genes({
            GeneDescription()
                .separation(true)
                .shape(ConstructorShape_Triangle)
                .nodes({
                    NodeDescription()
                        .color(1)
                        .referenceAngle(0.5f)
                        .cellTypeData(ConstructorGenomeDescription()
                            .geneIndex(1)
                            .constructionActivationTime(50)),
                    NodeDescription()
                        .color(2)
                        .cellTypeData(SensorGenomeDescription()
                            .minDensity(0.2f)
                            .restrictToColor(3)),
                    NodeDescription()
                        .color(3)
                        .cellTypeData(MuscleGenomeDescription()
                            .mode(ManualBendingGenomeDescription()
                                .maxAngleDeviation(0.3f)
                                .frontBackVelRatio(0.4f))),
                    NodeDescription()
                        .color(4)
                        .cellTypeData(DefenderGenomeDescription()
                            .mode(DefenderMode_DefendAgainstAttacker)),
                    NodeDescription()
                        .color(5)
                        .cellTypeData(DetonatorGenomeDescription()
                            .countdown(20))
                })
        })
        .frontAngle(2.0f);
    
    std::unordered_map<GenomeDescription, int> complexMap;
    complexMap[complexGenome] = 42;
    
    EXPECT_EQ(complexMap[complexGenome], 42);
    EXPECT_EQ(complexMap.size(), 1);
}
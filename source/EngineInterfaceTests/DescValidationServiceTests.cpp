
#include <gtest/gtest.h>

#include <EngineInterface/DescValidationService.h>
#include <EngineInterface/Descs.h>
#include <EngineInterface/GenomeDesc.h>

class DescValidationServiceTests : public ::testing::Test
{
public:
    virtual ~DescValidationServiceTests() = default;

protected:
    DescValidationService& _descValidationService = DescValidationService::get();
};

TEST_F(DescValidationServiceTests, genome_removesConstructorFromVoidCell)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc(),
            NodeDesc().cellType(VoidGenomeDesc()).constructor(ConstructorGenomeDesc().geneIndex(0).separation(true)),
            NodeDesc(),
        }),
    });

    _descValidationService.validateAndCorrect(genome);

    EXPECT_FALSE(genome._genes.at(0)._nodes.at(1)._constructor.has_value());
}

TEST_F(DescValidationServiceTests, genome_keepsConstructorOnVoidCellInHomogeneousGene)
{
    // In a homogeneous gene the effective cell type comes from the first node, so a void node is expressed as that type
    // and legitimately keeps its constructor (matching ConstructorProcessor and MutationProcessor).
    auto genome = GenomeDesc().genes({
        GeneDesc().homogeneousCellType(true).nodes({
            NodeDesc(),
            NodeDesc().cellType(VoidGenomeDesc()).constructor(ConstructorGenomeDesc().geneIndex(0).separation(true)),
            NodeDesc(),
        }),
    });

    _descValidationService.validateAndCorrect(genome);

    EXPECT_TRUE(genome._genes.at(0)._nodes.at(1)._constructor.has_value());
}

TEST_F(DescValidationServiceTests, genome_keepsConstructorOnNonVoidCell)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(true)),
            NodeDesc(),
        }),
    });

    _descValidationService.validateAndCorrect(genome);

    EXPECT_TRUE(genome._genes.at(0)._nodes.at(0)._constructor.has_value());
}

TEST_F(DescValidationServiceTests, cell_removesConstructorFromVoidCell)
{
    ExtendedObjectDesc extendedObject;
    extendedObject.object = ObjectDesc().type(CellDesc().cellType(VoidDesc()).constructor(ConstructorDesc().geneIndex(0).separation(true)));

    _descValidationService.validateAndCorrect(extendedObject);

    EXPECT_FALSE(extendedObject.object.getCellRef()._constructor.has_value());
}

TEST_F(DescValidationServiceTests, cell_keepsConstructorOnNonVoidCell)
{
    ExtendedObjectDesc extendedObject;
    extendedObject.object = ObjectDesc().type(CellDesc().constructor(ConstructorDesc().geneIndex(0).separation(true)));

    _descValidationService.validateAndCorrect(extendedObject);

    EXPECT_TRUE(extendedObject.object.getCellRef()._constructor.has_value());
}

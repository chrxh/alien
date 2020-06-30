#include "IntegrationGpuTestFramework.h"

class ParticleGpuTests
	: public IntegrationGpuTestFramework
{
public:
    ParticleGpuTests(
        IntVector2D const& universeSize = { 900, 600 },
        optional<ModelGpuData> const& modelData = boost::none)
        : IntegrationGpuTestFramework(universeSize, modelData)
    {}

	virtual ~ParticleGpuTests() = default;
};

namespace
{
    ModelGpuData getModelGpuDataWithOneBlock()
    {
        CudaConstants cudaConstants;
        cudaConstants.NUM_THREADS_PER_BLOCK = 16;
        cudaConstants.NUM_BLOCKS = 1;
        cudaConstants.MAX_CLUSTERS = 100000;
        cudaConstants.MAX_CELLS = 500000;
        cudaConstants.MAX_PARTICLES = 500000;
        cudaConstants.MAX_TOKENS = 50000;
        cudaConstants.MAX_CELLPOINTERS = 500000 * 10;
        cudaConstants.MAX_CLUSTERPOINTERS = 100000 * 10;
        cudaConstants.MAX_PARTICLEPOINTERS = 500000 * 10;
        cudaConstants.MAX_TOKENPOINTERS = 50000 * 10;
        cudaConstants.DYNAMIC_MEMORY_SIZE = 100000000;
        cudaConstants.METADATA_DYNAMIC_MEMORY_SIZE = 1000;
        return ModelGpuData(cudaConstants);
    }
}

class ParticleGpuWithOneBlockTests : public ParticleGpuTests
{
public:
    ParticleGpuWithOneBlockTests()
        : ParticleGpuTests({ 100, 100}, getModelGpuDataWithOneBlock())
    { }

    virtual ~ParticleGpuWithOneBlockTests() = default;
};

                                                                    
/**
* Situation: fusion of two particles
* Expected result: one particle remains with average velocity
*/
TEST_F(ParticleGpuTests, testFusionOfTwoParticles)
{
	DataDescription origData;
	auto particleEnergy = _parameters.cellMinEnergy / 3.0;

	auto particleId1 = _numberGen->getId();
	auto particle1 = ParticleDescription().setId(particleId1).setEnergy(particleEnergy).setPos({ 100, 100 }).setVel({ 0.5, 0.0 });
	origData.addParticle(particle1);

	auto particleId2 = _numberGen->getId();
	auto particle2 = ParticleDescription().setId(particleId2).setEnergy(particleEnergy).setPos({ 110, 100 }).setVel({ -0.5, 0.0 });
	origData.addParticle(particle2);

	IntegrationTestHelper::updateData(_access, origData);
	IntegrationTestHelper::runSimulation(30, _controller);

	IntRect rect = { { 0, 0 },{ _universeSize.x, _universeSize.y } };
	DataDescription newData = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_FALSE(newData.clusters);
	ASSERT_EQ(1, newData.particles->size());
	auto newParticle = newData.particles->front();
	checkCompatibility(QVector2D(0, 0), *newParticle.vel);

    check(origData, newData);
}

/**
* Situation: fusion of many particles
* Expected result: energy balance is fulfilled
*/
TEST_F(ParticleGpuTests, testFusionOfManyParticles)
{
	auto particleEnergy = _parameters.cellMinEnergy / 120.0;

	DataDescription origData;
	for (int i = 0; i < 100; ++i) {
		origData.addParticle(
			ParticleDescription().setId(_numberGen->getId()).setEnergy(particleEnergy).setPos({ 100, 100 }).setVel({ 0.5, 0.0 }));
	}

	IntegrationTestHelper::updateData(_access, origData);
	IntegrationTestHelper::runSimulation(300, _controller);
	DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    check(origData, newData);
}

/**
* Situation: particle with high energy
* Expected result: particle transforms to cell
*/
TEST_F(ParticleGpuTests, testTransformationParticleToCell)
{
    auto size = _spaceProp->getSize();
    DataDescription origData;
    float cellMinEnergy = static_cast<float>(_parameters.cellMinEnergy);
    origData.addParticle(ParticleDescription().setId(_numberGen->getId()).setPos({ 0, 0 }).setVel({ 0.5, 0 }).setEnergy(cellMinEnergy * 2));

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(100, _controller);
    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    ASSERT_EQ(1, newData.clusters->size());
    EXPECT_EQ(1, newData.clusters->at(0).cells->size());

    check(origData, newData);
}

/**
* Situation: many random particles and one 10x10 cluster
* Fixed error: wrong block partition in ParticleProcessor::updateMap_system led to dead cells in particle map
* Expected result: energy balance fulfilled
*/
TEST_F(ParticleGpuWithOneBlockTests, regressionTestFusionOfManyParticles)
{
    _parameters.radiationProb = 0;  //exclude radiation
    _parameters.cellTransformationProb = 0;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    for (int i = 0; i < 10000; ++i) {
        auto const particle = ParticleDescription()
                                  .setId(_numberGen->getId())
                                  .setEnergy(3)
                                  .setPos(
                                      {static_cast<float>(_numberGen->getRandomReal(0, _universeSize.x)),
                                       static_cast<float>(_numberGen->getRandomReal(0, _universeSize.y))})
                                  .setVel(
                                      {static_cast<float>(_numberGen->getRandomReal(-1, 1)),
                                       static_cast<float>(_numberGen->getRandomReal(-1, 1))});
        origData.addParticle(particle);
    }
    origData.addCluster(createRectangularCluster({ 10, 10 }, QVector2D{ 0, 0 }, QVector2D{}));

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1000, _controller);

    IntRect rect = {{0, 0}, {_universeSize.x, _universeSize.y}};
    DataDescription newData = IntegrationTestHelper::getContent(_access, rect);

    check(origData, newData);
}

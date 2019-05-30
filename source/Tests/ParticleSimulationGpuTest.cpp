#include "IntegrationGpuTestFramework.h"

class ParticleSimulationGpuTest
	: public IntegrationGpuTestFramework
{
public:
	virtual ~ParticleSimulationGpuTest() = default;
};

/**
* Situation: fusion of two particles
* Expected result: one particle remains with average velocity
*/
TEST_F(ParticleSimulationGpuTest, testFusionOfSingleParticles)
{
	DataDescription origData;
	auto particleEnergy = _parameters.cellMinEnergy / 3.0;

	auto particleId1 = _numberGen->getId();
	auto particle1 = ParticleDescription().setId(particleId1).setEnergy(particleEnergy).setPos({ 100, 100 }).setVel({ 0.5, 0.0 });
	origData.addParticle(particle1);

	auto particleId2 = _numberGen->getId();
	auto particle2 = ParticleDescription().setId(particleId1).setEnergy(particleEnergy).setPos({ 110, 100 }).setVel({ -0.5, 0.0 });
	origData.addParticle(particle2);

	IntegrationTestHelper::updateData(_access, origData);
	IntegrationTestHelper::runSimulation(30, _controller);

	IntRect rect = { { 0, 0 },{ _universeSize.x, _universeSize.y } };
	DataDescription newData = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_FALSE(newData.clusters);
	ASSERT_EQ(1, newData.particles->size());
	auto newParticle = newData.particles->front();
	EXPECT_TRUE(isCompatible(QVector2D(0, 0), *newParticle.vel));

	checkEnergy(origData, newData);
}

/**
* Situation: fusion of many particles
* Expected result: energy balance is fulfilled
*/
TEST_F(ParticleSimulationGpuTest, testFusionOfManyParticles)
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

	checkEnergy(origData, newData);
}

/**
* Situation: particle with high energy
* Expected result: particle transforms to cell
*/
TEST_F(ParticleSimulationGpuTest, testTransformationParticleToCell)
{
	auto size = _spaceProp->getSize();
	DataDescription origData;
	float cellMinEnergy = static_cast<float>(_parameters.cellMinEnergy);
	origData.addParticle(ParticleDescription().setId(_numberGen->getId()).setPos({ 0, 0 }).setVel({ 0, 0 }).setEnergy(cellMinEnergy * 2));

	IntegrationTestHelper::updateData(_access, origData);
	IntegrationTestHelper::runSimulation(100, _controller);
	DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_EQ(1, newData.clusters->size());
	EXPECT_EQ(1, newData.clusters->at(0).cells->size());
}

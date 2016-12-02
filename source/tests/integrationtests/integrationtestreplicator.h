#ifndef INTEGRATIONTESTREPLICATOR_H
#define INTEGRATIONTESTREPLICATOR_H

#include <gtest/gtest.h>

class SimulationController;

class IntegrationTestReplicator : public ::testing::Test
{
public:
	IntegrationTestReplicator();
	~IntegrationTestReplicator();

protected:
    SimulationController* _simulationController;
};



#endif // INTEGRATIONTESTREPLICATOR_H

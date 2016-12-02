#ifndef INTEGRATIONTESTCOMPARISON_H
#define INTEGRATIONTESTCOMPARISON_H

#include <gtest/gtest.h>

class SimulationController;
class Grid;

class IntegrationTestComparison : public ::testing::Test
{
public:
	IntegrationTestComparison();
	~IntegrationTestComparison();

protected:
    SimulationController* _simulationController;
};

#endif // INTEGRATIONTESTCOMPARISON_H

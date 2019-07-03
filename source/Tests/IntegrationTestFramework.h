#pragma once

#include <gtest/gtest.h>

#include "ModelCpu/Definitions.h"
#include "ModelGpu/Definitions.h"

#include "TestSettings.h"

class IntegrationTestFramework : public ::testing::Test
{
public:
	IntegrationTestFramework(IntVector2D const& universeSize);
	virtual ~IntegrationTestFramework();

protected:

	//boost::none means random
	ClusterDescription createRectangularCluster(IntVector2D const& size,
		optional<QVector2D> const& centerPos = boost::none,
		optional<QVector2D> const& centerVel = boost::none) const;
	ClusterDescription createLineCluster(int numCells,
		optional<QVector2D> const& centerPos = boost::none,
		optional<QVector2D> const& centerVel = boost::none,
		optional<double> const& angle = boost::none,
		optional<double> const& angularVel = boost::none) const;
	ClusterDescription createHorizontalCluster(int numCells,
		optional<QVector2D> const& centerPos = boost::none,
		optional<QVector2D> const& centerVel = boost::none,
		optional<double> const& angularVel = boost::none) const;	
	ClusterDescription createVerticalCluster(int numCells,
		optional<QVector2D> const& centerPos = boost::none,
		optional<QVector2D> const& centerVel = boost::none) const;	
	ClusterDescription createSingleCellCluster(uint64_t clusterId = 0, uint64_t cellId = 0) const;
	ClusterDescription createSingleCellClusterWithCompleteData(uint64_t clusterId = 0, uint64_t cellId = 0) const;
	TokenDescription createSimpleToken() const;

    ParticleDescription createParticle(
        optional<QVector2D> const& pos = boost::none,
        optional<QVector2D> const& vel = boost::none) const;

    ModelBasicBuilderFacade* _basicFacade = nullptr;
	ModelCpuBuilderFacade* _cpuFacade = nullptr;
	ModelGpuBuilderFacade* _gpuFacade = nullptr;
	SimulationParameters _parameters;
	NumberGenerator* _numberGen = nullptr;
	SymbolTable* _symbols = nullptr;
	IntVector2D _universeSize{ 12 * 33 * 3, 12 * 17 * 3 };
};

template<typename T>
bool isCompatible(T a, T b)
{
	return a == b;
}

template<typename T>
bool isCompatible(optional<T> a, optional<T> b)
{
	if (!a || !b) {
		return true;
	}
	return isCompatible(*a, *b);
}

template<typename T>
bool isCompatible(vector<T> a, vector<T> b)
{
	if (a.size() != b.size()) {
		return false;
	}
	for (int i = 0; i < a.size(); ++i) {
		if (!isCompatible(a.at(i), b.at(i))) {
			return false;
		}
	}
	return true;
}

template<> bool isCompatible<QVector2D>(QVector2D vec1, QVector2D vec2);
template<> bool isCompatible<double>(double a, double b);
template<> bool isCompatible<TokenDescription>(TokenDescription token1, TokenDescription token2);
template<> bool isCompatible<CellFeatureDescription>(CellFeatureDescription feature1, CellFeatureDescription feature2);
template<> bool isCompatible<CellDescription>(CellDescription cell1, CellDescription cell2);
template<> bool isCompatible<ClusterDescription>(ClusterDescription cluster1, ClusterDescription cluster2);
template<> bool isCompatible<ParticleDescription>(ParticleDescription particle1, ParticleDescription particle2);
template<> bool isCompatible<DataDescription>(DataDescription data1, DataDescription data2);

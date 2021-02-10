#pragma once

#include <gtest/gtest.h>

#include "EngineGpu/Definitions.h"

#include "TestSettings.h"

class IntegrationTestFramework : public ::testing::Test
{
public:
	IntegrationTestFramework(IntVector2D const& universeSize);
	virtual ~IntegrationTestFramework();

protected:

	//important: boost::none means random
    enum class Boundary {Sticky, NonSticky};
    ClusterDescription createRectangularCluster(
        IntVector2D const& size,
        optional<QVector2D> const& centerPos = boost::none,
        optional<QVector2D> const& centerVel = boost::none,
        Boundary boundary = Boundary::NonSticky) const;
    ClusterDescription createLineCluster(int numCells,
		optional<QVector2D> const& centerPos = boost::none,
		optional<QVector2D> const& centerVel = boost::none,
		optional<double> const& angle = boost::none,
		optional<double> const& angularVel = boost::none) const;
	ClusterDescription createHorizontalCluster(int numCells,
		optional<QVector2D> const& centerPos = boost::none,
		optional<QVector2D> const& centerVel = boost::none,
		optional<double> const& angularVel = boost::none,
        Boundary boundary = Boundary::NonSticky) const;
	ClusterDescription createVerticalCluster(int numCells,
		optional<QVector2D> const& centerPos = boost::none,
		optional<QVector2D> const& centerVel = boost::none) const;	
	ClusterDescription createSingleCellCluster(uint64_t clusterId = 0, uint64_t cellId = 0) const;
	ClusterDescription createSingleCellClusterWithCompleteData(uint64_t clusterId = 0, uint64_t cellId = 0) const;
	TokenDescription createSimpleToken() const;

    ParticleDescription createParticle(
        optional<QVector2D> const& pos = boost::none,
        optional<QVector2D> const& vel = boost::none) const;

    //prevent indeterminism when position is between two pixels
    QVector2D addSmallDisplacement(QVector2D const& value) const;

    EngineInterfaceBuilderFacade* _basicFacade = nullptr;
	EngineGpuBuilderFacade* _gpuFacade = nullptr;
	SimulationParameters _parameters;
	NumberGenerator* _numberGen = nullptr;
	SymbolTable* _symbols = nullptr;
	IntVector2D _universeSize{ 12 * 33 * 3, 12 * 17 * 3 };
};

template<typename T>
bool checkCompatibility(T a, T b)
{
    bool result = true;
    EXPECT_EQ(a, b);
    return a == b;
}

template<typename T>
bool checkCompatibility(optional<T> a, optional<T> b)
{
	if (!a || !b) {
		return true;
	}
	return checkCompatibility(*a, *b);
}

template<typename T>
bool checkCompatibility(vector<T> a, vector<T> b)
{
    auto result = true;
    EXPECT_EQ(a.size(), b.size());
    for (int i = 0; i < a.size(); ++i) {
        result &= checkCompatibility(a.at(i), b.at(i));
    }
    return result;
}

template<> bool checkCompatibility<QVector2D>(QVector2D vec1, QVector2D vec2);
template<> bool checkCompatibility<double>(double a, double b);
template<> bool checkCompatibility<float>(float a, float b);
template<> bool checkCompatibility<CellFeatureDescription>(CellFeatureDescription feature1, CellFeatureDescription feature2);
template<> bool checkCompatibility<CellMetadata>(CellMetadata metadata1, CellMetadata metadata2);
template<> bool checkCompatibility<TokenDescription>(TokenDescription token1, TokenDescription token2);
template<> bool checkCompatibility<CellDescription>(CellDescription cell1, CellDescription cell2);
template<> bool checkCompatibility<ClusterDescription>(ClusterDescription cluster1, ClusterDescription cluster2);
template<> bool checkCompatibility<ParticleDescription>(ParticleDescription particle1, ParticleDescription particle2);
template<> bool checkCompatibility<DataDescription>(DataDescription data1, DataDescription data2);

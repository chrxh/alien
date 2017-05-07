#ifndef LIGHTDESCRIPTIONS_H
#define LIGHTDESCRIPTIONS_H

#include "model/Definitions.h"

struct CellLightDescription
{
	uint64_t id = 0.0;
	QVector2D relPos;
	double energy = 0.0;
	uint8_t color = 0;
	int numTokens = 0;
};

struct CellClusterLightDescription
{
	uint64_t id = 0.0;
	QVector2D pos;
	QVector2D vel;
	double angle = 0.0;
	double angularVel = 0.0;
	vector<CellLightDescription> cells;
};

struct EnergyParticleLightDescription
{
	uint64_t id = 0.0;
	QVector2D pos;
	QVector2D vel;
	double energy = 0.0;
};

struct DataLightDescription
{
	vector<CellClusterLightDescription> clusters;
	vector<EnergyParticleLightDescription> particles;
};

#endif // LIGHTDESCRIPTIONS_H

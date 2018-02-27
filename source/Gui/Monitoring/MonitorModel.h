#pragma once

struct _MonitorModel
{
	int numClusters = 0;
	int numCells = 0;
	int numParticles = 0;
	int numToken = 0;
	double totalInternalEnergy = 0.0;
	double totalKineticEnergy = 0.0;
	double totalKineticEnergyTranslationalPart = 0.0;
	double totalKineticEnergyRotationalPart = 0.0;
};

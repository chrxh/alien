#ifndef SIMULATIONPARAMETERS_H
#define SIMULATIONPARAMETERS_H

#include "definitions.h"

struct SimulationParameters
{
	qreal MUTATION_PROB = 0.0;
	qreal CRIT_CELL_DIST_MIN = 0.0;
	qreal CRIT_CELL_DIST_MAX = 0.0;
	qreal INTERNAL_TO_KINETIC_ENERGY = 0.0; //related to 1/mass
	qreal CELL_MAX_FORCE = 0.0;
	qreal CELL_MAX_FORCE_DECAY_PROB = 0.0;
	int MAX_CELL_CONNECTIONS = 0;
	int CELL_TOKENSTACKSIZE = 0;
	int MAX_TOKEN_ACCESS_NUMBERS = 0;
	qreal NEW_CELL_ENERGY = 0.0;
	int NEW_CELL_MAX_CONNECTION = 0;	//TODO: add to editor
	int NEW_CELL_TOKEN_ACCESS_NUMBER = 0; //TODO: add to editor
	qreal CRIT_CELL_TRANSFORM_ENERGY = 0.0;
	qreal CELL_TRANSFORM_PROB = 0.0;
	qreal CLUSTER_FUSION_VEL = 0.0;

	int CELL_NUM_INSTR = 0;
	int CELL_MEMSIZE = 0;
	int TOKEN_MEMSIZE = 0;
	qreal CELL_WEAPON_STRENGTH = 0.0;
	qreal CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST = 0.0;
	qreal CELL_FUNCTION_SENSOR_RANGE = 0.0;
	qreal CELL_FUNCTION_COMMUNICATOR_RANGE = 0.0;

	qreal NEW_TOKEN_ENERGY = 0.0;
	qreal MIN_TOKEN_ENERGY = 0.0;

	qreal RAD_EXPONENT = 0.0;
	qreal RAD_FACTOR = 0.0;
	qreal RAD_PROBABILITY = 0.0;

	qreal CELL_RAD_ENERGY_VEL_MULT = 0.0;
	qreal CELL_RAD_ENERGY_VEL_PERTURB = 0.0;

	void setParameters(SimulationParameters* other);

	void serializePrimitives(QDataStream& stream);
	void deserializePrimitives(QDataStream& stream);
};

#endif // SIMULATIONPARAMETERS_H

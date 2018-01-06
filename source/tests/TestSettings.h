#ifndef SETTINGS_H
#define SETTINGS_H

#include <QtGlobal>
#include <QString>

const double FLOATINGPOINT_MEDIUM_PRECISION = 1.0e-4;
const double FLOATINGPOINT_LOW_PRECISION = 1.0e-1f;

const QString INITIAL_DATA_FILENAME = "tests/testdata/comparison/initial.sim";
const QString COMPUTATION_DATA_FILENAME = "tests/testdata/comparison/computation.dat";
const bool COMPUTATION_DATA_UPDATE = true;
const int COMPUTATION_DATA_TIMESTEPS = 500;

const QString REPLICATOR_DATA_FILENAME = "tests/testdata/replicator/initial.sim";
const int REPLICATOR_DATA_TIMESTEPS = 5000;

#endif // SETTINGS_H

#ifndef SETTINGS_H
#define SETTINGS_H

#include <QtGlobal>
#include <QString>

const qreal TEST_MEDIUM_PRECISION = 1.0e-5;
const qreal TEST_LOW_PRECISION = 1.0e-1;

const QString INTEGRATIONTEST_COMPARISON_INIT = "../../source/tests/testdata/comparison/initial.sim";
const QString INTEGRATIONTEST_COMPARISON_REF = "../../source/tests/testdata/comparison/computation.dat";
const bool INTEGRATIONTEST_COMPARISON_UPDATE_REF = true;
const int INTEGRATIONTEST_COMPARISON_TIMESTEPS = 500;

const QString INTEGRATIONTEST_REPLICATOR_INIT = "../../source/tests/testdata/replicator/initial.sim";
const int INTEGRATIONTEST_REPLICATOR_TIMESTEPS = 5000;

#endif // SETTINGS_H

#ifndef TESTSETTINGS_H
#define TESTSETTINGS_H

#include <QtGlobal>
#include <QString>

const qreal TEST_REAL_PRECISION = 1.0e-5;
const qreal TEST_LOW_REAL_PRECISION = 1.0e-1;

const QString INTEGRATIONTEST_COMPARISON_INIT = "../source/testdata/comparison/initial.sim";
const QString INTEGRATIONTEST_COMPARISON_REF = "../source/testdata/comparison/computation.dat";
const bool INTEGRATIONTEST_COMPARISON_UPDATE_REF = true;
const int INTEGRATIONTEST_COMPARISON_TIMESTEPS = 1000;

const QString INTEGRATIONTEST_REPLICATOR_INIT = "../source/testdata/replicator/initial.sim";
const int INTEGRATIONTEST_REPLICATOR_TIMESTEPS = 40000;

#endif // TESTSETTINGS_H

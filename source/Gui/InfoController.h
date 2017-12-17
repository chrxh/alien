#pragma once
#include <QLabel>

#include "Definitions.h"

class InfoController
	: public QObject
{
	Q_OBJECT

public:
	InfoController(QObject * parent = nullptr);
	virtual ~InfoController() = default;

	virtual void init(QLabel* infoLabel, MainController* mainController);

	virtual void increaseTimestep();
	virtual void setZoomFactor(double factor);

private:
	Q_SLOT void oneSecondTimerTimeout();

	void updateInfoLabel();

	QTimer* _oneSecondTimer = nullptr;
	QLabel* _infoLabel = nullptr;
	MainController* _mainController = nullptr;
	int _tpsCounting = 0;
	int _tps = 0;
	double _zoomFactor = 2.0;
};

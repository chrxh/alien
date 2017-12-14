#pragma once
#include <QLabel>

class InfoController
	: public QObject
{
	Q_OBJECT

public:
	InfoController(QObject * parent = nullptr);
	virtual ~InfoController() = default;

	virtual void init(QLabel* infoLabel);

	virtual void setTimestep(int timestep);
	virtual void increaseTimestep();
	virtual int getTimestep() const;

private:
	Q_SLOT void oneSecondTimerTimeout();

	void updateInfoLabel();

	QTimer* _oneSecondTimer = nullptr;
	QLabel* _infoLabel = nullptr;
	int _timestep = 0;
	int _tpsCounting = 0;
	int _tps = 0;
	double _magnification = 2.0;
};

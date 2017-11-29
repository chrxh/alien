#pragma once
#include <QObject>

#include "Definitions.h"

class MainController
	: public QObject
{
	Q_OBJECT
public:
	MainController(QObject * parent = nullptr);
	virtual ~MainController();

	void init();

private:
	MainView* _view = nullptr;
	MainModel* _model = nullptr;

};
